"""Scalar or categorical DIME-style critic training with OptiQ distillation."""

from functools import partial
from pathlib import Path
from typing import ClassVar

import flax
import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

from common.type_aliases import ReplayBufferSamplesNp, RLTrainState
from diffusion.dime import DIME

from .policy import OptiQPolicy
from .behavior import parse_behavior_best_of_k, select_best_of_k
from .winner_distillation import update_winner_actor
from .bestk_proposal import make_bestk_proposal
from .temperature import parse_temperature_schedule, scheduled_temperature
from .critic_utils import critic_expectation
from .distillation import conditional_ot_nll, hard_projection_mass_error
from .soft_improvement import sampled_soft_update
from .proximal import proximal_policy_weights
from .latent import FiniteMixtureTrainState, stratified_finite_latents
from .semi_implicit import (
    ConditionalGaussianProposal, PretanhTeacherKDE,
    conditional_mixture_log_prob, idac_action_and_log_density,
)
from .transport import (
    TruncatedGaussianKDE,
    clip_action,
    sample_truncated_gaussian,
    sample_truncated_gaussian_mixture,
    select_density_beta_for_ess,
    sinkhorn,
)


class OptiQDIME(DIME):
    """One-step OptiQ actor with scalar or categorical CrossQ critics."""

    policy_aliases: ClassVar[dict[str, type[OptiQPolicy]]] = {
        "MlpPolicy": OptiQPolicy,
        "MultiInputPolicy": OptiQPolicy,
    }
    policy: OptiQPolicy

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # DIME defaults to live-network CrossQ; scalar defaults select a
        # conventional target critic explicitly without changing legacy runs.
        self.crossq_style = bool(self.cfg.alg.critic.get("crossq_style", True))
        self.behavior_uniform_probability = float(
            self.cfg.alg.get("behavior_uniform_probability", 0.0)
        )
        if not 0.0 <= self.behavior_uniform_probability <= 1.0:
            raise ValueError("behavior_uniform_probability must be between 0 and 1")
        self.behavior_rng = np.random.default_rng(
            np.random.SeedSequence([int(self.seed or 0), 510010])
        )
        self.behavior_uniform_count = 0
        self.behavior_action_count = 0
        self.behavior_best_of_k = parse_behavior_best_of_k(self.cfg.alg)
        self.behavior_best_of_k_key = jax.random.fold_in(
            jax.random.PRNGKey(int(self.seed or 0)), 580008
        )
        self.behavior_best_of_k_count = 0
        self.soft_guard_attempts = 0
        self.soft_guard_accepts = 0
        self.backup_mode = self.cfg.alg.critic.get(
            "backup_mode",
            "soft_td" if self.cfg.alg.actor.get("type") == "semi_implicit" else "td",
        )
        self.temperature_schedule = parse_temperature_schedule(
            self.cfg.alg.actor, self.backup_mode
        )

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        # Always consume the original actor/warmup RNG sequence first. This hook
        # changes collection only; evaluation and TD targets use predict/sample_action.
        action, buffer_action = super()._sample_action(
            learning_starts, action_noise, n_envs
        )
        if self.num_timesteps < learning_starts:
            return action, buffer_action

        k = getattr(self, "behavior_best_of_k", 1)
        if k > 1:
            if action_noise is not None:
                raise ValueError("best-of-k collection does not support additional action_noise")
            observations, _ = self.policy.prepare_obs(self._last_obs)
            self.behavior_best_of_k_key, selection_key = jax.random.split(self.behavior_best_of_k_key)
            chosen, metrics = select_best_of_k(
                self.policy.actor_state, self.policy.qf_state, observations,
                buffer_action, selection_key, k,
            )
            buffer_action = np.asarray(chosen, dtype=buffer_action.dtype)
            if not np.isfinite(buffer_action).all():
                raise FloatingPointError("Nonfinite best-of-k collection action")
            action = self.policy.unscale_action(buffer_action)
            self.behavior_best_of_k_count += n_envs
            self.logger.record("rollout/behavior_best_of_k", k)
            self.logger.record("rollout/best_of_k_action_count", self.behavior_best_of_k_count)
            for name, value in metrics.items():
                self.logger.record(f"rollout/{name}", float(value))

        probability = self.behavior_uniform_probability
        if probability == 0.0:
            return action, buffer_action

        # One Bernoulli decision per environment, replacing the entire action.
        selected = self.behavior_rng.random(n_envs) < probability
        uniform = self.behavior_rng.uniform(
            -1.0, 1.0, size=buffer_action.shape
        ).astype(buffer_action.dtype)
        mask = selected.reshape((n_envs,) + (1,) * (buffer_action.ndim - 1))
        buffer_action = np.where(mask, uniform, buffer_action)
        action = self.policy.unscale_action(buffer_action)

        self.behavior_uniform_count += int(selected.sum())
        self.behavior_action_count += n_envs
        self.logger.record("rollout/behavior_uniform_probability", probability)
        self.logger.record("rollout/behavior_uniform_count", self.behavior_uniform_count)
        self.logger.record("rollout/behavior_action_count", self.behavior_action_count)
        self.logger.record(
            "rollout/behavior_uniform_fraction",
            self.behavior_uniform_count / self.behavior_action_count,
        )
        return action, buffer_action

    def train(self, batch_size, gradient_steps):
        data = self.replay_buffer.sample(
            batch_size * gradient_steps, env=self._vec_normalize_env
        )
        actor_learning_starts = self.cfg.alg.actor.get("learning_starts", self.learning_starts)
        policy_delay_indices = {
            i: True
            for i in range(gradient_steps)
            if self.num_timesteps > actor_learning_starts
            and ((self._n_updates + i + 1) % self.policy_delay) == 0
        }
        policy_delay_indices = flax.core.FrozenDict(policy_delay_indices)

        if isinstance(data.observations, dict):
            keys = list(self.observation_space.keys())
            obs = np.concatenate(
                [data.observations[key].numpy() for key in keys], axis=1
            )
            next_obs = np.concatenate(
                [data.next_observations[key].numpy() for key in keys], axis=1
            )
        else:
            obs = data.observations.numpy()
            next_obs = data.next_observations.numpy()

        data = ReplayBufferSamplesNp(
            obs,
            data.actions.numpy(),
            next_obs,
            data.dones.numpy().flatten(),
            data.rewards.numpy().flatten(),
        )
        actor = self.cfg.alg.actor
        temperature = 0.0 if actor.get("teacher_distribution") == "best_of_k_winners" else actor.temperature
        schedule_metrics = {}
        if self.temperature_schedule is not None:
            temperature, progress = scheduled_temperature(
                temperature, self.temperature_schedule, self.num_timesteps, self.learning_starts
            )
            schedule_metrics = {
                "temperature": temperature,
                "temperature_anneal_progress": progress,
                "temperature_schedule_enabled": 1.0,
                "temperature_final_target": self.temperature_schedule.final_temperature,
                "temperature_env_steps": self.num_timesteps,
            }
        guard = actor.get("soft_guard", {})
        guard_enabled = bool(guard.get("enabled", False))
        validation_observations = None
        if guard_enabled:
            # Separate replay draw: these states do not select the OT teacher
            # or its supervised gradient. The disabled path consumes no RNG.
            validation_data = self.replay_buffer.sample(
                int(guard.get("batch_size", 32)) * gradient_steps,
                env=self._vec_normalize_env,
            )
            if isinstance(validation_data.observations, dict):
                validation_observations = np.concatenate(
                    [validation_data.observations[k].numpy()
                     for k in self.observation_space.keys()], axis=1,
                )
            else:
                validation_observations = validation_data.observations.numpy()
        (
            self.policy.qf_state,
            self.policy.actor_state,
            self.policy.target_actor_state,
            self.ent_coef_state,
            self.key,
            log_metrics,
        ) = self._train(
            self.crossq_style,
            self.use_bnstats_from_live_net,
            self.gamma,
            self.tau,
            self.policy_tau,
            gradient_steps,
            data,
            policy_delay_indices,
            self.policy.qf_state,
            self.policy.actor_state,
            self.policy.target_actor_state,
            self.ent_coef_state,
            self.key,
            self.num_timesteps,
            self.cfg.alg.critic.v_min,
            self.cfg.alg.critic.v_max,
            self.cfg.alg.critic.entr_coeff,
            self.cfg.alg.critic.n_atoms,
            actor.num_policy_samples,
            actor.proposals_per_policy_sample,
            actor.proposal_sampling_mode,
            actor.proposal_std,
            actor.proposal_clip,
            actor.include_anchor,
            actor.density_correction,
            actor.density_beta,
            actor.adaptive_density_beta,
            actor.minimum_source_ess,
            actor.density_beta_grid_size,
            temperature,
            actor.sinkhorn_epsilon,
            actor.sinkhorn_iterations,
            actor.source_q_eval,
            actor.transport_target_mode,
            actor.td_noise_std,
            actor.td_noise_clip,
            actor.get("type", "implicit") == "semi_implicit",
            int(actor.get("entropy_samples", 16)),
            bool(actor.get("normalize_ot_cost", True)),
            actor.get("distillation_loss", "pointwise_mse"),
            actor.get("teacher_distribution", "realized_kde"),
            guard_enabled,
            validation_observations,
            int(guard.get("components", 16)),
            int(guard.get("draws", 8)),
            float(guard.get("standard_error_multiplier", 2.0)),
            float(actor.get("soft_proximal_ess_fraction", 0.0)),
            self.backup_mode,
            bool(actor.get("entropy_diagnostics", True)),
            actor.get("ot_student_action", "sample"),
            int(actor.get("teacher_best_of_k", 1)),
            int(actor.get("proposal_best_of_k", 1)),
            float(actor.get("proposal_guided_fraction", 0.0)),
        )
        self._n_updates += gradient_steps
        if guard_enabled:
            self.soft_guard_attempts += int(log_metrics["soft_guard_attempts"])
            self.soft_guard_accepts += int(log_metrics["soft_guard_accepts"])
            self.logger.record("train/soft_guard_acceptance_cumulative",
                self.soft_guard_accepts / max(self.soft_guard_attempts, 1))
            self.logger.record("train/soft_guard_attempts_cumulative", self.soft_guard_attempts)

        checkpoint_due = self.model_save_path is not None and (
            self.num_timesteps % self.save_every_n_steps == 0
            or self.num_timesteps == self.learning_starts + 1
        )
        if checkpoint_due:
            self._save_model()
            # Keep one fixed batch of states for comparing the same local Q
            # landscape across all later actor/critic checkpoints.  The 5k
            # warm-up save is deliberately excluded so the probe comes from
            # the first regular checkpoint's replay distribution.
            if self.num_timesteps >= self.save_every_n_steps:
                probe_path = Path(self.model_save_path) / "landscape_probe_batch.npz"
                if not probe_path.exists():
                    probe_size = min(128, data.observations.shape[0])
                    np.savez_compressed(
                        probe_path,
                        observations=data.observations[:probe_size],
                        actions=data.actions[:probe_size],
                        rewards=data.rewards[:probe_size],
                        dones=data.dones[:probe_size],
                        source_step=np.asarray(self.num_timesteps, dtype=np.int64),
                    )

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/actor_updates", int(self.policy.actor_state.step))
        diagnostic_interval = int(self.cfg.get("diagnostic_interval", 0))
        diagnostic_due = diagnostic_interval > 0 and self.num_timesteps % diagnostic_interval == 0
        core_metrics = {"actor_loss", "critic_loss", "current_q_values", "next_q_values",
                        "entrQ_1", "entrQ_2", "ent_coef", "backup_entropy_lower",
                        "backup_entropy_term", "policy_entropy_lower", "actor_std_mean"}
        # The exact scalar supplied to the dynamic JIT argument is recorded at
        # every logger flush. The saved config keeps the initial value/schedule.
        log_metrics = dict(log_metrics, **schedule_metrics)
        core_metrics.update(schedule_metrics)
        if actor.get("proposal_best_of_k", 1) > 1:
            core_metrics.update({"proposal_best_of_k", "proposal_guided_fraction",
                "proposal_pilot_count", "proposal_pilot_winner_q_gain",
                "source_ess_absolute", "temperature", "density_beta_mean"})
        if actor.get("teacher_distribution") == "best_of_k_winners":
            core_metrics.update({"teacher_best_of_k", "teacher_winner_count",
                "teacher_candidate_count", "teacher_uniform_mass", "teacher_winner_q_gain"})
        anchor_metrics = {"local_best_q_gain_over_anchor", "local_improvement_fraction",
                          "local_anchor_argmax_fraction", "twin_local_delta_correlation",
                          "twin_local_improvement_sign_agreement"}
        for key, value in log_metrics.items():
            if diagnostic_interval and not diagnostic_due and key not in core_metrics:
                continue
            if not actor.include_anchor and key in anchor_metrics:
                continue  # These metrics require candidate anchors, absent in this baseline.
            try:
                value = value.item()
            except (AttributeError, ValueError):
                pass
            self.logger.record(f"train/{key}", value)
        if diagnostic_due:
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

    @staticmethod
    @partial(
        jax.jit,
        static_argnames=[
            "crossq_style",
            "use_bnstats_from_live_net",
            "num_atoms",
            "v_min",
            "v_max",
            "entr_coeff",
            "semi_implicit",
            "entropy_samples",
            "backup_mode",
        ],
    )
    def update_critic(
        crossq_style: bool,
        use_bnstats_from_live_net: bool,
        gamma: float,
        target_actor_state: TrainState,
        qf_state: RLTrainState,
        observations: np.ndarray,
        actions: np.ndarray,
        next_observations: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        num_atoms: int,
        z_atoms: jax.Array,
        v_min: int,
        v_max: int,
        entr_coeff: float,
        td_noise_std: float,
        td_noise_clip: float,
        key,
        semi_implicit: bool = False,
        entropy_samples: int = 16,
        temperature: float = 0.25,
        backup_mode: str | None = None,
    ):
        # The policy family and Bellman objective are independent. None preserves
        # the historical v2/implicit defaults for callers without an explicit mode.
        backup_mode = backup_mode or ("soft_td" if semi_implicit else "td")
        if backup_mode not in {"td", "soft_td"}:
            raise ValueError("backup_mode must be td or soft_td")
        if backup_mode == "soft_td" and not semi_implicit:
            raise ValueError("Soft TD requires a conditional Gaussian policy")
        soft_backup = backup_mode == "soft_td"
        (
            key,
            actor_key,
            noise_key,
            dropout_key_target,
            dropout_key_current,
            redq_key,
        ) = jax.random.split(key, 6)
        if soft_backup:
            # _train passes the CURRENT actor here. All M components, including
            # the action's generating component, come from that same actor.
            next_actions, next_log_density = idac_action_and_log_density(
                target_actor_state, next_observations, actor_key, entropy_samples
            )
            entropy_adjustment = jax.lax.stop_gradient(-temperature * next_log_density)
        else:
            next_actions = OptiQPolicy.sample_action(
                target_actor_state, next_observations, actor_key, deterministic=False
            )
            if not semi_implicit:
                next_actions = sample_truncated_gaussian(
                    noise_key, next_actions, repeats=1, std=td_noise_std,
                    perturb_clip=td_noise_clip, include_anchor=False,
                )[:, 0]
            entropy_adjustment = jnp.zeros_like(rewards)
        next_actions = jax.lax.stop_gradient(next_actions)

        def critic_loss(params, batch_stats, dropout_key):
            if not crossq_style:
                next_q_values = qf_state.apply_fn(
                    {
                        "params": qf_state.target_params,
                        "batch_stats": (
                            batch_stats
                            if use_bnstats_from_live_net
                            else qf_state.target_batch_stats
                        ),
                    },
                    next_observations,
                    next_actions,
                    rngs={"dropout": dropout_key_target},
                    train=False,
                )
                current_q_values, state_updates = qf_state.apply_fn(
                    {"params": params, "batch_stats": batch_stats},
                    observations,
                    actions,
                    rngs={"dropout": dropout_key},
                    mutable=["batch_stats"],
                    train=True,
                )
            else:
                combined_q_values, state_updates = qf_state.apply_fn(
                    {"params": params, "batch_stats": batch_stats},
                    jnp.concatenate([observations, next_observations], axis=0),
                    jnp.concatenate([actions, next_actions], axis=0),
                    rngs={"dropout": dropout_key},
                    mutable=["batch_stats"],
                    train=True,
                )
                current_q_values, next_q_values = jnp.split(
                    combined_q_values, 2, axis=1
                )

            if next_q_values.shape[0] > 2:
                next_q_values = jax.random.choice(
                    redq_key,
                    next_q_values,
                    (2,),
                    replace=False,
                    axis=0,
                )

            if num_atoms == 1:
                # Conventional clipped double-Q backup for scalar critics.
                # Scalar Q has no categorical support clipping or categorical
                # entropy regularizer; v2 policy entropy enters its TD target.
                next_q = next_q_values[..., 0].min(axis=0)
                target_q = jax.lax.stop_gradient(
                    rewards + (1.0 - dones) * gamma * (next_q + entropy_adjustment)
                )
                current_q = current_q_values[..., 0]
                loss = jnp.square(current_q - target_q[None, :]).mean(axis=1).sum()
                zero = jnp.asarray(0.0, dtype=current_q.dtype)
                return loss, (
                    state_updates, current_q.min(axis=0).mean(),
                    target_q.mean(), zero, zero,
                )

            def projection(next_dist):
                delta_z = (v_max - v_min) / (num_atoms - 1)
                target_z = jnp.clip(
                    rewards[:, None] + (1.0 - dones[:, None]) * gamma
                    * (z_atoms + entropy_adjustment[:, None]),
                    a_min=v_min,
                    a_max=v_max,
                )
                b = (target_z - v_min) / delta_z
                lower = jnp.floor(b).astype(jnp.int32)
                upper = jnp.ceil(b).astype(jnp.int32)
                lower = jnp.where((upper > 0) & (lower == upper), lower - 1, lower)
                upper = jnp.where(
                    (lower < num_atoms - 1) & (lower == upper), upper + 1, upper
                )
                offset = jnp.arange(rewards.shape[0])[:, None] * num_atoms
                projected = jnp.zeros_like(next_dist).ravel()
                projected = projected.at[(lower + offset).ravel()].add(
                    (next_dist * (upper.astype(jnp.float32) - b)).ravel()
                )
                projected = projected.at[(upper + offset).ravel()].add(
                    (next_dist * (b - lower.astype(jnp.float32))).ravel()
                )
                return projected.reshape(rewards.shape[0], num_atoms)

            target_q1 = projection(next_q_values[0])
            target_q2 = projection(next_q_values[1])
            target_distribution = jax.lax.stop_gradient(
                jnp.mean(jnp.stack([target_q1, target_q2]), axis=0)
            )

            def cross_entropy(prediction):
                return -jnp.mean(
                    jnp.sum(
                        target_distribution * jnp.log(prediction + 1.0e-15), axis=-1
                    )
                ) + entr_coeff * jnp.mean(
                    jnp.sum(prediction * jnp.log(prediction + 1.0e-15), axis=-1)
                )

            loss = cross_entropy(current_q_values[0]) + cross_entropy(
                current_q_values[1]
            )
            current_q = jnp.sum(current_q_values * z_atoms, axis=-1)
            current_q = jnp.min(current_q, axis=0)
            entropy_1 = -jnp.mean(
                jnp.sum(
                    current_q_values[0] * jnp.log(current_q_values[0] + 1.0e-15),
                    axis=-1,
                )
            )
            entropy_2 = -jnp.mean(
                jnp.sum(
                    current_q_values[1] * jnp.log(current_q_values[1] + 1.0e-15),
                    axis=-1,
                )
            )
            target_q = jnp.sum(target_distribution * z_atoms, axis=-1)
            return loss, (
                state_updates,
                current_q.mean(),
                target_q.mean(),
                entropy_1,
                entropy_2,
            )

        (loss, aux), grads = jax.value_and_grad(critic_loss, has_aux=True)(
            qf_state.params, qf_state.batch_stats, dropout_key_current
        )
        state_updates, current_q, target_q, entropy_1, entropy_2 = aux
        qf_state = qf_state.apply_gradients(grads=grads)
        qf_state = qf_state.replace(batch_stats=state_updates.get("batch_stats", qf_state.batch_stats))
        metrics = {
            "critic_loss": loss,
            "current_q_values": current_q,
            "next_q_values": target_q,
            "entrQ_1": entropy_1,
            "entrQ_2": entropy_2,
            "ent_coef": jnp.asarray(temperature if soft_backup else 0.0),
        }
        if soft_backup:
            metrics.update(
                backup_entropy_lower=-next_log_density.mean(),
                backup_policy_density_exact=jnp.asarray(isinstance(target_actor_state, FiniteMixtureTrainState), dtype=jnp.float32),
                backup_entropy_term=entropy_adjustment.mean(),
                backup_discounted_entropy_term=((1.0 - dones) * gamma * entropy_adjustment).mean(),
                backup_entropy_std=next_log_density.std(),
                reward_mean=rewards.mean(),
                reward_abs_mean=jnp.abs(rewards).mean(),
                backup_action_saturation_fraction=jnp.mean(jnp.abs(next_actions) > 0.99),
            )
        elif semi_implicit:
            metrics.update(
                backup_entropy_term=jnp.asarray(0.0),
                backup_discounted_entropy_term=jnp.asarray(0.0),
                reward_mean=rewards.mean(),
                reward_abs_mean=jnp.abs(rewards).mean(),
                backup_action_saturation_fraction=jnp.mean(jnp.abs(next_actions) > 0.99),
            )
        return qf_state, metrics, key

    @staticmethod
    @partial(
        jax.jit,
        static_argnames=[
            "num_policy_samples",
            "proposals_per_policy_sample",
            "proposal_sampling_mode",
            "include_anchor",
            "density_correction",
            "adaptive_density_beta",
            "density_beta_grid_size",
            "sinkhorn_iterations",
            "source_q_eval",
            "transport_target_mode",
            "semi_implicit",
            "normalize_ot_cost",
            "distillation_loss",
            "teacher_distribution",
            "soft_proximal_ess_fraction",
            "entropy_diagnostics",
            "ot_student_action",
            "teacher_best_of_k",
            "proposal_best_of_k",
        ],
    )
    def update_actor(
        actor_state: TrainState,
        qf_state: RLTrainState,
        observations: np.ndarray,
        key,
        z_atoms: jax.Array,
        num_policy_samples: int,
        proposals_per_policy_sample: int,
        proposal_sampling_mode: str,
        proposal_std: float,
        proposal_clip: float,
        include_anchor: bool,
        density_correction: bool,
        density_beta: float,
        adaptive_density_beta: bool,
        minimum_source_ess: float,
        density_beta_grid_size: int,
        temperature: float,
        sinkhorn_epsilon: float,
        sinkhorn_iterations: int,
        source_q_eval: str,
        transport_target_mode: str,
        semi_implicit: bool = False,
        normalize_ot_cost: bool = True,
        distillation_loss: str = "pointwise_mse",
        teacher_distribution: str = "realized_kde",
        soft_proximal_ess_fraction: float = 0.0,
        entropy_diagnostics: bool = True,
        ot_student_action: str = "sample",
        teacher_best_of_k: int = 1,
        proposal_best_of_k: int = 1,
        proposal_guided_fraction: float = 0.0,
    ):
        if teacher_distribution == "best_of_k_winners":
            if (not semi_implicit or ot_student_action != "mean"
                or distillation_loss != "conditional_ot_nll" or normalize_ot_cost
                or density_correction or include_anchor or adaptive_density_beta
                or soft_proximal_ess_fraction > 0 or source_q_eval != "min"
                or isinstance(actor_state, FiniteMixtureTrainState) or teacher_best_of_k < 1):
                raise ValueError("Winner OT requires continuous mean-action full NLL and uniform winner mass")
            return update_winner_actor(actor_state, qf_state, observations, key,
                num_policy_samples, proposals_per_policy_sample, teacher_best_of_k,
                sinkhorn_epsilon, sinkhorn_iterations)
        if ot_student_action not in {"sample", "mean"}:
            raise ValueError("ot_student_action must be sample or mean")
        if ot_student_action == "mean" and (
            not semi_implicit or teacher_distribution != "conditional_mixture"
            or distillation_loss != "conditional_ot_nll"
        ):
            raise ValueError("Mean-action OT requires a conditional-mixture teacher and conditional OT NLL")
        if soft_proximal_ess_fraction > 0 and (
            not isinstance(actor_state, FiniteMixtureTrainState)
            or not semi_implicit or teacher_distribution != "conditional_mixture"
            or not density_correction or include_anchor or adaptive_density_beta
        ):
            raise ValueError("Proximal extraction requires the full actual finite policy proposal")
        key, latent_key, proposal_key, dropout_key = jax.random.split(key, 4)
        batch_size, observation_dim = observations.shape

        def actor_loss(actor_params):
            output_layer = "mu" if semi_implicit else f"Dense_{len(actor_params) - 1}"
            action_dim = actor_params[output_layer]["bias"].shape[0]
            z_key, eps_key = jax.random.split(latent_key) if semi_implicit else (latent_key, latent_key)
            if isinstance(actor_state, FiniteMixtureTrainState):
                latents = stratified_finite_latents(actor_state, batch_size, num_policy_samples,
                                                    action_dim, observations.dtype)
            else:
                latents = jax.random.normal(
                    z_key,
                    (batch_size, num_policy_samples, action_dim),
                    dtype=observations.dtype,
                )
            repeated_observations = jnp.broadcast_to(
                observations[:, None, :],
                (batch_size, num_policy_samples, observation_dim),
            )
            actor_output = actor_state.apply_fn(
                {"params": actor_params},
                repeated_observations.reshape(batch_size * num_policy_samples, -1),
                latents.reshape(batch_size * num_policy_samples, action_dim),
            )
            if semi_implicit:
                mu, log_std = [x.reshape(batch_size, num_policy_samples, action_dim) for x in actor_output]
                # Same latent/noise realization is used for the student forward
                # value and its gradient; teacher construction is stopped below.
                student_noise = jax.random.normal(
                    eps_key, mu.shape, dtype=mu.dtype
                )
                student_u = mu + jnp.exp(log_std) * student_noise
                raw_actions = policy_samples = jnp.tanh(student_u)
            else:
                raw_actions = actor_output.reshape(batch_size, num_policy_samples, action_dim)
                policy_samples = clip_action(raw_actions)
            proposal_metrics = {}
            teacher_draw_key = proposal_key
            # Define the KDE from actor centers BEFORE drawing its random candidates.
            # Density evaluation below uses this very same distribution, never a
            # KDE fitted to the newly drawn candidate cloud.
            if semi_implicit:
                if include_anchor:
                    raise ValueError("Semi-implicit teacher candidates must not contain anchors")
                if teacher_distribution == "conditional_mixture":
                    proposal_kde = ConditionalGaussianProposal(
                        jax.lax.stop_gradient(mu), jax.lax.stop_gradient(log_std),
                        # Proximal IS requires q == pi_old, with no scale floor.
                        0.0 if soft_proximal_ess_fraction > 0 else proposal_std)
                else:
                    proposal_kde = PretanhTeacherKDE(jax.lax.stop_gradient(student_u), proposal_std)
                if proposal_best_of_k > 1:
                    if (teacher_distribution != "conditional_mixture" or not density_correction
                        or include_anchor or adaptive_density_beta or soft_proximal_ess_fraction > 0):
                        raise ValueError("Best-k proposals require conditional-mixture teacher and full density correction")
                    pilot_key, teacher_draw_key = jax.random.split(proposal_key)
                    proposal_kde, proposal_metrics = make_bestk_proposal(
                        proposal_kde, qf_state, observations, pilot_key,
                        proposal_best_of_k, proposal_guided_fraction, source_q_eval)
                proposals, proposal_u, proposal_component_indices = proposal_kde.sample(
                    teacher_draw_key, proposals_per_policy_sample, proposal_sampling_mode
                )
            else:
                proposal_kde = TruncatedGaussianKDE.from_centers(
                    jax.lax.stop_gradient(policy_samples), proposal_std, proposal_clip
                )
            if not semi_implicit and proposal_sampling_mode == "stratified":
                sampled = proposal_kde.sample_stratified(
                    proposal_key, proposals_per_policy_sample, include_anchor
                )
            elif not semi_implicit and proposal_sampling_mode == "exact":
                sampled = sample_truncated_gaussian_mixture(
                    proposal_key, proposal_kde.centers, proposals_per_policy_sample,
                    proposal_std, proposal_clip, include_anchor, return_component_indices=True,
                )
            elif not semi_implicit:
                raise ValueError(
                    f"Unknown proposal_sampling_mode: {proposal_sampling_mode}"
                )
            if semi_implicit:
                proposals = jax.lax.stop_gradient(proposals)
            elif proposal_sampling_mode == "exact":
                proposals, proposal_component_indices = sampled
            else:
                proposals = sampled
                proposal_component_indices = jnp.broadcast_to(
                    jnp.arange(num_policy_samples)[None, :, None], proposals.shape[:-1]
                )
            proposal_component_indices = proposal_component_indices.reshape(batch_size, -1)
            proposals = proposals.reshape(
                batch_size,
                num_policy_samples * proposals_per_policy_sample,
                action_dim,
            )
            num_proposals = proposals.shape[1]
            proposal_observations = jnp.broadcast_to(
                observations[:, None, :],
                (batch_size, num_proposals, observation_dim),
            )
            source_distributions = qf_state.apply_fn(
                {"params": qf_state.params, "batch_stats": qf_state.batch_stats},
                proposal_observations.reshape(batch_size * num_proposals, -1),
                proposals.reshape(batch_size * num_proposals, action_dim),
                rngs={"dropout": dropout_key},
                train=False,
            ).reshape(2, batch_size, num_proposals, -1)
            source_qs = critic_expectation(source_distributions, z_atoms)
            if source_q_eval == "mean":
                source_q = source_qs.mean(axis=0)
            elif source_q_eval == "min":
                source_q = source_qs.min(axis=0)
            else:
                raise ValueError(f"Unknown source_q_eval: {source_q_eval}")
            source_q = jax.lax.stop_gradient(source_q)

            proposal_log_density = jnp.zeros_like(source_q)
            if density_correction:
                proposal_log_density = proposal_kde.log_prob(
                    jax.lax.stop_gradient(proposal_u if semi_implicit else proposals)
                )
            q_score = source_q / temperature
            density_score = -proposal_log_density
            if adaptive_density_beta and density_correction:
                selected_density_beta, selected_beta_ess = (
                    select_density_beta_for_ess(
                        q_score,
                        density_score,
                        minimum_source_ess,
                        density_beta_grid_size,
                    )
                )
            else:
                selected_density_beta = jnp.full(
                    (batch_size,), density_beta, dtype=q_score.dtype
                )
                fixed_logits = (
                    q_score + selected_density_beta[:, None] * density_score
                )
                fixed_weights = jax.nn.softmax(fixed_logits, axis=-1)
                selected_beta_ess = 1.0 / jnp.sum(
                    jnp.square(fixed_weights), axis=-1
                )
            selected_density_beta = jax.lax.stop_gradient(selected_density_beta)
            effective_density_score = selected_density_beta[:, None] * density_score
            logits = q_score + effective_density_score
            source_weights = jax.lax.stop_gradient(jax.nn.softmax(logits, axis=-1))
            if soft_proximal_ess_fraction > 0:
                full_step_ess = 1.0 / jnp.square(source_weights).sum(axis=-1)
                source_weights, proximal_fraction = proximal_policy_weights(
                    source_q, proposal_log_density, temperature,
                    soft_proximal_ess_fraction * num_proposals,
                )

            # v5 assigns each student by its mean action. Keep the original
            # epsilon draw, Gaussian teacher/density and full-row NLL unchanged.
            # Earlier profiles keep sample-action OT through the default mode.
            ot_policy_samples = jnp.tanh(mu) if ot_student_action == "mean" else policy_samples
            squared_costs = jnp.sum(
                jnp.square(ot_policy_samples[:, :, None, :] - proposals[:, None, :, :]),
                axis=-1,
            )
            costs = squared_costs / (
                squared_costs.mean(axis=(-2, -1), keepdims=True) + 1.0e-8
            ) if normalize_ot_cost else squared_costs
            transport = jax.lax.stop_gradient(
                sinkhorn(
                    costs,
                    source_weights,
                    sinkhorn_epsilon,
                    sinkhorn_iterations,
                )
            )
            row_distribution = transport / jnp.maximum(
                transport.sum(axis=-1, keepdims=True), 1.0e-20
            )
            if transport_target_mode == "argmax":
                selected_indices = jnp.argmax(row_distribution, axis=-1)
                selected_actions = jax.vmap(lambda actions, indices: actions[indices])(
                    proposals, selected_indices
                )
            elif transport_target_mode == "barycentric":
                selected_actions = jnp.einsum(
                    "bnm,bma->bna", row_distribution, proposals
                )
            else:
                raise ValueError(
                    f"Unknown transport_target_mode: {transport_target_mode}"
                )
            selected_actions = jax.lax.stop_gradient(selected_actions)
            if distillation_loss == "conditional_ot_nll":
                if not semi_implicit:
                    raise ValueError("Conditional OT likelihood requires a semi-implicit actor")
                loss = conditional_ot_nll(mu, log_std, proposal_u, row_distribution)
            else:
                loss = jnp.mean(
                    jnp.sum(jnp.square(raw_actions - selected_actions), axis=-1)
                )
            source_ess = 1.0 / jnp.sum(jnp.square(source_weights), axis=-1)
            q_only_weights = jax.nn.softmax(q_score, axis=-1)
            q_only_ess = 1.0 / jnp.sum(jnp.square(q_only_weights), axis=-1)
            density_only_weights = jax.nn.softmax(density_score, axis=-1)
            density_only_ess = 1.0 / jnp.sum(jnp.square(density_only_weights), axis=-1)
            effective_density_only_weights = jax.nn.softmax(
                effective_density_score, axis=-1
            )
            effective_density_only_ess = 1.0 / jnp.sum(
                jnp.square(effective_density_only_weights), axis=-1
            )
            centered_q = source_q - source_q.mean(axis=-1, keepdims=True)
            centered_density = density_score - density_score.mean(
                axis=-1, keepdims=True
            )
            q_density_correlation = jnp.mean(centered_q * centered_density, axis=-1) / (
                jnp.std(source_q, axis=-1) * jnp.std(density_score, axis=-1) + 1.0e-8
            )
            q_logit_std = jnp.std(q_score, axis=-1)
            density_logit_std = jnp.std(density_score, axis=-1)
            effective_density_logit_std = jnp.std(
                effective_density_score, axis=-1
            )

            def weighted_q_gain(weights, q_values):
                return jnp.sum(weights * q_values, axis=-1) - q_values.mean(axis=-1)

            q_only_q_gain = weighted_q_gain(q_only_weights, source_q)
            density_only_q_gain = weighted_q_gain(density_only_weights, source_q)
            full_q_gain = weighted_q_gain(source_weights, source_q)

            # Cross-evaluate selection by each twin critic with the other critic.
            # This is still not a fully independent estimate, but is less optimistic
            # than evaluating a Q-weighted selection with the exact same Q values.
            q1, q2 = source_qs[0], source_qs[1]
            q1_weights = jax.nn.softmax(
                q1 / temperature + effective_density_score, axis=-1
            )
            q2_weights = jax.nn.softmax(
                q2 / temperature + effective_density_score, axis=-1
            )
            cross_critic_q_gain = 0.5 * (
                weighted_q_gain(q1_weights, q2)
                + weighted_q_gain(q2_weights, q1)
            )

            # Group local diagnostics by the component that actually generated
            # each proposal. This is identical to the old layout reduction for
            # stratified draws and remains valid for IID mixture draws.
            component_mask = (
                proposal_component_indices[:, None, :]
                == jnp.arange(num_policy_samples)[None, :, None]
            )
            component_count = component_mask.sum(axis=-1)

            def grouped_values(values, fill_value):
                return jnp.where(component_mask, values[:, None, :], fill_value)

            grouped_q = grouped_values(source_q, -jnp.inf)
            component_max_q = grouped_q.max(axis=-1)
            component_min_q = grouped_values(source_q, jnp.inf).min(axis=-1)
            local_q_range = jnp.where(
                component_count > 0, component_max_q - component_min_q, 0.0
            )
            local_top_two = jax.lax.top_k(grouped_q, min(2, num_proposals))[0]
            local_q_top_gap = jnp.where(
                component_count > 1,
                local_top_two[..., 0] - local_top_two[..., -1],
                0.0,
            )

            grouped_q1 = grouped_values(q1, -jnp.inf)
            grouped_q2 = grouped_values(q2, -jnp.inf)
            q1_argmax = jnp.argmax(grouped_q1, axis=-1)
            q2_argmax = jnp.argmax(grouped_q2, axis=-1)
            valid_components = component_count > 0
            twin_local_argmax_agreement = jnp.sum(
                valid_components * (q1_argmax == q2_argmax)
            ) / jnp.maximum(valid_components.sum(), 1)

            local_best_q_gain = jnp.zeros_like(local_q_range)
            local_improvement_fraction = jnp.asarray(0.0)
            local_anchor_argmax_fraction = jnp.asarray(0.0)
            twin_local_delta_correlation = jnp.asarray(0.0)
            twin_local_improvement_sign_agreement = jnp.asarray(0.0)
            if include_anchor:
                anchor_indices = (
                    jnp.arange(num_policy_samples) * proposals_per_policy_sample
                )
                anchor_q = source_q[:, anchor_indices]
                anchor_q1 = q1[:, anchor_indices]
                anchor_q2 = q2[:, anchor_indices]
                local_best_q_gain = component_max_q - anchor_q
                local_anchor_argmax_fraction = jnp.mean(
                    jnp.argmax(grouped_q, axis=-1) == anchor_indices[None, :]
                )

                anchor_slot_mask = jnp.zeros(
                    (num_policy_samples, proposals_per_policy_sample), dtype=bool
                ).at[:, 0].set(True).reshape(num_proposals)
                random_mask = ~anchor_slot_mask[None, :]
                sample_anchor_q = jnp.take_along_axis(
                    anchor_q, proposal_component_indices, axis=1
                )
                sample_anchor_q1 = jnp.take_along_axis(
                    anchor_q1, proposal_component_indices, axis=1
                )
                sample_anchor_q2 = jnp.take_along_axis(
                    anchor_q2, proposal_component_indices, axis=1
                )
                local_improvement_fraction = jnp.sum(
                    random_mask * (source_q > sample_anchor_q)
                ) / jnp.maximum(random_mask.sum() * batch_size, 1)

                q1_deltas = q1 - sample_anchor_q1
                q2_deltas = q2 - sample_anchor_q2
                random_component_mask = component_mask & random_mask[:, None, :]
                random_count = random_component_mask.sum(axis=-1)
                safe_count = jnp.maximum(random_count, 1)
                q1_mean = jnp.sum(
                    random_component_mask * q1_deltas[:, None, :], axis=-1
                ) / safe_count
                q2_mean = jnp.sum(
                    random_component_mask * q2_deltas[:, None, :], axis=-1
                ) / safe_count
                centered_q1 = q1_deltas[:, None, :] - q1_mean[..., None]
                centered_q2 = q2_deltas[:, None, :] - q2_mean[..., None]
                covariance = jnp.sum(
                    random_component_mask * centered_q1 * centered_q2, axis=-1
                ) / safe_count
                variance_q1 = jnp.sum(
                    random_component_mask * jnp.square(centered_q1), axis=-1
                ) / safe_count
                variance_q2 = jnp.sum(
                    random_component_mask * jnp.square(centered_q2), axis=-1
                ) / safe_count
                correlations = covariance / (
                    jnp.sqrt(variance_q1 * variance_q2) + 1.0e-8
                )
                valid_correlations = random_count > 1
                twin_local_delta_correlation = jnp.sum(
                    jnp.where(valid_correlations, correlations, 0.0)
                ) / jnp.maximum(valid_correlations.sum(), 1)
                twin_local_improvement_sign_agreement = jnp.sum(
                    random_mask * ((q1_deltas > 0.0) == (q2_deltas > 0.0))
                ) / jnp.maximum(random_mask.sum() * batch_size, 1)
            metrics = {
                "actor_loss": loss,
                "source_ess_fraction": (source_ess / num_proposals).mean(),
                "source_ess_absolute": source_ess.mean(),
                "source_ess_min": source_ess.min(),
                "source_ess_target": jnp.asarray(
                    minimum_source_ess, dtype=source_q.dtype
                ),
                "q_only_ess_fraction": (q_only_ess / num_proposals).mean(),
                "density_only_ess_fraction": (density_only_ess / num_proposals).mean(),
                "effective_density_only_ess_fraction": (
                    effective_density_only_ess / num_proposals
                ).mean(),
                "density_beta_mean": selected_density_beta.mean(),
                "density_beta_min": selected_density_beta.min(),
                "density_beta_max": selected_density_beta.max(),
                "density_beta_at_one_fraction": jnp.mean(
                    selected_density_beta >= 1.0 - 1.0e-6
                ),
                "density_beta_infeasible_fraction": jnp.mean(
                    q_only_ess < minimum_source_ess
                ),
                "selected_beta_search_ess": selected_beta_ess.mean(),
                "max_source_weight": source_weights.max(axis=-1).mean(),
                "source_q_std": jnp.std(source_q, axis=-1).mean(),
                "neg_log_proposal_std": jnp.std(density_score, axis=-1).mean(),
                "q_logit_std": q_logit_std.mean(),
                "density_logit_std": density_logit_std.mean(),
                "effective_density_logit_std": effective_density_logit_std.mean(),
                "combined_logit_std": jnp.std(logits, axis=-1).mean(),
                "q_to_density_logit_std_ratio": (
                    q_logit_std / (density_logit_std + 1.0e-8)
                ).mean(),
                "q_to_effective_density_logit_std_ratio": (
                    q_logit_std / (effective_density_logit_std + 1.0e-8)
                ).mean(),
                "q_neglogq_correlation": q_density_correlation.mean(),
                "q_only_weighted_q_gain": q_only_q_gain.mean(),
                "density_only_weighted_q_gain": density_only_q_gain.mean(),
                "full_weighted_q_gain": full_q_gain.mean(),
                "cross_critic_weighted_q_gain": cross_critic_q_gain.mean(),
                "twin_q_abs_diff": jnp.abs(q1 - q2).mean(),
                "source_q_mean": source_q.mean(),
                "source_q_global_range": (
                    source_q.max(axis=-1) - source_q.min(axis=-1)
                ).mean(),
                "local_q_range": local_q_range.mean(),
                "local_q_top_gap": local_q_top_gap.mean(),
                "local_best_q_gain_over_anchor": local_best_q_gain.mean(),
                "local_improvement_fraction": local_improvement_fraction,
                "local_anchor_argmax_fraction": local_anchor_argmax_fraction,
                "twin_local_argmax_agreement": twin_local_argmax_agreement,
                "twin_local_delta_correlation": twin_local_delta_correlation,
                "twin_local_improvement_sign_agreement": (
                    twin_local_improvement_sign_agreement
                ),
                "selected_delta_l2": jnp.linalg.norm(
                    policy_samples - selected_actions, axis=-1
                ).mean(),
                "policy_spread_l2": jnp.linalg.norm(
                    policy_samples.std(axis=1), axis=-1
                ).mean(),
            }
            for label, counterfactual_temperature in (
                ("0p05", 0.05),
                ("0p1", 0.1),
                ("0p2", 0.2),
                ("0p25", 0.25),
                ("0p5", 0.5),
                ("1p0", 1.0),
            ):
                counterfactual_weights = jax.nn.softmax(
                    source_q / counterfactual_temperature + effective_density_score,
                    axis=-1,
                )
                counterfactual_ess = 1.0 / jnp.sum(
                    jnp.square(counterfactual_weights), axis=-1
                )
                metrics[f"counterfactual_ess_T{label}"] = (
                    counterfactual_ess / num_proposals
                ).mean()
            if semi_implicit:
                if entropy_diagnostics:
                    policy_log_g = conditional_mixture_log_prob(student_u, mu, log_std)
                    metrics.update(
                        policy_entropy_lower=-policy_log_g.mean(),
                        policy_entropy_lower_std=policy_log_g.std(),
                    )
                std = jnp.exp(log_std)
                between = jnp.var(mu, axis=1).sum(axis=-1)
                within = jnp.mean(std**2, axis=1).sum(axis=-1)
                metrics.update(
                    policy_density_exact=jnp.asarray(isinstance(actor_state, FiniteMixtureTrainState), dtype=jnp.float32),
                    actor_latent_mean_variance_fraction=jnp.mean(between/jnp.maximum(between+within, 1.e-20)),
                    actor_std_mean=std.mean(), actor_std_min=std.min(), actor_std_max=std.max(),
                    actor_log_std_mean=log_std.mean(), actor_log_std_min=log_std.min(),
                    actor_log_std_max=log_std.max(),
                    student_action_saturation_fraction=jnp.mean(jnp.abs(policy_samples) > 0.99),
                    teacher_action_saturation_fraction=jnp.mean(jnp.abs(proposals) > 0.99),
                    teacher_log_density_mean=proposal_log_density.mean(),
                    proposal_std_pretanh=jnp.asarray(0.0 if soft_proximal_ess_fraction > 0 else proposal_std),
                    temperature=jnp.asarray(temperature), ot_cost_mean=squared_costs.mean(),
                    ot_row_marginal_error=jnp.abs(transport.sum(axis=-1) - 1.0 / num_policy_samples).mean(),
                    ot_col_marginal_error=jnp.abs(transport.sum(axis=-2) - source_weights).mean(),
                    hard_projection_mass_tv=hard_projection_mass_error(row_distribution, source_weights),
                )
            if soft_proximal_ess_fraction > 0:
                metrics.update(
                    proximal_fraction_mean=proximal_fraction.mean(),
                    proximal_fraction_min=proximal_fraction.min(),
                    proximal_fraction_max=proximal_fraction.max(),
                    proximal_full_step_ess=full_step_ess.mean(),
                    proximal_target_ess=jnp.asarray(soft_proximal_ess_fraction*num_proposals),
                    proximal_actual_policy_proposal=jnp.asarray(1.0),
                )
            metrics.update(proposal_metrics)
            return loss, metrics

        (loss, metrics), grads = jax.value_and_grad(actor_loss, has_aux=True)(
            actor_state.params
        )
        actor_state = actor_state.apply_gradients(grads=grads)
        return actor_state, loss, key, metrics

    @classmethod
    @partial(
        jax.jit,
        static_argnames=[
            "cls",
            "crossq_style",
            "use_bnstats_from_live_net",
            "gradient_steps",
            "v_min",
            "v_max",
            "num_atoms",
            "entr_coeff",
            "num_policy_samples",
            "proposals_per_policy_sample",
            "proposal_sampling_mode",
            "include_anchor",
            "density_correction",
            "sinkhorn_iterations",
            "source_q_eval",
            "transport_target_mode",
            "adaptive_density_beta",
            "density_beta_grid_size",
            "semi_implicit",
            "entropy_samples",
            "normalize_ot_cost",
            "distillation_loss",
            "teacher_distribution",
            "soft_guard_enabled",
            "soft_guard_components",
            "soft_guard_draws",
            "soft_proximal_ess_fraction",
            "backup_mode",
            "entropy_diagnostics",
            "ot_student_action",
            "teacher_best_of_k",
            "proposal_best_of_k",
        ],
    )
    def _train(
        cls,
        crossq_style,
        use_bnstats_from_live_net,
        gamma,
        tau,
        policy_tau,
        gradient_steps,
        data,
        policy_delay_indices,
        qf_state,
        actor_state,
        target_actor_state,
        ent_coef_state,
        key,
        n_env_interacts,
        v_min,
        v_max,
        entr_coeff,
        num_atoms,
        num_policy_samples,
        proposals_per_policy_sample,
        proposal_sampling_mode,
        proposal_std,
        proposal_clip,
        include_anchor,
        density_correction,
        density_beta,
        adaptive_density_beta,
        minimum_source_ess,
        density_beta_grid_size,
        temperature,
        sinkhorn_epsilon,
        sinkhorn_iterations,
        source_q_eval,
        transport_target_mode,
        td_noise_std,
        td_noise_clip,
        semi_implicit=False,
        entropy_samples=16,
        normalize_ot_cost=True,
        distillation_loss="pointwise_mse",
        teacher_distribution="realized_kde",
        soft_guard_enabled=False,
        validation_observations=None,
        soft_guard_components=16,
        soft_guard_draws=8,
        soft_guard_standard_error_multiplier=2.0,
        soft_proximal_ess_fraction=0.0,
        backup_mode=None,
        entropy_diagnostics=True,
        ot_student_action="sample",
        teacher_best_of_k=1,
        proposal_best_of_k=1,
        proposal_guided_fraction=0.0,
    ):
        del n_env_interacts
        backup_mode = backup_mode or ("soft_td" if semi_implicit else "td")
        if backup_mode == "td" and soft_guard_enabled:
            raise ValueError("Plain TD must not use the soft-value acceptance guard")
        actor_metrics = {
            "actor_loss": jnp.asarray(0.0),
            "source_ess_fraction": jnp.asarray(0.0),
            "source_ess_absolute": jnp.asarray(0.0),
            "source_ess_min": jnp.asarray(0.0),
            "source_ess_target": jnp.asarray(0.0),
            "q_only_ess_fraction": jnp.asarray(0.0),
            "density_only_ess_fraction": jnp.asarray(0.0),
            "effective_density_only_ess_fraction": jnp.asarray(0.0),
            "density_beta_mean": jnp.asarray(0.0),
            "density_beta_min": jnp.asarray(0.0),
            "density_beta_max": jnp.asarray(0.0),
            "density_beta_at_one_fraction": jnp.asarray(0.0),
            "density_beta_infeasible_fraction": jnp.asarray(0.0),
            "selected_beta_search_ess": jnp.asarray(0.0),
            "max_source_weight": jnp.asarray(0.0),
            "source_q_std": jnp.asarray(0.0),
            "neg_log_proposal_std": jnp.asarray(0.0),
            "q_logit_std": jnp.asarray(0.0),
            "density_logit_std": jnp.asarray(0.0),
            "effective_density_logit_std": jnp.asarray(0.0),
            "combined_logit_std": jnp.asarray(0.0),
            "q_to_density_logit_std_ratio": jnp.asarray(0.0),
            "q_to_effective_density_logit_std_ratio": jnp.asarray(0.0),
            "q_neglogq_correlation": jnp.asarray(0.0),
            "q_only_weighted_q_gain": jnp.asarray(0.0),
            "density_only_weighted_q_gain": jnp.asarray(0.0),
            "full_weighted_q_gain": jnp.asarray(0.0),
            "cross_critic_weighted_q_gain": jnp.asarray(0.0),
            "twin_q_abs_diff": jnp.asarray(0.0),
            "source_q_mean": jnp.asarray(0.0),
            "source_q_global_range": jnp.asarray(0.0),
            "local_q_range": jnp.asarray(0.0),
            "local_q_top_gap": jnp.asarray(0.0),
            "local_best_q_gain_over_anchor": jnp.asarray(0.0),
            "local_improvement_fraction": jnp.asarray(0.0),
            "local_anchor_argmax_fraction": jnp.asarray(0.0),
            "twin_local_argmax_agreement": jnp.asarray(0.0),
            "twin_local_delta_correlation": jnp.asarray(0.0),
            "twin_local_improvement_sign_agreement": jnp.asarray(0.0),
            "selected_delta_l2": jnp.asarray(0.0),
            "policy_spread_l2": jnp.asarray(0.0),
        }
        for label in ("0p05", "0p1", "0p2", "0p25", "0p5", "1p0"):
            actor_metrics[f"counterfactual_ess_T{label}"] = jnp.asarray(0.0)
        if semi_implicit:
            if entropy_diagnostics:
                actor_metrics["policy_entropy_lower"] = jnp.asarray(0.0)
                actor_metrics["policy_entropy_lower_std"] = jnp.asarray(0.0)
            for name in ("actor_std_mean",
                         "actor_std_min", "actor_std_max", "actor_log_std_mean", "actor_log_std_min",
                         "actor_log_std_max", "student_action_saturation_fraction",
                         "teacher_action_saturation_fraction", "teacher_log_density_mean",
                         "proposal_std_pretanh", "temperature", "ot_cost_mean",
                         "ot_row_marginal_error", "ot_col_marginal_error"):
                actor_metrics[name] = jnp.asarray(0.0)
            actor_metrics["hard_projection_mass_tv"] = jnp.asarray(0.0)
        if teacher_distribution == "best_of_k_winners":
            # Do not publish inapplicable Boltzmann/density diagnostics as zeros.
            actor_metrics = {}
        guard_metrics = {}
        guard_attempts = jnp.asarray(0.0)
        guard_accepts = jnp.asarray(0.0)
        for i in range(gradient_steps):

            def slice_batch(array, step=i):
                batch_size = array.shape[0] // gradient_steps
                return array[batch_size * step : batch_size * (step + 1)]

            z_atoms = jnp.linspace(v_min, v_max, num_atoms)
            qf_state, critic_metrics, key = cls.update_critic(
                crossq_style,
                use_bnstats_from_live_net,
                gamma,
                actor_state if semi_implicit else target_actor_state,
                qf_state,
                slice_batch(data.observations),
                slice_batch(data.actions),
                slice_batch(data.next_observations),
                slice_batch(data.rewards),
                slice_batch(data.dones),
                num_atoms,
                z_atoms,
                v_min,
                v_max,
                entr_coeff,
                td_noise_std,
                td_noise_clip,
                key,
                semi_implicit,
                entropy_samples,
                temperature,
                backup_mode,
            )
            qf_state = cls.soft_update(tau, qf_state)
            if i in policy_delay_indices:
                old_actor_state = actor_state
                actor_state, _, key, actor_metrics = cls.update_actor(
                    actor_state,
                    qf_state,
                    slice_batch(data.observations),
                    key,
                    z_atoms,
                    num_policy_samples,
                    proposals_per_policy_sample,
                    proposal_sampling_mode,
                    proposal_std,
                    proposal_clip,
                    include_anchor,
                    density_correction,
                    density_beta,
                    adaptive_density_beta,
                    minimum_source_ess,
                    density_beta_grid_size,
                    temperature,
                    sinkhorn_epsilon,
                    sinkhorn_iterations,
                    source_q_eval,
                    transport_target_mode,
                    semi_implicit,
                    normalize_ot_cost,
                    distillation_loss,
                    teacher_distribution,
                    soft_proximal_ess_fraction,
                    entropy_diagnostics,
                    ot_student_action,
                    teacher_best_of_k,
                    proposal_best_of_k,
                    proposal_guided_fraction,
                )
                if soft_guard_enabled:
                    key, guard_key = jax.random.split(key)
                    actor_state, guard_metrics = sampled_soft_update(
                        old_actor_state, actor_state, qf_state,
                        slice_batch(validation_observations), guard_key,
                        temperature, z_atoms, soft_guard_components,
                        soft_guard_draws, soft_guard_standard_error_multiplier,
                    )
                    guard_attempts += 1
                    guard_accepts += guard_metrics["soft_guard_accepted"]
                target_actor_state = cls.soft_update_target_actor(
                    policy_tau, actor_state, target_actor_state
                )
        log_metrics = {**actor_metrics, **critic_metrics, **guard_metrics}
        if soft_guard_enabled:
            log_metrics.update(soft_guard_attempts=guard_attempts, soft_guard_accepts=guard_accepts)
        return (
            qf_state,
            actor_state,
            target_actor_state,
            ent_coef_state,
            key,
            log_metrics,
        )
