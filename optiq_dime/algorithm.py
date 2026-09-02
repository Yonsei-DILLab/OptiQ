"""DIME critic/replay/UTD training with OptiQ actor distillation."""

from functools import partial
from typing import ClassVar

import flax
import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

from common.type_aliases import ReplayBufferSamplesNp, RLTrainState
from diffusion.dime import DIME

from .policy import OptiQPolicy
from .transport import (
    clip_action,
    sample_truncated_gaussian,
    sinkhorn,
    truncated_mixture_log_density,
)


class OptiQDIME(DIME):
    """One-step OptiQ actor with DIME's distributional CrossQ critic."""

    policy_aliases: ClassVar[dict[str, type[OptiQPolicy]]] = {
        "MlpPolicy": OptiQPolicy,
        "MultiInputPolicy": OptiQPolicy,
    }
    policy: OptiQPolicy

    def train(self, batch_size, gradient_steps):
        data = self.replay_buffer.sample(
            batch_size * gradient_steps, env=self._vec_normalize_env
        )
        policy_delay_indices = {
            i: True
            for i in range(gradient_steps)
            if ((self._n_updates + i + 1) % self.policy_delay) == 0
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
            actor.proposal_std,
            actor.proposal_clip,
            actor.include_anchor,
            actor.density_correction,
            actor.temperature,
            actor.sinkhorn_epsilon,
            actor.sinkhorn_iterations,
            actor.source_q_eval,
            actor.transport_target_mode,
            actor.td_noise_std,
            actor.td_noise_clip,
        )
        self._n_updates += gradient_steps

        if self.model_save_path is not None and (
            self.num_timesteps % self.save_every_n_steps == 0
            or self.num_timesteps == self.learning_starts + 1
        ):
            self._save_model()

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        for key, value in log_metrics.items():
            try:
                value = value.item()
            except (AttributeError, ValueError):
                pass
            self.logger.record(f"train/{key}", value)

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
    ):
        (
            key,
            actor_key,
            noise_key,
            dropout_key_target,
            dropout_key_current,
            redq_key,
        ) = jax.random.split(key, 6)
        next_actions = OptiQPolicy.sample_action(
            target_actor_state, next_observations, actor_key, deterministic=False
        )
        next_actions = sample_truncated_gaussian(
            noise_key,
            next_actions,
            repeats=1,
            std=td_noise_std,
            perturb_clip=td_noise_clip,
            include_anchor=False,
        )[:, 0]
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

            def projection(next_dist):
                delta_z = (v_max - v_min) / (num_atoms - 1)
                target_z = jnp.clip(
                    rewards[:, None] + (1.0 - dones[:, None]) * gamma * z_atoms,
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
        qf_state = qf_state.replace(batch_stats=state_updates["batch_stats"])
        metrics = {
            "critic_loss": loss,
            "current_q_values": current_q,
            "next_q_values": target_q,
            "entrQ_1": entropy_1,
            "entrQ_2": entropy_2,
            "ent_coef": jnp.asarray(0.0),
        }
        return qf_state, metrics, key

    @staticmethod
    @partial(
        jax.jit,
        static_argnames=[
            "num_policy_samples",
            "proposals_per_policy_sample",
            "include_anchor",
            "density_correction",
            "sinkhorn_iterations",
            "source_q_eval",
            "transport_target_mode",
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
        proposal_std: float,
        proposal_clip: float,
        include_anchor: bool,
        density_correction: bool,
        temperature: float,
        sinkhorn_epsilon: float,
        sinkhorn_iterations: int,
        source_q_eval: str,
        transport_target_mode: str,
    ):
        key, latent_key, proposal_key, dropout_key = jax.random.split(key, 4)
        batch_size, observation_dim = observations.shape

        def actor_loss(actor_params):
            output_layer = f"Dense_{len(actor_params) - 1}"
            action_dim = actor_params[output_layer]["bias"].shape[0]
            latents = jax.random.normal(
                latent_key,
                (batch_size, num_policy_samples, action_dim),
                dtype=observations.dtype,
            )
            repeated_observations = jnp.broadcast_to(
                observations[:, None, :],
                (batch_size, num_policy_samples, observation_dim),
            )
            raw_actions = actor_state.apply_fn(
                {"params": actor_params},
                repeated_observations.reshape(batch_size * num_policy_samples, -1),
                latents.reshape(batch_size * num_policy_samples, action_dim),
            ).reshape(batch_size, num_policy_samples, action_dim)
            policy_samples = clip_action(raw_actions)
            proposals = sample_truncated_gaussian(
                proposal_key,
                policy_samples,
                repeats=proposals_per_policy_sample,
                std=proposal_std,
                perturb_clip=proposal_clip,
                include_anchor=include_anchor,
            ).reshape(
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
            source_qs = jnp.sum(source_distributions * z_atoms, axis=-1)
            if source_q_eval == "mean":
                source_q = source_qs.mean(axis=0)
            elif source_q_eval == "min":
                source_q = source_qs.min(axis=0)
            else:
                raise ValueError(f"Unknown source_q_eval: {source_q_eval}")
            source_q = jax.lax.stop_gradient(source_q)

            proposal_log_density = jnp.zeros_like(source_q)
            if density_correction:
                proposal_log_density = truncated_mixture_log_density(
                    jax.lax.stop_gradient(proposals),
                    jax.lax.stop_gradient(policy_samples),
                    proposal_std,
                    proposal_clip,
                )
            q_score = source_q / temperature
            density_score = -proposal_log_density
            logits = q_score + density_score
            source_weights = jax.lax.stop_gradient(jax.nn.softmax(logits, axis=-1))

            squared_costs = jnp.sum(
                jnp.square(policy_samples[:, :, None, :] - proposals[:, None, :, :]),
                axis=-1,
            )
            costs = squared_costs / (
                squared_costs.mean(axis=(-2, -1), keepdims=True) + 1.0e-8
            )
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
            loss = jnp.mean(
                jnp.sum(jnp.square(raw_actions - selected_actions), axis=-1)
            )
            source_ess = 1.0 / jnp.sum(jnp.square(source_weights), axis=-1)
            q_only_weights = jax.nn.softmax(q_score, axis=-1)
            q_only_ess = 1.0 / jnp.sum(jnp.square(q_only_weights), axis=-1)
            density_only_weights = jax.nn.softmax(density_score, axis=-1)
            density_only_ess = 1.0 / jnp.sum(jnp.square(density_only_weights), axis=-1)
            centered_q = source_q - source_q.mean(axis=-1, keepdims=True)
            centered_density = density_score - density_score.mean(
                axis=-1, keepdims=True
            )
            q_density_correlation = jnp.mean(centered_q * centered_density, axis=-1) / (
                jnp.std(source_q, axis=-1) * jnp.std(density_score, axis=-1) + 1.0e-8
            )
            q_logit_std = jnp.std(q_score, axis=-1)
            density_logit_std = jnp.std(density_score, axis=-1)

            def weighted_q_gain(weights, q_values):
                return jnp.sum(weights * q_values, axis=-1) - q_values.mean(axis=-1)

            q_only_q_gain = weighted_q_gain(q_only_weights, source_q)
            density_only_q_gain = weighted_q_gain(density_only_weights, source_q)
            full_q_gain = weighted_q_gain(source_weights, source_q)

            # Cross-evaluate selection by each twin critic with the other critic.
            # This is still not a fully independent estimate, but is less optimistic
            # than evaluating a Q-weighted selection with the exact same Q values.
            q1, q2 = source_qs[0], source_qs[1]
            q1_weights = jax.nn.softmax(q1 / temperature + density_score, axis=-1)
            q2_weights = jax.nn.softmax(q2 / temperature + density_score, axis=-1)
            cross_critic_q_gain = 0.5 * (
                weighted_q_gain(q1_weights, q2)
                + weighted_q_gain(q2_weights, q1)
            )
            metrics = {
                "actor_loss": loss,
                "source_ess_fraction": (source_ess / num_proposals).mean(),
                "q_only_ess_fraction": (q_only_ess / num_proposals).mean(),
                "density_only_ess_fraction": (density_only_ess / num_proposals).mean(),
                "max_source_weight": source_weights.max(axis=-1).mean(),
                "source_q_std": jnp.std(source_q, axis=-1).mean(),
                "neg_log_proposal_std": jnp.std(density_score, axis=-1).mean(),
                "q_logit_std": q_logit_std.mean(),
                "density_logit_std": density_logit_std.mean(),
                "combined_logit_std": jnp.std(logits, axis=-1).mean(),
                "q_to_density_logit_std_ratio": (
                    q_logit_std / (density_logit_std + 1.0e-8)
                ).mean(),
                "q_neglogq_correlation": q_density_correlation.mean(),
                "q_only_weighted_q_gain": q_only_q_gain.mean(),
                "density_only_weighted_q_gain": density_only_q_gain.mean(),
                "full_weighted_q_gain": full_q_gain.mean(),
                "cross_critic_weighted_q_gain": cross_critic_q_gain.mean(),
                "twin_q_abs_diff": jnp.abs(q1 - q2).mean(),
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
                    source_q / counterfactual_temperature + density_score,
                    axis=-1,
                )
                counterfactual_ess = 1.0 / jnp.sum(
                    jnp.square(counterfactual_weights), axis=-1
                )
                metrics[f"counterfactual_ess_T{label}"] = (
                    counterfactual_ess / num_proposals
                ).mean()
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
            "include_anchor",
            "density_correction",
            "sinkhorn_iterations",
            "source_q_eval",
            "transport_target_mode",
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
        proposal_std,
        proposal_clip,
        include_anchor,
        density_correction,
        temperature,
        sinkhorn_epsilon,
        sinkhorn_iterations,
        source_q_eval,
        transport_target_mode,
        td_noise_std,
        td_noise_clip,
    ):
        del n_env_interacts
        actor_metrics = {
            "actor_loss": jnp.asarray(0.0),
            "source_ess_fraction": jnp.asarray(0.0),
            "q_only_ess_fraction": jnp.asarray(0.0),
            "density_only_ess_fraction": jnp.asarray(0.0),
            "max_source_weight": jnp.asarray(0.0),
            "source_q_std": jnp.asarray(0.0),
            "neg_log_proposal_std": jnp.asarray(0.0),
            "q_logit_std": jnp.asarray(0.0),
            "density_logit_std": jnp.asarray(0.0),
            "combined_logit_std": jnp.asarray(0.0),
            "q_to_density_logit_std_ratio": jnp.asarray(0.0),
            "q_neglogq_correlation": jnp.asarray(0.0),
            "q_only_weighted_q_gain": jnp.asarray(0.0),
            "density_only_weighted_q_gain": jnp.asarray(0.0),
            "full_weighted_q_gain": jnp.asarray(0.0),
            "cross_critic_weighted_q_gain": jnp.asarray(0.0),
            "twin_q_abs_diff": jnp.asarray(0.0),
            "selected_delta_l2": jnp.asarray(0.0),
            "policy_spread_l2": jnp.asarray(0.0),
        }
        for label in ("0p05", "0p1", "0p2", "0p25", "0p5", "1p0"):
            actor_metrics[f"counterfactual_ess_T{label}"] = jnp.asarray(0.0)
        for i in range(gradient_steps):

            def slice_batch(array, step=i):
                batch_size = array.shape[0] // gradient_steps
                return array[batch_size * step : batch_size * (step + 1)]

            z_atoms = jnp.linspace(v_min, v_max, num_atoms)
            qf_state, critic_metrics, key = cls.update_critic(
                crossq_style,
                use_bnstats_from_live_net,
                gamma,
                target_actor_state,
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
            )
            qf_state = cls.soft_update(tau, qf_state)
            if i in policy_delay_indices:
                actor_state, _, key, actor_metrics = cls.update_actor(
                    actor_state,
                    qf_state,
                    slice_batch(data.observations),
                    key,
                    z_atoms,
                    num_policy_samples,
                    proposals_per_policy_sample,
                    proposal_std,
                    proposal_clip,
                    include_anchor,
                    density_correction,
                    temperature,
                    sinkhorn_epsilon,
                    sinkhorn_iterations,
                    source_q_eval,
                    transport_target_mode,
                )
                target_actor_state = cls.soft_update_target_actor(
                    policy_tau, actor_state, target_actor_state
                )
        log_metrics = {**actor_metrics, **critic_metrics}
        return (
            qf_state,
            actor_state,
            target_actor_state,
            ent_coef_state,
            key,
            log_metrics,
        )
