"""Short counterfactual ESS diagnostic over perturbation sigma and temperature."""

import json
from functools import partial

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig

from optiq_dime.proposals import (
    proposal_log_density,
    q_action_gradients,
    sample_proposals,
    stabilize_proposal_log_density,
)
from optiq_dime.transport import clip_action
from run_optiq_dime import create_algorithm

SIGMAS = (0.05, 0.1, 0.2, 0.3, 0.4)
TEMPERATURES = (0.05, 0.1, 0.2, 0.25, 0.5, 1.0)
DIAGNOSTIC_BATCH_SIZE = 128


def flatten_observations(model, observations) -> np.ndarray:
    if isinstance(observations, dict):
        keys = list(model.observation_space.keys())
        return np.concatenate([observations[key].numpy() for key in keys], axis=1)
    return observations.numpy()


@partial(jax.jit, static_argnames=("num_policy_samples",))
def policy_samples(
    actor_state,
    observations,
    latent_key,
    num_policy_samples: int,
):
    batch_size, observation_dim = observations.shape
    output_layer = f"Dense_{len(actor_state.params) - 1}"
    action_dim = actor_state.params[output_layer]["bias"].shape[0]
    latents = jax.random.normal(
        latent_key,
        (batch_size, num_policy_samples, action_dim),
        dtype=observations.dtype,
    )
    repeated_observations = jnp.broadcast_to(
        observations[:, None, :],
        (batch_size, num_policy_samples, observation_dim),
    )
    actions = actor_state.apply_fn(
        {"params": actor_state.params},
        repeated_observations.reshape(batch_size * num_policy_samples, -1),
        latents.reshape(batch_size * num_policy_samples, action_dim),
    )
    return clip_action(actions.reshape(batch_size, num_policy_samples, action_dim))


@partial(
    jax.jit,
    static_argnames=(
        "proposals_per_policy_sample",
        "proposal_family",
        "proposal_sampling_mode",
        "clip_untruncated_proposals",
    ),
)
def diagnose_sigma(
    qf_state,
    observations,
    centers,
    q_gradients,
    proposal_key,
    dropout_key,
    z_atoms,
    temperatures,
    sigma,
    perturb_clip,
    proposals_per_policy_sample: int,
    proposal_family: str,
    proposal_sampling_mode: str,
    proposal_perpendicular_std_ratio: float,
    gamma_shape: float,
    gamma_scale: float,
    clip_untruncated_proposals: bool,
):
    batch_size, num_policy_samples, action_dim = centers.shape
    observation_dim = observations.shape[-1]
    proposals, _, out_of_bounds_fraction = sample_proposals(
        proposal_key,
        centers,
        q_gradients,
        proposals_per_policy_sample,
        proposal_sampling_mode,
        proposal_family,
        sigma,
        perturb_clip,
        True,
        proposal_perpendicular_std_ratio,
        gamma_shape,
        gamma_scale,
        sigma,
        clip_untruncated_proposals,
    )
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
    distributions = qf_state.apply_fn(
        {"params": qf_state.params, "batch_stats": qf_state.batch_stats},
        proposal_observations.reshape(batch_size * num_proposals, -1),
        proposals.reshape(batch_size * num_proposals, action_dim),
        rngs={"dropout": dropout_key},
        train=False,
    ).reshape(2, batch_size, num_proposals, -1)
    expected_q = jnp.sum(distributions * z_atoms, axis=-1).mean(axis=0)
    log_density = proposal_log_density(
        proposals,
        centers,
        q_gradients,
        proposal_family,
        sigma,
        perturb_clip,
        proposal_perpendicular_std_ratio,
        gamma_shape,
        gamma_scale,
        sigma,
    )
    unsupported_fraction = jnp.mean(~jnp.isfinite(log_density))
    log_density = stabilize_proposal_log_density(log_density)
    density_score = -log_density

    logits = (
        expected_q[None, :, :] / temperatures[:, None, None] + density_score[None, :, :]
    )
    weights = jax.nn.softmax(logits, axis=-1)
    ess_fraction = (1.0 / jnp.sum(jnp.square(weights), axis=-1) / num_proposals).mean(
        axis=-1
    )
    max_weight = jnp.max(weights, axis=-1).mean(axis=-1)

    density_weights = jax.nn.softmax(density_score, axis=-1)
    density_ess_fraction = (
        1.0 / jnp.sum(jnp.square(density_weights), axis=-1) / num_proposals
    ).mean()
    centered_q = expected_q - expected_q.mean(axis=-1, keepdims=True)
    centered_density = density_score - density_score.mean(axis=-1, keepdims=True)
    correlation = jnp.mean(centered_q * centered_density, axis=-1) / (
        jnp.std(expected_q, axis=-1) * jnp.std(density_score, axis=-1) + 1.0e-8
    )
    return {
        "ess_fraction": ess_fraction,
        "max_weight": max_weight,
        "density_only_ess_fraction": density_ess_fraction,
        "source_q_std": jnp.std(expected_q, axis=-1).mean(),
        "neg_log_proposal_std": jnp.std(density_score, axis=-1).mean(),
        "q_neglogq_correlation": correlation.mean(),
        "proposal_out_of_bounds_fraction": out_of_bounds_fraction,
        "proposal_unsupported_fraction": unsupported_fraction,
    }


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="optiq_dime_dog",
)
def main(cfg: DictConfig) -> None:
    cfg.wandb.activate = False
    cfg.eval_interval = 0
    model, _ = create_algorithm(cfg)
    model.learn(
        total_timesteps=int(cfg.total_steps),
        progress_bar=False,
        callback=None,
    )

    replay = model.replay_buffer.sample(DIAGNOSTIC_BATCH_SIZE)
    observations = jnp.asarray(flatten_observations(model, replay.observations))
    key = jax.random.key(int(cfg.seed) + 10_000)
    key, latent_key = jax.random.split(key)
    centers = policy_samples(
        model.policy.actor_state,
        observations,
        latent_key,
        int(cfg.alg.actor.num_policy_samples),
    )
    temperatures = jnp.asarray(TEMPERATURES, dtype=observations.dtype)
    z_atoms = jnp.linspace(
        cfg.alg.critic.v_min,
        cfg.alg.critic.v_max,
        cfg.alg.critic.n_atoms,
    )
    q_gradients = None
    if str(cfg.alg.actor.proposal_family) != "isotropic_truncated":
        key, qgrad_dropout_key = jax.random.split(key)
        q_gradients = q_action_gradients(
            model.policy.qf_state,
            observations,
            centers,
            z_atoms,
            qgrad_dropout_key,
            str(cfg.alg.actor.source_q_eval),
        )

    rows = []
    for sigma in SIGMAS:
        key, proposal_key, dropout_key = jax.random.split(key, 3)
        perturb_clip = 2.5 * sigma
        metrics = diagnose_sigma(
            model.policy.qf_state,
            observations,
            centers,
            q_gradients,
            proposal_key,
            dropout_key,
            z_atoms,
            temperatures,
            sigma,
            perturb_clip,
            int(cfg.alg.actor.proposals_per_policy_sample),
            str(cfg.alg.actor.proposal_family),
            str(cfg.alg.actor.proposal_sampling_mode),
            float(cfg.alg.actor.proposal_perpendicular_std_ratio),
            float(cfg.alg.actor.gamma_shape),
            float(cfg.alg.actor.gamma_scale),
            bool(cfg.alg.actor.clip_untruncated_proposals),
        )
        row = {
            "sigma": sigma,
            "perturb_clip": perturb_clip,
            "density_only_ess_fraction": float(metrics["density_only_ess_fraction"]),
            "source_q_std": float(metrics["source_q_std"]),
            "neg_log_proposal_std": float(metrics["neg_log_proposal_std"]),
            "q_neglogq_correlation": float(metrics["q_neglogq_correlation"]),
            "proposal_out_of_bounds_fraction": float(
                metrics["proposal_out_of_bounds_fraction"]
            ),
            "proposal_unsupported_fraction": float(
                metrics["proposal_unsupported_fraction"]
            ),
            "ess_fraction": {
                str(temperature): float(value)
                for temperature, value in zip(
                    TEMPERATURES,
                    metrics["ess_fraction"].tolist(),
                    strict=True,
                )
            },
            "max_weight": {
                str(temperature): float(value)
                for temperature, value in zip(
                    TEMPERATURES,
                    metrics["max_weight"].tolist(),
                    strict=True,
                )
            },
        }
        rows.append(row)

    result = {
        "task": cfg.task,
        "seed": int(cfg.seed),
        "training_steps": int(cfg.total_steps),
        "num_policy_samples": int(cfg.alg.actor.num_policy_samples),
        "proposals_per_policy_sample": int(cfg.alg.actor.proposals_per_policy_sample),
        "proposal_family": str(cfg.alg.actor.proposal_family),
        "proposal_sampling_mode": str(cfg.alg.actor.proposal_sampling_mode),
        "proposal_perpendicular_std_ratio": float(
            cfg.alg.actor.proposal_perpendicular_std_ratio
        ),
        "gamma_shape": float(cfg.alg.actor.gamma_shape),
        "gamma_scale": float(cfg.alg.actor.gamma_scale),
        "clip_untruncated_proposals": bool(
            cfg.alg.actor.clip_untruncated_proposals
        ),
        "diagnostic_batch_size": DIAGNOSTIC_BATCH_SIZE,
        "temperatures": TEMPERATURES,
        "rows": rows,
    }
    print(
        "TEMPERATURE_PERTURBATION_DIAGNOSTIC=" + json.dumps(result),
        flush=True,
    )


if __name__ == "__main__":
    main()
