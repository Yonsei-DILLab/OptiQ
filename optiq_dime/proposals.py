"""Proposal-family dispatch and critic action gradients for OptiQ."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .transport import (
    GammaQgradCov_gaussian_mixture_log_density,
    QgradCov_mixture_log_density,
    QgradCov_truncated_mixture_log_density,
    clip_action,
    sample_GammaQgradCov_gaussian,
    sample_GammaQgradCov_gaussian_mixture,
    sample_QgradCov_gaussian,
    sample_QgradCov_gaussian_mixture,
    sample_QgradCov_truncated_gaussian,
    sample_QgradCov_truncated_gaussian_mixture,
    sample_truncated_gaussian,
    sample_truncated_gaussian_mixture,
    truncated_mixture_log_density,
)


PROPOSAL_FAMILIES = (
    "isotropic_truncated",
    "qgradcov_truncated",
    "qgradcov_gaussian",
    "gamma_qgradcov_gaussian",
)


def q_action_gradients(
    qf_state,
    observations: jax.Array,
    centers: jax.Array,
    z_atoms: jax.Array,
    dropout_key: jax.Array,
    q_eval: str,
) -> jax.Array:
    """Differentiate summed pointwise expected Q values with respect to actions."""
    batch_size, num_centers, action_dim = centers.shape
    observation_dim = observations.shape[-1]
    repeated_observations = jnp.broadcast_to(
        observations[:, None, :],
        (batch_size, num_centers, observation_dim),
    ).reshape(batch_size * num_centers, observation_dim)
    flat_centers = jax.lax.stop_gradient(
        centers.reshape(batch_size * num_centers, action_dim)
    )

    def summed_q(flat_actions):
        distributions = qf_state.apply_fn(
            {"params": qf_state.params, "batch_stats": qf_state.batch_stats},
            repeated_observations,
            flat_actions,
            rngs={"dropout": dropout_key},
            train=False,
        )
        twin_q = jnp.sum(distributions * z_atoms, axis=-1)
        if q_eval == "mean":
            scalar_q = twin_q.mean(axis=0)
        elif q_eval == "min":
            scalar_q = twin_q.min(axis=0)
        else:
            raise ValueError(f"Unknown q_eval: {q_eval}")
        return scalar_q.sum()

    gradients = jax.grad(summed_q)(flat_centers)
    return jax.lax.stop_gradient(
        gradients.reshape(batch_size, num_centers, action_dim)
    )


def sample_proposals(
    rng: jax.Array,
    centers: jax.Array,
    q_gradients: jax.Array | None,
    repeats: int,
    sampling_mode: str,
    proposal_family: str,
    std: float,
    perturb_clip: float,
    include_anchor: bool,
    perpendicular_std_ratio: float,
    gamma_shape: float,
    gamma_scale: float,
    gamma_perpendicular_std: float,
    clip_untruncated: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Sample one configured proposal cloud and return labels and clip rate."""
    if proposal_family not in PROPOSAL_FAMILIES:
        raise ValueError(f"Unknown proposal_family: {proposal_family}")
    if sampling_mode not in ("stratified", "exact"):
        raise ValueError(f"Unknown proposal_sampling_mode: {sampling_mode}")
    if proposal_family != "isotropic_truncated" and q_gradients is None:
        raise ValueError(f"{proposal_family} requires q_gradients")

    exact = sampling_mode == "exact"
    sampler_kwargs = {
        "rng": rng,
        "centers": centers,
        "repeats": repeats,
        "include_anchor": include_anchor,
    }
    if proposal_family == "isotropic_truncated":
        sampler = (
            sample_truncated_gaussian_mixture
            if exact
            else sample_truncated_gaussian
        )
        family_kwargs = {"std": std, "perturb_clip": perturb_clip}
    elif proposal_family == "qgradcov_truncated":
        sampler = (
            sample_QgradCov_truncated_gaussian_mixture
            if exact
            else sample_QgradCov_truncated_gaussian
        )
        family_kwargs = {
            "q_gradients": q_gradients,
            "std": std,
            "perturb_clip": perturb_clip,
            "perpendicular_std_ratio": perpendicular_std_ratio,
        }
    elif proposal_family == "qgradcov_gaussian":
        sampler = (
            sample_QgradCov_gaussian_mixture
            if exact
            else sample_QgradCov_gaussian
        )
        family_kwargs = {
            "q_gradients": q_gradients,
            "std": std,
            "perpendicular_std_ratio": perpendicular_std_ratio,
        }
    else:
        sampler = (
            sample_GammaQgradCov_gaussian_mixture
            if exact
            else sample_GammaQgradCov_gaussian
        )
        family_kwargs = {
            "q_gradients": q_gradients,
            "gamma_shape": gamma_shape,
            "gamma_scale": gamma_scale,
            "perpendicular_std": gamma_perpendicular_std,
        }

    if exact:
        samples, component_indices = sampler(
            **sampler_kwargs,
            **family_kwargs,
            return_component_indices=True,
        )
    else:
        samples = sampler(**sampler_kwargs, **family_kwargs)
        component_indices = jnp.broadcast_to(
            jnp.arange(centers.shape[1], dtype=jnp.int32)[None, :, None],
            samples.shape[:-1],
        )

    out_of_bounds = jnp.any((samples < -1.0) | (samples > 1.0), axis=-1)
    out_of_bounds_fraction = jnp.mean(out_of_bounds)
    if clip_untruncated and proposal_family in (
        "qgradcov_gaussian",
        "gamma_qgradcov_gaussian",
    ):
        samples = clip_action(samples)
    return samples, component_indices, out_of_bounds_fraction


def proposal_log_density(
    samples: jax.Array,
    centers: jax.Array,
    q_gradients: jax.Array | None,
    proposal_family: str,
    std: float,
    perturb_clip: float,
    perpendicular_std_ratio: float,
    gamma_shape: float,
    gamma_scale: float,
    gamma_perpendicular_std: float,
) -> jax.Array:
    """Evaluate the mixture density paired with ``sample_proposals``."""
    if proposal_family == "isotropic_truncated":
        return truncated_mixture_log_density(
            samples, centers, std, perturb_clip
        )
    if q_gradients is None:
        raise ValueError(f"{proposal_family} requires q_gradients")
    if proposal_family == "qgradcov_truncated":
        return QgradCov_truncated_mixture_log_density(
            samples,
            centers,
            q_gradients,
            std,
            perturb_clip,
            perpendicular_std_ratio,
        )
    if proposal_family == "qgradcov_gaussian":
        return QgradCov_mixture_log_density(
            samples,
            centers,
            q_gradients,
            std,
            perpendicular_std_ratio,
        )
    if proposal_family == "gamma_qgradcov_gaussian":
        return GammaQgradCov_gaussian_mixture_log_density(
            samples,
            centers,
            q_gradients,
            gamma_shape,
            gamma_scale,
            gamma_perpendicular_std,
        )
    raise ValueError(f"Unknown proposal_family: {proposal_family}")


def stabilize_proposal_log_density(log_density: jax.Array) -> jax.Array:
    """Replace unsupported/nonfinite candidates by the row's lowest finite log q.

    Post-sampling clipping can move a Gamma proposal outside every component's
    forward half-space.  The analytical density is then zero, but an infinite
    inverse-density score would make softmax undefined.  This explicit heuristic
    keeps the run finite while preserving all finite density values unchanged.
    """
    finite = jnp.isfinite(log_density)
    row_floor = jnp.min(
        jnp.where(finite, log_density, jnp.inf), axis=-1, keepdims=True
    )
    row_floor = jnp.where(jnp.isfinite(row_floor), row_floor, 0.0)
    return jnp.where(finite, log_density, row_floor)
