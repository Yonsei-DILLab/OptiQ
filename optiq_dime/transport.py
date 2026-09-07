"""Density-corrected local proposals and entropic optimal transport."""

import jax
import jax.numpy as jnp
import jax.scipy as jsp


def clip_action(actions: jax.Array) -> jax.Array:
    """Clip normalized actions to the range expected by SB3 policies."""
    return jnp.clip(actions, -1.0, 1.0)


def sample_truncated_gaussian(
    rng: jax.Array,
    centers: jax.Array,
    repeats: int,
    std: float,
    perturb_clip: float,
    include_anchor: bool = False,
) -> jax.Array:
    """Sample an exact Gaussian truncated by local and action-space bounds."""
    lower = jnp.maximum(-1.0 - centers, -perturb_clip)
    upper = jnp.minimum(1.0 - centers, perturb_clip)
    random_repeats = repeats - int(include_anchor)
    sample_shape = centers.shape[:-1] + (random_repeats, centers.shape[-1])
    lower_cdf = jsp.special.ndtr(lower / std)[..., None, :]
    upper_cdf = jsp.special.ndtr(upper / std)[..., None, :]
    uniform = jax.random.uniform(
        rng,
        sample_shape,
        minval=jnp.finfo(centers.dtype).eps,
        maxval=1.0 - jnp.finfo(centers.dtype).eps,
    )
    quantiles = lower_cdf + uniform * (upper_cdf - lower_cdf)
    noise = std * jsp.special.ndtri(jnp.clip(quantiles, 1.0e-7, 1.0 - 1.0e-7))
    if include_anchor:
        anchor = jnp.zeros(centers.shape[:-1] + (1, centers.shape[-1]), centers.dtype)
        noise = jnp.concatenate((anchor, noise), axis=-2)
    return centers[..., None, :] + noise


def sample_truncated_gaussian_mixture(
    rng: jax.Array,
    centers: jax.Array,
    repeats: int,
    std: float,
    perturb_clip: float,
    include_anchor: bool = False,
    return_component_indices: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Sample IID random proposals from the uniform mixture over centers.

    The leading two dimensions of ``centers`` are batch and mixture component.
    Every random proposal first selects a component independently and uniformly,
    then draws from that component's exactly truncated Gaussian.  When requested,
    one deterministic copy of every original center is retained in slot zero;
    those anchors are intentionally separate from the IID random proposals.
    """
    if centers.ndim != 3:
        raise ValueError("centers must have shape [batch, components, action_dim]")

    random_repeats = repeats - int(include_anchor)
    if random_repeats < 0:
        raise ValueError("repeats must be at least one when include_anchor=True")

    batch_size, num_centers, action_dim = centers.shape
    component_key, sample_key = jax.random.split(rng)
    component_indices = jax.random.randint(
        component_key,
        (batch_size, num_centers, random_repeats),
        minval=0,
        maxval=num_centers,
    )
    selected_centers = jax.vmap(
        lambda batch_centers, batch_indices: batch_centers[batch_indices]
    )(centers, component_indices)

    lower = jnp.maximum(-1.0 - selected_centers, -perturb_clip)
    upper = jnp.minimum(1.0 - selected_centers, perturb_clip)
    lower_cdf = jsp.special.ndtr(lower / std)
    upper_cdf = jsp.special.ndtr(upper / std)
    uniform = jax.random.uniform(
        sample_key,
        (batch_size, num_centers, random_repeats, action_dim),
        minval=jnp.finfo(centers.dtype).eps,
        maxval=1.0 - jnp.finfo(centers.dtype).eps,
    )
    quantiles = lower_cdf + uniform * (upper_cdf - lower_cdf)
    noise = std * jsp.special.ndtri(jnp.clip(quantiles, 1.0e-7, 1.0 - 1.0e-7))
    random_samples = selected_centers + noise

    if include_anchor:
        random_samples = jnp.concatenate((centers[..., None, :], random_samples), axis=-2)
        anchors = jnp.broadcast_to(jnp.arange(num_centers)[None, :, None], (batch_size, num_centers, 1))
        component_indices = jnp.concatenate((anchors, component_indices), axis=-1)
    if return_component_indices:
        return random_samples, component_indices
    return random_samples


def truncated_mixture_log_density(
    samples: jax.Array,
    centers: jax.Array,
    std: float,
    perturb_clip: float,
) -> jax.Array:
    """Evaluate the local proposal mixture density at normalized actions."""
    lower = jnp.maximum(-1.0 - centers, -perturb_clip)
    upper = jnp.minimum(1.0 - centers, perturb_clip)
    differences = samples[:, :, None, :] - centers[:, None, :, :]
    lower_cdf = jsp.special.ndtr(lower / std)
    upper_cdf = jsp.special.ndtr(upper / std)
    log_normalizer = jnp.sum(
        jnp.log(jnp.maximum(upper_cdf - lower_cdf, 1.0e-20)), axis=-1
    )
    action_dim = samples.shape[-1]
    log_density = (
        -0.5 * jnp.sum(jnp.square(differences / std), axis=-1)
        - action_dim * (jnp.log(std) + 0.5 * jnp.log(2.0 * jnp.pi))
        - log_normalizer[:, None, :]
    )
    in_support = jnp.all(
        (differences >= lower[:, None, :, :] - 1.0e-6)
        & (differences <= upper[:, None, :, :] + 1.0e-6),
        axis=-1,
    )
    log_density = jnp.where(in_support, log_density, -jnp.inf)
    return jsp.special.logsumexp(log_density, axis=-1) - jnp.log(centers.shape[1])


def select_density_beta_for_ess(
    q_score: jax.Array,
    density_score: jax.Array,
    minimum_ess: float,
    grid_size: int = 257,
) -> tuple[jax.Array, jax.Array]:
    """Select the strongest density correction whose ESS stays above a floor.

    For every batch row this searches beta in ``[0, 1]`` for weights

      softmax(q_score + beta * density_score).

    The returned beta is the largest point in the feasible prefix connected to
    beta=0.  Using a prefix instead of an unconstrained maximum is robust to a
    rare non-monotone ESS curve caused by correlation between Q and density.
    ``grid_size=257`` resolves beta to 1/256 while keeping the computation tiny
    relative to the critic forward pass.
    """
    beta_grid = jnp.linspace(0.0, 1.0, grid_size, dtype=q_score.dtype)
    candidate_logits = (
        q_score[None, ...]
        + beta_grid[:, None, None] * density_score[None, ...]
    )
    log_weight_sum = jsp.special.logsumexp(candidate_logits, axis=-1)
    log_squared_weight_sum = jsp.special.logsumexp(
        2.0 * candidate_logits, axis=-1
    )
    candidate_ess = jnp.exp(2.0 * log_weight_sum - log_squared_weight_sum)
    feasible = candidate_ess >= jnp.asarray(minimum_ess, q_score.dtype)
    feasible_prefix = jnp.cumprod(feasible.astype(jnp.int32), axis=0).astype(bool)
    grid_indices = jnp.arange(grid_size, dtype=jnp.int32)[:, None]
    selected_indices = jnp.max(
        jnp.where(feasible_prefix, grid_indices, 0), axis=0
    )
    selected_beta = beta_grid[selected_indices]
    selected_ess = jnp.take_along_axis(
        candidate_ess, selected_indices[None, :], axis=0
    )[0]
    return selected_beta, selected_ess


def sinkhorn(
    costs: jax.Array,
    source_weights: jax.Array,
    epsilon: float,
    iterations: int,
) -> jax.Array:
    """Solve batched entropic OT with uniform policy-sample marginals."""
    batch_size, num_rows, _ = costs.shape
    row_weights = jnp.full((batch_size, num_rows), 1.0 / num_rows, costs.dtype)
    source_weights = jnp.clip(source_weights, 1.0e-20, 1.0)
    source_weights /= source_weights.sum(axis=-1, keepdims=True)
    log_kernel = -costs / epsilon
    log_rows = jnp.log(row_weights)
    log_columns = jnp.log(source_weights)
    log_u = jnp.zeros_like(log_rows)
    log_v = jnp.zeros_like(log_columns)

    def iteration(_, potentials):
        current_u, current_v = potentials
        current_u = log_rows - jsp.special.logsumexp(
            log_kernel + current_v[:, None, :], axis=-1
        )
        current_v = log_columns - jsp.special.logsumexp(
            log_kernel + current_u[:, :, None], axis=-2
        )
        return current_u, current_v

    log_u, log_v = jax.lax.fori_loop(0, iterations, iteration, (log_u, log_v))
    return jnp.exp(log_kernel + log_u[:, :, None] + log_v[:, None, :])
