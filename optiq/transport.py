import jax
import jax.numpy as jnp
import jax.scipy as jsp


def _action_geometry(action_low, action_high, dtype: jnp.dtype):
    low = jnp.asarray(action_low, dtype=dtype)
    high = jnp.asarray(action_high, dtype=dtype)
    midpoint = 0.5 * (low + high)
    half_range = jnp.maximum(0.5 * (high - low), 1.0e-6)
    return low, high, midpoint, half_range


def clip_action(actions: jax.Array, action_low, action_high) -> jax.Array:
    low, high, _, _ = _action_geometry(action_low, action_high, actions.dtype)
    return jnp.clip(actions, low, high)


def sample_truncated_gaussian(
    rng: jax.Array,
    centers: jax.Array,
    repeats: int,
    std: float,
    perturb_clip: float,
    action_low,
    action_high,
    include_anchor: bool = False,
) -> jax.Array:
    """Sample an exact box-truncated Gaussian using inverse-CDF sampling."""
    _, _, midpoint, half_range = _action_geometry(
        action_low, action_high, centers.dtype
    )
    normalized_centers = (centers - midpoint) / half_range
    lower = jnp.maximum(-1.0 - normalized_centers, -perturb_clip)
    upper = jnp.minimum(1.0 - normalized_centers, perturb_clip)

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
    noise = std * jsp.special.ndtri(
        jnp.clip(quantiles, 1.0e-7, 1.0 - 1.0e-7)
    )
    if include_anchor:
        anchor = jnp.zeros(
            centers.shape[:-1] + (1, centers.shape[-1]), centers.dtype
        )
        noise = jnp.concatenate((anchor, noise), axis=-2)

    normalized_actions = normalized_centers[..., None, :] + noise
    return midpoint + half_range * normalized_actions


def truncated_mixture_log_density(
    samples: jax.Array,
    centers: jax.Array,
    std: float,
    perturb_clip: float,
    action_low,
    action_high,
) -> jax.Array:
    """Evaluate the exact continuous proposal-mixture density."""
    _, _, midpoint, half_range = _action_geometry(
        action_low, action_high, samples.dtype
    )
    normalized_samples = (samples - midpoint) / half_range
    normalized_centers = (centers - midpoint) / half_range

    lower = jnp.maximum(-1.0 - normalized_centers, -perturb_clip)
    upper = jnp.minimum(1.0 - normalized_centers, perturb_clip)
    differences = (
        normalized_samples[:, :, None, :] - normalized_centers[:, None, :, :]
    )

    lower_cdf = jsp.special.ndtr(lower / std)
    upper_cdf = jsp.special.ndtr(upper / std)
    log_normalizer = jnp.sum(
        jnp.log(jnp.maximum(upper_cdf - lower_cdf, 1.0e-20)), axis=-1
    )
    action_dim = samples.shape[-1]
    log_density = (
        -0.5 * jnp.sum(jnp.square(differences / std), axis=-1)
        - action_dim * (jnp.log(std) + 0.5 * jnp.log(2.0 * jnp.pi))
        - jnp.sum(jnp.log(half_range))
        - log_normalizer[:, None, :]
    )
    in_support = jnp.all(
        (differences >= lower[:, None, :, :] - 1.0e-6)
        & (differences <= upper[:, None, :, :] + 1.0e-6),
        axis=-1,
    )
    log_density = jnp.where(in_support, log_density, -jnp.inf)
    return jsp.special.logsumexp(log_density, axis=-1) - jnp.log(
        centers.shape[1]
    )


def sinkhorn(
    costs: jax.Array,
    source_weights: jax.Array,
    epsilon: float,
    iterations: int,
) -> jax.Array:
    """Solve batched entropic OT with uniform row marginals."""
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

    log_u, log_v = jax.lax.fori_loop(
        0, iterations, iteration, (log_u, log_v)
    )
    return jnp.exp(log_kernel + log_u[:, :, None] + log_v[:, None, :])
