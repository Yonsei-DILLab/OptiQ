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


def _sample_truncated_chi_square(
    rng: jax.Array,
    shape: tuple[int, ...],
    degrees_of_freedom: int,
    maximum_radius: float,
    dtype: jnp.dtype,
) -> jax.Array:
    """Sample chi-square radii conditioned on a fixed maximum radius."""

    maximum_squared_radius = jnp.square(jnp.asarray(maximum_radius, dtype=dtype))
    half_degrees = jnp.asarray(0.5 * degrees_of_freedom, dtype=dtype)
    maximum_cdf = jsp.special.gammainc(half_degrees, 0.5 * maximum_squared_radius)
    uniform = jax.random.uniform(
        rng,
        shape,
        minval=jnp.finfo(dtype).eps,
        maxval=1.0 - jnp.finfo(dtype).eps,
        dtype=dtype,
    )
    target_cdf = uniform * maximum_cdf
    lower = jnp.zeros(shape, dtype=dtype)
    upper = jnp.full(shape, maximum_squared_radius, dtype=dtype)

    def bisect(_, bounds):
        current_lower, current_upper = bounds
        midpoint = 0.5 * (current_lower + current_upper)
        midpoint_cdf = jsp.special.gammainc(half_degrees, 0.5 * midpoint)
        move_lower = midpoint_cdf < target_cdf
        return (
            jnp.where(move_lower, midpoint, current_lower),
            jnp.where(move_lower, current_upper, midpoint),
        )

    lower, upper = jax.lax.fori_loop(0, 32, bisect, (lower, upper))
    return 0.5 * (lower + upper)


def sample_gradient_skewed_mixture(
    rng: jax.Array,
    centers: jax.Array,
    gradients: jax.Array,
    repeats: int,
    std: float,
    perturb_clip: float,
    action_low,
    action_high,
    include_anchor: bool = False,
) -> jax.Array:
    """Sample exact pairwise-skewed proposals from a uniform center mixture."""

    if centers.ndim != 3 or gradients.shape != centers.shape:
        raise ValueError(
            "centers and gradients must have shape [batch, components, action_dim]"
        )
    random_repeats = repeats - int(include_anchor)
    if random_repeats < 0:
        raise ValueError("repeats must be at least one when include_anchor=True")

    _, _, _, half_range = _action_geometry(action_low, action_high, centers.dtype)
    batch_size, num_centers, action_dim = centers.shape
    component_key, direction_key, radius_key, sign_key = jax.random.split(rng, 4)
    component_indices = jax.random.randint(
        component_key,
        (batch_size, num_centers, random_repeats),
        minval=0,
        maxval=num_centers,
    )

    def select(values, indices):
        return values[indices]

    selected_centers = jax.vmap(select)(centers, component_indices)
    selected_gradients = jax.vmap(select)(gradients, component_indices)
    normalized_gradients = selected_gradients * half_range
    gradient_norms = jnp.linalg.norm(
        normalized_gradients, axis=-1, keepdims=True
    )
    selected_directions = jnp.where(
        gradient_norms > 1.0e-8,
        normalized_gradients / jnp.maximum(gradient_norms, 1.0e-8),
        0.0,
    )
    spherical_directions = jax.random.normal(
        direction_key,
        (batch_size, num_centers, random_repeats, action_dim),
        dtype=centers.dtype,
    )
    spherical_directions /= jnp.maximum(
        jnp.linalg.norm(spherical_directions, axis=-1, keepdims=True), 1.0e-8
    )
    squared_radii = _sample_truncated_chi_square(
        radius_key,
        (batch_size, num_centers, random_repeats, 1),
        action_dim,
        perturb_clip / std,
        centers.dtype,
    )
    normalized_noise = std * spherical_directions * jnp.sqrt(squared_radii)
    keep_logits = (
        2.0
        * jnp.sum(selected_directions * normalized_noise, axis=-1)
        / std
    )
    keep = jax.random.bernoulli(sign_key, jax.nn.sigmoid(keep_logits))
    oriented_normalized_noise = jnp.where(
        keep[..., None], normalized_noise, -normalized_noise
    )
    oriented_noise = half_range * oriented_normalized_noise
    random_samples = selected_centers + oriented_noise

    if include_anchor:
        return jnp.concatenate((centers[..., None, :], random_samples), axis=-2)
    return random_samples


def gradient_skewed_mixture_log_density(
    samples: jax.Array,
    centers: jax.Array,
    gradients: jax.Array,
    std: float,
    perturb_clip: float,
    action_low,
    action_high,
) -> jax.Array:
    """Evaluate the exact pairwise-skewed uniform-mixture log density."""

    _, _, _, half_range = _action_geometry(action_low, action_high, samples.dtype)
    action_differences = samples[:, :, None, :] - centers[:, None, :, :]
    normalized_differences = action_differences / half_range
    squared_mahalanobis = jnp.sum(
        jnp.square(normalized_differences / std), axis=-1
    )
    action_dim = samples.shape[-1]
    maximum_radius = perturb_clip / std
    maximum_squared_radius = jnp.square(
        jnp.asarray(maximum_radius, dtype=samples.dtype)
    )
    log_radial_normalizer = jnp.log(
        jnp.maximum(
            jsp.special.gammainc(
                jnp.asarray(0.5 * action_dim, dtype=samples.dtype),
                0.5 * maximum_squared_radius,
            ),
            jnp.finfo(samples.dtype).tiny,
        )
    )
    base_component_log_density = (
        -0.5 * squared_mahalanobis
        - 0.5 * action_dim * jnp.log(2.0 * jnp.pi)
        - action_dim * jnp.log(std)
        - jnp.sum(jnp.log(half_range))
        - log_radial_normalizer
    )
    normalized_gradients = gradients * half_range
    gradient_norms = jnp.linalg.norm(normalized_gradients, axis=-1, keepdims=True)
    directions = jnp.where(
        gradient_norms > 1.0e-8,
        normalized_gradients / jnp.maximum(gradient_norms, 1.0e-8),
        0.0,
    )
    skew_logits = (
        2.0
        * jnp.sum(normalized_differences * directions[:, None, :, :], axis=-1)
        / std
    )
    component_log_density = (
        base_component_log_density
        + jnp.log(jnp.asarray(2.0, dtype=samples.dtype))
        + jax.nn.log_sigmoid(skew_logits)
    )
    component_log_density = jnp.where(
        squared_mahalanobis <= maximum_squared_radius + 1.0e-5,
        component_log_density,
        -jnp.inf,
    )
    return jsp.special.logsumexp(component_log_density, axis=-1) - jnp.log(
        centers.shape[1]
    )


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
