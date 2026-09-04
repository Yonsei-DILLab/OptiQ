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
) -> jax.Array:
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
        return jnp.concatenate((centers[..., None, :], random_samples), axis=-2)
    return random_samples


def _gradient_directions(gradients: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return unit Q-gradient directions and a mask for non-flat gradients."""
    norms = jnp.linalg.norm(gradients, axis=-1, keepdims=True)
    valid = norms > 1.0e-8
    directions = jnp.where(valid, gradients / jnp.maximum(norms, 1.0e-8), 0.0)
    return directions, valid[..., 0]


def _sample_truncated_chi_square(
    rng: jax.Array,
    shape: tuple[int, ...],
    degrees_of_freedom: int,
    maximum_radius: float,
    dtype: jnp.dtype,
) -> jax.Array:
    """Sample chi-square radii conditioned on ``radius <= maximum_radius``.

    JAX does not expose an inverse chi-square CDF, so a fixed bisection inverts
    the regularized lower incomplete gamma function.  This remains efficient
    even when a fixed-radius truncation has tiny probability in high dimension,
    where rejection sampling would be unusable.
    """
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


def sample_gradient_chi_square_mixture(
    rng: jax.Array,
    centers: jax.Array,
    gradients: jax.Array,
    repeats: int,
    perpendicular_std: float,
    parallel_std: float,
    gradient_step: float,
    maximum_radius: float,
    include_anchor: bool = False,
) -> jax.Array:
    """Sample IID proposals from a Q-gradient-aligned truncated mixture.

    Every random proposal independently selects an actor center.  The component
    mean is shifted along its normalized Q gradient and its covariance has
    ``parallel_std`` on the gradient axis and ``perpendicular_std`` elsewhere.
    Its squared Mahalanobis radius follows a chi-square distribution truncated
    at ``maximum_radius ** 2``.  Original actor samples can be retained as
    deterministic anchors in slot zero.
    """
    if centers.ndim != 3 or gradients.shape != centers.shape:
        raise ValueError(
            "centers and gradients must have shape [batch, components, action_dim]"
        )

    random_repeats = repeats - int(include_anchor)
    if random_repeats < 0:
        raise ValueError("repeats must be at least one when include_anchor=True")

    batch_size, num_centers, action_dim = centers.shape
    directions, _ = _gradient_directions(gradients)
    means = centers + gradient_step * directions
    component_key, direction_key, radius_key = jax.random.split(rng, 3)
    component_indices = jax.random.randint(
        component_key,
        (batch_size, num_centers, random_repeats),
        minval=0,
        maxval=num_centers,
    )

    def select(values, indices):
        return values[indices]

    selected_means = jax.vmap(select)(means, component_indices)
    selected_directions = jax.vmap(select)(directions, component_indices)
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
        maximum_radius,
        centers.dtype,
    )
    whitened_noise = spherical_directions * jnp.sqrt(squared_radii)
    parallel_coordinates = jnp.sum(
        whitened_noise * selected_directions, axis=-1, keepdims=True
    )
    noise = perpendicular_std * whitened_noise + (
        parallel_std - perpendicular_std
    ) * parallel_coordinates * selected_directions
    random_samples = selected_means + noise

    if include_anchor:
        return jnp.concatenate((centers[..., None, :], random_samples), axis=-2)
    return random_samples


def gradient_chi_square_mixture_log_density(
    samples: jax.Array,
    centers: jax.Array,
    gradients: jax.Array,
    perpendicular_std: float,
    parallel_std: float,
    gradient_step: float,
    maximum_radius: float,
) -> jax.Array:
    """Evaluate the exact gradient-aligned truncated mixture log density."""
    directions, valid_gradients = _gradient_directions(gradients)
    means = centers + gradient_step * directions
    differences = samples[:, :, None, :] - means[:, None, :, :]
    parallel_coordinates = jnp.sum(
        differences * directions[:, None, :, :], axis=-1
    )
    squared_norms = jnp.sum(jnp.square(differences), axis=-1)
    perpendicular_squared_norms = jnp.maximum(
        squared_norms - jnp.square(parallel_coordinates), 0.0
    )
    squared_mahalanobis = (
        perpendicular_squared_norms / jnp.square(perpendicular_std)
        + jnp.square(parallel_coordinates) / jnp.square(parallel_std)
    )

    action_dim = samples.shape[-1]
    log_scale_determinant = (
        action_dim * jnp.log(perpendicular_std)
        + valid_gradients
        * (jnp.log(parallel_std) - jnp.log(perpendicular_std))
    )
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
    component_log_density = (
        -0.5 * squared_mahalanobis
        - 0.5 * action_dim * jnp.log(2.0 * jnp.pi)
        - log_scale_determinant[:, None, :]
        - log_radial_normalizer
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
