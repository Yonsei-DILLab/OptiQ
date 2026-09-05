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
    Diagnostics can request the generating component index for every returned
    sample; actor training only consumes the samples themselves.
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
        samples = jnp.concatenate((centers[..., None, :], random_samples), axis=-2)
        anchor_indices = jnp.broadcast_to(
            jnp.arange(num_centers, dtype=component_indices.dtype)[None, :, None],
            (batch_size, num_centers, 1),
        )
        sample_component_indices = jnp.concatenate(
            (anchor_indices, component_indices), axis=-1
        )
    else:
        samples = random_samples
        sample_component_indices = component_indices

    if return_component_indices:
        return samples, sample_component_indices
    return samples


def _normalize_q_gradients(
    q_gradients: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Normalize Q gradients and identify rows with a defined direction."""
    gradient_norm = jnp.linalg.norm(q_gradients, axis=-1, keepdims=True)
    has_direction = gradient_norm[..., 0] > 1.0e-12
    unit_gradient = jnp.where(
        has_direction[..., None],
        q_gradients / jnp.maximum(gradient_norm, 1.0e-12),
        0.0,
    )
    return unit_gradient, has_direction


def _qgrad_covariance_parameters(
    q_gradients: jax.Array,
    parallel_std: float,
    perpendicular_std_ratio: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return the unit gradient, marginal stds, and rank-one variances."""
    unit_gradient, _ = _normalize_q_gradients(q_gradients)
    parallel_variance = jnp.asarray(parallel_std, dtype=q_gradients.dtype) ** 2
    perpendicular_std = parallel_std * perpendicular_std_ratio
    perpendicular_variance = (
        jnp.asarray(perpendicular_std, dtype=q_gradients.dtype) ** 2
    )
    variance_gap = parallel_variance - perpendicular_variance
    marginal_std = jnp.sqrt(
        perpendicular_variance + variance_gap * jnp.square(unit_gradient)
    )
    return unit_gradient, marginal_std, perpendicular_variance, variance_gap


def _sample_qgrad_cov_unbounded(
    rng: jax.Array,
    centers: jax.Array,
    q_gradients: jax.Array,
    parallel_std: float,
    perpendicular_std_ratio: float,
) -> jax.Array:
    """Apply the rank-one covariance square root without a matrix factorization."""
    unit_gradient, _ = _normalize_q_gradients(q_gradients)
    perpendicular_std = parallel_std * perpendicular_std_ratio
    standard_normal = jax.random.normal(rng, centers.shape, dtype=centers.dtype)
    parallel_projection = jnp.sum(
        unit_gradient * standard_normal, axis=-1, keepdims=True
    )
    noise = (
        perpendicular_std * standard_normal
        + (parallel_std - perpendicular_std)
        * unit_gradient
        * parallel_projection
    )
    return centers + noise


def sample_QgradCov_gaussian(
    rng: jax.Array,
    centers: jax.Array,
    q_gradients: jax.Array,
    repeats: int,
    std: float,
    include_anchor: bool = False,
    perpendicular_std_ratio: float = 0.5,
) -> jax.Array:
    """Sample untruncated Gaussians elongated along the local Q gradient.

    ``std`` is the parallel standard deviation and the perpendicular standard
    deviation is ``std * perpendicular_std_ratio``.  A zero Q gradient falls
    back to the isotropic perpendicular covariance.  Deterministic anchors
    requested through ``include_anchor`` are not Gaussian draws.
    """
    if centers.shape != q_gradients.shape:
        raise ValueError("q_gradients must have the same shape as centers")

    random_repeats = repeats - int(include_anchor)
    if random_repeats < 0:
        raise ValueError("repeats must be at least one when include_anchor=True")

    sample_shape = centers.shape[:-1] + (random_repeats, centers.shape[-1])
    sample_centers = jnp.broadcast_to(centers[..., None, :], sample_shape)
    sample_gradients = jnp.broadcast_to(q_gradients[..., None, :], sample_shape)
    random_samples = _sample_qgrad_cov_unbounded(
        rng,
        sample_centers,
        sample_gradients,
        std,
        perpendicular_std_ratio,
    )
    if include_anchor:
        return jnp.concatenate((centers[..., None, :], random_samples), axis=-2)
    return random_samples


def sample_QgradCov_gaussian_mixture(
    rng: jax.Array,
    centers: jax.Array,
    q_gradients: jax.Array,
    repeats: int,
    std: float,
    include_anchor: bool = False,
    perpendicular_std_ratio: float = 0.5,
    return_component_indices: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Sample IID from the uniform mixture of untruncated Q-gradient Gaussians."""
    if centers.ndim != 3:
        raise ValueError("centers must have shape [batch, components, action_dim]")
    if centers.shape != q_gradients.shape:
        raise ValueError("q_gradients must have the same shape as centers")

    random_repeats = repeats - int(include_anchor)
    if random_repeats < 0:
        raise ValueError("repeats must be at least one when include_anchor=True")

    batch_size, num_centers, _ = centers.shape
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
    selected_gradients = jax.vmap(
        lambda batch_gradients, batch_indices: batch_gradients[batch_indices]
    )(q_gradients, component_indices)
    random_samples = _sample_qgrad_cov_unbounded(
        sample_key,
        selected_centers,
        selected_gradients,
        std,
        perpendicular_std_ratio,
    )

    if include_anchor:
        samples = jnp.concatenate((centers[..., None, :], random_samples), axis=-2)
        anchor_indices = jnp.broadcast_to(
            jnp.arange(num_centers, dtype=component_indices.dtype)[None, :, None],
            (batch_size, num_centers, 1),
        )
        sample_component_indices = jnp.concatenate(
            (anchor_indices, component_indices), axis=-1
        )
    else:
        samples = random_samples
        sample_component_indices = component_indices

    if return_component_indices:
        return samples, sample_component_indices
    return samples


def _sample_qgrad_cov_bounded(
    rng: jax.Array,
    centers: jax.Array,
    q_gradients: jax.Array,
    parallel_std: float,
    perturb_clip: float,
    perpendicular_std_ratio: float,
) -> jax.Array:
    """Sample a gradient-aligned Gaussian copula with bounded marginals."""
    unit_gradient, marginal_std, _, _ = _qgrad_covariance_parameters(
        q_gradients,
        parallel_std,
        perpendicular_std_ratio,
    )
    perpendicular_std = parallel_std * perpendicular_std_ratio
    standard_normal = jax.random.normal(rng, centers.shape, dtype=centers.dtype)
    parallel_projection = jnp.sum(
        unit_gradient * standard_normal, axis=-1, keepdims=True
    )
    correlated_noise = (
        perpendicular_std * standard_normal
        + (parallel_std - perpendicular_std)
        * unit_gradient
        * parallel_projection
    )

    # Standardizing by the diagonal of Sigma gives a standard-normal vector
    # with Sigma's correlation matrix.  Its Gaussian copula is retained while
    # inverse-CDF transforming each marginal into the feasible action interval.
    correlated_uniform = jsp.special.ndtr(correlated_noise / marginal_std)
    lower = jnp.maximum(-1.0 - centers, -perturb_clip)
    upper = jnp.minimum(1.0 - centers, perturb_clip)
    lower_cdf = jsp.special.ndtr(lower / marginal_std)
    upper_cdf = jsp.special.ndtr(upper / marginal_std)
    quantiles = lower_cdf + correlated_uniform * (upper_cdf - lower_cdf)
    bounded_noise = marginal_std * jsp.special.ndtri(
        jnp.clip(quantiles, 1.0e-7, 1.0 - 1.0e-7)
    )
    return centers + bounded_noise


def sample_QgradCov_truncated_gaussian(
    rng: jax.Array,
    centers: jax.Array,
    q_gradients: jax.Array,
    repeats: int,
    std: float,
    perturb_clip: float,
    include_anchor: bool = False,
    perpendicular_std_ratio: float = 0.5,
) -> jax.Array:
    """Sample bounded proposals elongated along each center's Q gradient.

    Before truncation, the proposal covariance is

    ``std**2 uu^T + (std * perpendicular_std_ratio)**2 (I - uu^T)``,

    where ``u`` is the normalized Q gradient.  Axis-wise inverse-CDF
    truncation retains the corresponding Gaussian copula and guarantees both
    the local ``perturb_clip`` bound and the normalized action-space bound.
    A zero Q gradient falls back to the isotropic perpendicular covariance.
    """
    if centers.shape != q_gradients.shape:
        raise ValueError("q_gradients must have the same shape as centers")

    random_repeats = repeats - int(include_anchor)
    if random_repeats < 0:
        raise ValueError("repeats must be at least one when include_anchor=True")

    sample_shape = centers.shape[:-1] + (random_repeats, centers.shape[-1])
    sample_centers = jnp.broadcast_to(centers[..., None, :], sample_shape)
    sample_gradients = jnp.broadcast_to(q_gradients[..., None, :], sample_shape)
    random_samples = _sample_qgrad_cov_bounded(
        rng,
        sample_centers,
        sample_gradients,
        std,
        perturb_clip,
        perpendicular_std_ratio,
    )
    if include_anchor:
        return jnp.concatenate((centers[..., None, :], random_samples), axis=-2)
    return random_samples


def sample_QgradCov_truncated_gaussian_mixture(
    rng: jax.Array,
    centers: jax.Array,
    q_gradients: jax.Array,
    repeats: int,
    std: float,
    perturb_clip: float,
    include_anchor: bool = False,
    perpendicular_std_ratio: float = 0.5,
    return_component_indices: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Sample IID from the uniform mixture of Q-gradient covariances."""
    if centers.ndim != 3:
        raise ValueError("centers must have shape [batch, components, action_dim]")
    if centers.shape != q_gradients.shape:
        raise ValueError("q_gradients must have the same shape as centers")

    random_repeats = repeats - int(include_anchor)
    if random_repeats < 0:
        raise ValueError("repeats must be at least one when include_anchor=True")

    batch_size, num_centers, _ = centers.shape
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
    selected_gradients = jax.vmap(
        lambda batch_gradients, batch_indices: batch_gradients[batch_indices]
    )(q_gradients, component_indices)
    random_samples = _sample_qgrad_cov_bounded(
        sample_key,
        selected_centers,
        selected_gradients,
        std,
        perturb_clip,
        perpendicular_std_ratio,
    )

    if include_anchor:
        samples = jnp.concatenate((centers[..., None, :], random_samples), axis=-2)
        anchor_indices = jnp.broadcast_to(
            jnp.arange(num_centers, dtype=component_indices.dtype)[None, :, None],
            (batch_size, num_centers, 1),
        )
        sample_component_indices = jnp.concatenate(
            (anchor_indices, component_indices), axis=-1
        )
    else:
        samples = random_samples
        sample_component_indices = component_indices

    if return_component_indices:
        return samples, sample_component_indices
    return samples


def _sample_gamma_qgrad_cov(
    rng: jax.Array,
    centers: jax.Array,
    q_gradients: jax.Array,
    gamma_shape: float,
    gamma_scale: float,
    perpendicular_std: float,
) -> jax.Array:
    """Sample the untruncated Gamma-parallel, Gaussian-perpendicular law."""
    gamma_key, perpendicular_key = jax.random.split(rng)
    unit_gradient, _ = _normalize_q_gradients(q_gradients)
    gamma_steps = jax.random.gamma(
        gamma_key,
        jnp.asarray(gamma_shape, dtype=centers.dtype),
        shape=centers.shape[:-1],
        dtype=centers.dtype,
    ) * jnp.asarray(gamma_scale, dtype=centers.dtype)
    standard_normal = jax.random.normal(
        perpendicular_key, centers.shape, dtype=centers.dtype
    )
    projected_normal = jnp.sum(
        unit_gradient * standard_normal, axis=-1, keepdims=True
    )
    perpendicular_noise = perpendicular_std * (
        standard_normal - unit_gradient * projected_normal
    )
    return (
        centers
        + gamma_steps[..., None] * unit_gradient
        + perpendicular_noise
    )


def sample_GammaQgradCov_gaussian(
    rng: jax.Array,
    centers: jax.Array,
    q_gradients: jax.Array,
    repeats: int,
    gamma_shape: float,
    gamma_scale: float,
    perpendicular_std: float,
    include_anchor: bool = False,
) -> jax.Array:
    """Sample Gamma steps along Q gradients with perpendicular Gaussian noise.

    This function applies no action-space or local truncation.  For a nonzero Q
    gradient it samples ``T u + sigma_perp (I - uu^T) Z``, with independent
    Gamma ``T`` and standard-normal ``Z``.  A zero Q gradient falls back to an
    isotropic ``d``-dimensional Gaussian with standard deviation
    ``perpendicular_std``.  If requested, the anchor in slot zero is
    deterministic and is not a draw from this proposal law.
    """
    if centers.shape != q_gradients.shape:
        raise ValueError("q_gradients must have the same shape as centers")

    random_repeats = repeats - int(include_anchor)
    if random_repeats < 0:
        raise ValueError("repeats must be at least one when include_anchor=True")

    sample_shape = centers.shape[:-1] + (random_repeats, centers.shape[-1])
    sample_centers = jnp.broadcast_to(centers[..., None, :], sample_shape)
    sample_gradients = jnp.broadcast_to(q_gradients[..., None, :], sample_shape)
    random_samples = _sample_gamma_qgrad_cov(
        rng,
        sample_centers,
        sample_gradients,
        gamma_shape,
        gamma_scale,
        perpendicular_std,
    )
    if include_anchor:
        return jnp.concatenate((centers[..., None, :], random_samples), axis=-2)
    return random_samples


def sample_GammaQgradCov_gaussian_mixture(
    rng: jax.Array,
    centers: jax.Array,
    q_gradients: jax.Array,
    repeats: int,
    gamma_shape: float,
    gamma_scale: float,
    perpendicular_std: float,
    include_anchor: bool = False,
    return_component_indices: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Sample IID from the uniform mixture of Gamma-Q-gradient components.

    Random proposals choose their generating component independently and
    uniformly.  No action-space or local truncation is applied.  Deterministic
    anchors requested through ``include_anchor`` are not mixture draws.
    """
    if centers.ndim != 3:
        raise ValueError("centers must have shape [batch, components, action_dim]")
    if centers.shape != q_gradients.shape:
        raise ValueError("q_gradients must have the same shape as centers")

    random_repeats = repeats - int(include_anchor)
    if random_repeats < 0:
        raise ValueError("repeats must be at least one when include_anchor=True")

    batch_size, num_centers, _ = centers.shape
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
    selected_gradients = jax.vmap(
        lambda batch_gradients, batch_indices: batch_gradients[batch_indices]
    )(q_gradients, component_indices)
    random_samples = _sample_gamma_qgrad_cov(
        sample_key,
        selected_centers,
        selected_gradients,
        gamma_shape,
        gamma_scale,
        perpendicular_std,
    )

    if include_anchor:
        samples = jnp.concatenate((centers[..., None, :], random_samples), axis=-2)
        anchor_indices = jnp.broadcast_to(
            jnp.arange(num_centers, dtype=component_indices.dtype)[None, :, None],
            (batch_size, num_centers, 1),
        )
        sample_component_indices = jnp.concatenate(
            (anchor_indices, component_indices), axis=-1
        )
    else:
        samples = random_samples
        sample_component_indices = component_indices

    if return_component_indices:
        return samples, sample_component_indices
    return samples


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


def QgradCov_mixture_log_density(
    samples: jax.Array,
    centers: jax.Array,
    q_gradients: jax.Array,
    std: float,
    perpendicular_std_ratio: float = 0.5,
    mixture_weights: jax.Array | None = None,
) -> jax.Array:
    """Evaluate an untruncated Q-gradient Gaussian mixture density.

    The covariance inverse and determinant use the rank-one formulas for
    ``Sigma = sigma_perp^2 I + (std^2 - sigma_perp^2) uu^T``; no dense
    covariance matrix is formed.  The default mixture is uniform.  Optional
    nonnegative ``mixture_weights`` may have shape ``[components]`` or
    ``[batch, components]`` and are normalized per batch.
    """
    if centers.ndim != 3:
        raise ValueError("centers must have shape [batch, components, action_dim]")
    if centers.shape != q_gradients.shape:
        raise ValueError("q_gradients must have the same shape as centers")

    unit_gradient, _, perpendicular_variance, variance_gap = (
        _qgrad_covariance_parameters(
            q_gradients,
            std,
            perpendicular_std_ratio,
        )
    )
    differences = samples[:, :, None, :] - centers[:, None, :, :]
    unit_norm_squared = jnp.sum(jnp.square(unit_gradient), axis=-1)
    rank_one_variance = (
        perpendicular_variance + variance_gap * unit_norm_squared
    )
    projection = jnp.sum(
        differences * unit_gradient[:, None, :, :], axis=-1
    )
    inverse_rank_one_coefficient = variance_gap / (
        perpendicular_variance * rank_one_variance
    )
    quadratic = (
        jnp.sum(jnp.square(differences), axis=-1) / perpendicular_variance
        - inverse_rank_one_coefficient[:, None, :] * jnp.square(projection)
    )
    action_dim = centers.shape[-1]
    log_determinant = (
        action_dim * jnp.log(perpendicular_variance)
        + jnp.log(rank_one_variance / perpendicular_variance)
    )
    component_log_density = (
        -0.5 * quadratic
        - 0.5 * log_determinant[:, None, :]
        - 0.5 * action_dim * jnp.log(2.0 * jnp.pi)
    )
    if mixture_weights is None:
        return jsp.special.logsumexp(component_log_density, axis=-1) - jnp.log(
            centers.shape[1]
        )

    weights = jnp.asarray(mixture_weights, dtype=centers.dtype)
    if weights.ndim == 1:
        weights = jnp.broadcast_to(weights[None, :], centers.shape[:2])
    if weights.shape != centers.shape[:2]:
        raise ValueError(
            "mixture_weights must have shape [components] or [batch, components]"
        )
    weight_sum = jnp.sum(weights, axis=-1, keepdims=True)
    normalized_weights = weights / jnp.maximum(weight_sum, 1.0e-20)
    log_weights = jnp.where(
        normalized_weights > 0.0,
        jnp.log(normalized_weights),
        -jnp.inf,
    )
    return jsp.special.logsumexp(
        component_log_density + log_weights[:, None, :], axis=-1
    )


def QgradCov_truncated_mixture_log_density(
    samples: jax.Array,
    centers: jax.Array,
    q_gradients: jax.Array,
    std: float,
    perturb_clip: float,
    perpendicular_std_ratio: float = 0.5,
) -> jax.Array:
    """Evaluate the mixture density used by the Q-gradient sampler.

    This is the exact density of the Gaussian-copula, truncated-marginal
    construction in ``sample_QgradCov_truncated_gaussian[_mixture]``.  It does
    not form or decompose a dense covariance matrix.
    """
    if centers.ndim != 3:
        raise ValueError("centers must have shape [batch, components, action_dim]")
    if centers.shape != q_gradients.shape:
        raise ValueError("q_gradients must have the same shape as centers")

    unit_gradient, marginal_std, perpendicular_variance, variance_gap = (
        _qgrad_covariance_parameters(
            q_gradients,
            std,
            perpendicular_std_ratio,
        )
    )
    lower = jnp.maximum(-1.0 - centers, -perturb_clip)
    upper = jnp.minimum(1.0 - centers, perturb_clip)
    lower_cdf = jsp.special.ndtr(lower / marginal_std)
    upper_cdf = jsp.special.ndtr(upper / marginal_std)
    marginal_normalizer = jnp.maximum(upper_cdf - lower_cdf, 1.0e-20)

    differences = samples[:, :, None, :] - centers[:, None, :, :]
    standardized = differences / marginal_std[:, None, :, :]
    truncated_cdf = (
        jsp.special.ndtr(standardized) - lower_cdf[:, None, :, :]
    ) / marginal_normalizer[:, None, :, :]
    copula_coordinates = jsp.special.ndtri(
        jnp.clip(truncated_cdf, 1.0e-7, 1.0 - 1.0e-7)
    )

    log_marginal_density = jnp.sum(
        -0.5 * jnp.square(standardized)
        - jnp.log(marginal_std[:, None, :, :])
        - 0.5 * jnp.log(2.0 * jnp.pi)
        - jnp.log(marginal_normalizer[:, None, :, :]),
        axis=-1,
    )

    # Gaussian-copula correction.  D @ copula_coordinates follows N(0, Sigma),
    # where D contains Sigma's marginal standard deviations.  The determinant
    # and inverse below use the matrix determinant lemma and Sherman-Morrison
    # for Sigma = sigma_perp^2 I + variance_gap uu^T.
    latent_noise = marginal_std[:, None, :, :] * copula_coordinates
    unit_norm_squared = jnp.sum(jnp.square(unit_gradient), axis=-1)
    rank_one_variance = (
        perpendicular_variance + variance_gap * unit_norm_squared
    )
    action_dim = centers.shape[-1]
    log_determinant = (
        action_dim * jnp.log(perpendicular_variance)
        + jnp.log(rank_one_variance / perpendicular_variance)
    )
    projection = jnp.sum(
        latent_noise * unit_gradient[:, None, :, :], axis=-1
    )
    inverse_rank_one_coefficient = variance_gap / (
        perpendicular_variance * rank_one_variance
    )
    quadratic = (
        jnp.sum(jnp.square(latent_noise), axis=-1) / perpendicular_variance
        - inverse_rank_one_coefficient[:, None, :] * jnp.square(projection)
    )
    log_copula_density = (
        -0.5 * quadratic
        - 0.5 * log_determinant[:, None, :]
        + jnp.sum(jnp.log(marginal_std), axis=-1)[:, None, :]
        + 0.5 * jnp.sum(jnp.square(copula_coordinates), axis=-1)
    )
    log_density = log_marginal_density + log_copula_density

    in_support = jnp.all(
        (differences >= lower[:, None, :, :] - 1.0e-6)
        & (differences <= upper[:, None, :, :] + 1.0e-6),
        axis=-1,
    )
    log_density = jnp.where(in_support, log_density, -jnp.inf)
    return jsp.special.logsumexp(log_density, axis=-1) - jnp.log(centers.shape[1])


def GammaQgradCov_gaussian_mixture_log_density(
    samples: jax.Array,
    centers: jax.Array,
    q_gradients: jax.Array,
    gamma_shape: float,
    gamma_scale: float,
    perpendicular_std: float,
    mixture_weights: jax.Array | None = None,
) -> jax.Array:
    """Evaluate the uniform Gamma-Q-gradient mixture density.

    For every nonzero-gradient component this implements

    ``Gamma(u^T (y-x); shape, scale) * Normal_{d-1}(r; 0, sigma_perp^2 I)``

    on the forward half-space.  By default the mixture coefficient is ``1/N``:
    the stratified sampler uses equal fixed counts, while the IID sampler uses
    uniform component probabilities.  For deliberately unequal allocation,
    pass nonnegative ``mixture_weights`` with shape ``[components]`` or
    ``[batch, components]``; the weights are normalized per batch.  No
    action-space or local truncation correction is included.
    """
    if centers.ndim != 3:
        raise ValueError("centers must have shape [batch, components, action_dim]")
    if centers.shape != q_gradients.shape:
        raise ValueError("q_gradients must have the same shape as centers")

    unit_gradient, has_direction = _normalize_q_gradients(q_gradients)
    differences = samples[:, :, None, :] - centers[:, None, :, :]
    parallel_steps = jnp.sum(
        differences * unit_gradient[:, None, :, :], axis=-1
    )
    squared_norm = jnp.sum(jnp.square(differences), axis=-1)
    perpendicular_squared_norm = jnp.maximum(
        squared_norm - jnp.square(parallel_steps), 0.0
    )

    dtype = centers.dtype
    shape = jnp.asarray(gamma_shape, dtype=dtype)
    scale = jnp.asarray(gamma_scale, dtype=dtype)
    sigma_perp = jnp.asarray(perpendicular_std, dtype=dtype)
    nonnegative_steps = jnp.maximum(parallel_steps, 0.0)
    log_gamma_density = (
        jsp.special.xlogy(shape - 1.0, nonnegative_steps)
        - nonnegative_steps / scale
        - jsp.special.gammaln(shape)
        - shape * jnp.log(scale)
    )
    action_dim = centers.shape[-1]
    log_perpendicular_density = (
        -0.5 * perpendicular_squared_norm / jnp.square(sigma_perp)
        - (action_dim - 1)
        * (jnp.log(sigma_perp) + 0.5 * jnp.log(2.0 * jnp.pi))
    )
    directed_log_density = log_gamma_density + log_perpendicular_density
    directed_log_density = jnp.where(
        parallel_steps >= 0.0, directed_log_density, -jnp.inf
    )

    # The gradient direction is undefined at exactly zero.  The sampler drops
    # the Gamma term in that case and uses a full d-dimensional isotropic
    # Gaussian, so the density must make the same fallback.
    isotropic_log_density = (
        -0.5 * squared_norm / jnp.square(sigma_perp)
        - action_dim
        * (jnp.log(sigma_perp) + 0.5 * jnp.log(2.0 * jnp.pi))
    )
    component_log_density = jnp.where(
        has_direction[:, None, :],
        directed_log_density,
        isotropic_log_density,
    )
    if mixture_weights is None:
        return jsp.special.logsumexp(component_log_density, axis=-1) - jnp.log(
            centers.shape[1]
        )

    weights = jnp.asarray(mixture_weights, dtype=dtype)
    if weights.ndim == 1:
        weights = jnp.broadcast_to(weights[None, :], centers.shape[:2])
    if weights.shape != centers.shape[:2]:
        raise ValueError(
            "mixture_weights must have shape [components] or [batch, components]"
        )
    weight_sum = jnp.sum(weights, axis=-1, keepdims=True)
    normalized_weights = weights / jnp.maximum(weight_sum, 1.0e-20)
    log_weights = jnp.where(
        normalized_weights > 0.0,
        jnp.log(normalized_weights),
        -jnp.inf,
    )
    return jsp.special.logsumexp(
        component_log_density + log_weights[:, None, :], axis=-1
    )


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
