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


def sample_gradient_skewed_mixture(
    rng: jax.Array,
    centers: jax.Array,
    gradients: jax.Array,
    repeats: int,
    std: float,
    perturb_clip: float,
    include_anchor: bool = False,
) -> jax.Array:
    """Sample IID proposals from a pairwise-skewed truncated mixture.

    Every random proposal independently selects an actor center.  The component
    first draws ``delta`` from an isotropic Gaussian conditioned on
    ``||delta|| <= perturb_clip``.  It keeps ``delta`` with probability

        sigmoid(2 * unit_grad_Q.T @ delta / std)

    and otherwise reflects it to ``-delta``.  The symmetric truncation makes
    the resulting component density exactly normalized.  Original actor
    samples can be retained as deterministic anchors in slot zero.
    """
    if centers.ndim != 3 or gradients.shape != centers.shape:
        raise ValueError(
            "centers and gradients must have shape [batch, components, action_dim]"
        )

    random_repeats = repeats - int(include_anchor)
    if random_repeats < 0:
        raise ValueError("repeats must be at least one when include_anchor=True")

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
    gradient_norms = jnp.linalg.norm(selected_gradients, axis=-1, keepdims=True)
    selected_directions = jnp.where(
        gradient_norms > 1.0e-8,
        selected_gradients / jnp.maximum(gradient_norms, 1.0e-8),
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
    maximum_radius = perturb_clip / std
    squared_radii = _sample_truncated_chi_square(
        radius_key,
        (batch_size, num_centers, random_repeats, 1),
        action_dim,
        maximum_radius,
        centers.dtype,
    )
    whitened_noise = spherical_directions * jnp.sqrt(squared_radii)
    base_noise = std * whitened_noise
    keep_logits = (
        2.0
        * jnp.sum(selected_directions * base_noise, axis=-1)
        / std
    )
    keep = jax.random.bernoulli(sign_key, jax.nn.sigmoid(keep_logits))
    oriented_noise = jnp.where(keep[..., None], base_noise, -base_noise)
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
) -> jax.Array:
    """Evaluate the exact pairwise-skewed proposal-mixture log density."""
    differences = samples[:, :, None, :] - centers[:, None, :, :]
    squared_mahalanobis = jnp.sum(jnp.square(differences / std), axis=-1)

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
        - log_radial_normalizer
    )
    gradient_norms = jnp.linalg.norm(gradients, axis=-1, keepdims=True)
    directions = jnp.where(
        gradient_norms > 1.0e-8,
        gradients / jnp.maximum(gradient_norms, 1.0e-8),
        0.0,
    )
    skew_logits = (
        2.0 * jnp.sum(differences * directions[:, None, :, :], axis=-1) / std
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


def _positive_kl_integrand(
    reference_weights: jax.Array,
    weights: jax.Array,
    log_ratio: jax.Array,
) -> jax.Array:
    """KL as a sum of nonnegative terms, without first-order cancellation.

    For normalized distributions, KL(w || r) = sum r * (u log u - u + 1),
    where u = w/r. Two equivalent forms avoid exponentiating large positive
    log-ratios. Taylor polynomials near zero preserve KLs well below float32
    epsilon, unlike summing signed ``w * (log(w) - log(r))`` terms.
    """
    positive = jnp.maximum(log_ratio, 0.0)
    negative = jnp.minimum(log_ratio, 0.0)
    # g(t) = t + exp(-t) - 1, used with w for t >= 0.
    small_positive = jnp.minimum(positive, 0.1)
    g_series = jnp.square(small_positive) * (
        0.5 + small_positive * (
            -1.0 / 6.0 + small_positive * (
                1.0 / 24.0 + small_positive * (
                    -1.0 / 120.0 + small_positive * (
                        1.0 / 720.0 + small_positive * (
                            -1.0 / 5040.0 + small_positive / 40320.0
                        )
                    )
                )
            )
        )
    )
    g = jnp.where(positive < 0.1, g_series, positive + jnp.expm1(-positive))
    # h(t) = exp(t)*t - expm1(t), used with r for t < 0.
    small_negative = jnp.maximum(negative, -0.1)
    h_series = jnp.square(small_negative) * (
        0.5 + small_negative * (
            1.0 / 3.0 + small_negative * (
                1.0 / 8.0 + small_negative * (
                    1.0 / 30.0 + small_negative * (
                        1.0 / 144.0 + small_negative * (
                            1.0 / 840.0 + small_negative / 5760.0
                        )
                    )
                )
            )
        )
    )
    h = jnp.where(
        negative > -0.1,
        h_series,
        jnp.exp(negative) * negative - jnp.expm1(negative),
    )
    return jnp.maximum(
        jnp.sum(jnp.where(log_ratio >= 0.0, weights * g, reference_weights * h), axis=-1),
        0.0,
    )


def _exponential_tilt_kl(
    reference_logits: jax.Array,
    tilt: jax.Array,
) -> jax.Array:
    """Stable KL(softmax(reference_logits + tilt) || softmax(reference_logits))."""
    reference_weights = jax.nn.softmax(reference_logits, axis=-1)
    # The normalization removes a constant from tilt anyway. Centering first
    # lets log1p/expm1 resolve tiny changes around a nearly flat reference.
    centered = tilt - jnp.sum(reference_weights * tilt, axis=-1, keepdims=True)
    small = jnp.max(jnp.abs(centered), axis=-1, keepdims=True) < 0.5
    clipped = jnp.clip(centered, -0.5, 0.5)
    small_log_normalizer = jnp.log1p(
        jnp.sum(reference_weights * jnp.expm1(clipped), axis=-1, keepdims=True)
    )
    large_log_normalizer = jsp.special.logsumexp(
        jax.nn.log_softmax(reference_logits, axis=-1) + centered,
        axis=-1,
        keepdims=True,
    )
    log_ratio = centered - jnp.where(small, small_log_normalizer, large_log_normalizer)
    weights = jax.nn.softmax(reference_logits + centered, axis=-1)
    return _positive_kl_integrand(reference_weights, weights, log_ratio)


def select_density_beta_for_kl(
    q_score: jax.Array,
    density_score: jax.Array,
    bisection_iterations: int = 32,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Maximal beta in [0, 1] with KL(w_beta || r) <= KL(uniform || r).

    ``r = softmax(q_score)`` and
    ``w_beta = softmax(q_score + beta * density_score)``. The last axis holds
    candidates; selection is independent across all leading axes. The budget
    multiplier is exactly one, with no additional tunable coefficient.

    The budget direction is essential: for u=uniform and x=q_score,
    E_w[x] - E_u[x] = KL(w || u) + KL(u || r) - KL(w || r).
    Feasibility therefore guarantees E_w[x] - E_u[x] >= KL(w || u) >= 0
    in exact arithmetic, on this fixed candidate set. The former KL(r || u)
    budget does not provide this expected-score improvement guarantee.

    KL(w_beta || r) is monotone since its beta derivative is
    beta * Var_{w_beta}(density_score). Bisection returns the feasible lower
    endpoint, except beta=1 is returned exactly when full correction is feasible.
    Flat Q gives beta=0 unless density is constant (all betas then agree, so 1).

    Inputs must be finite. Computation uses at least float32 and a cancellation-
    resistant KL, important when Q-only KL is around 1e-10. Differences already
    rounded out of input scores cannot be recovered. This is a finite-candidate
    KL constraint, not a guaranteed population-KL, minimum-ESS, or actor-return
    improvement constraint. Distillation and KDE smoothing introduce errors.
    """
    if q_score.shape != density_score.shape or q_score.ndim < 1 or q_score.shape[-1] < 1:
        raise ValueError("q_score and density_score must have matching, nonempty candidate axes")
    if bisection_iterations < 1:
        raise ValueError("bisection_iterations must be positive")
    dtype = jnp.result_type(q_score, density_score, jnp.float32)
    q_score = jnp.asarray(q_score, dtype=dtype)
    density_score = jnp.asarray(density_score, dtype=dtype)
    # Remove large offsets before taking means; flatness is retained exactly.
    q_centered = q_score - jnp.max(q_score, axis=-1, keepdims=True)
    q_centered -= jnp.mean(q_centered, axis=-1, keepdims=True)
    density_centered = density_score - jnp.max(density_score, axis=-1, keepdims=True)
    density_centered -= jnp.mean(density_centered, axis=-1, keepdims=True)
    # Starting from r and undoing its Q tilt gives u, hence KL(u || r).
    # Keep the cancellation-resistant helper for nearly flat Q in float32.
    kl_budget = _exponential_tilt_kl(q_centered, -q_centered)

    def correction_kl(beta):
        return _exponential_tilt_kl(q_centered, beta[..., None] * density_centered)

    zero = jnp.zeros(q_score.shape[:-1], dtype=dtype)
    one = jnp.ones_like(zero)
    full_kl = correction_kl(one)

    def iteration(_, bounds):
        lower, upper = bounds
        midpoint = lower + 0.5 * (upper - lower)
        feasible = correction_kl(midpoint) <= kl_budget
        return jnp.where(feasible, midpoint, lower), jnp.where(feasible, upper, midpoint)

    lower, _ = jax.lax.fori_loop(0, bisection_iterations, iteration, (zero, one))
    selected_beta = jnp.where(full_kl <= kl_budget, one, lower)
    # Enforce the exact flat-Q endpoint even when a nonconstant density's KL
    # underflows to zero. A constant density leaves every beta equivalent.
    flat_q = jnp.all(q_score == q_score[..., :1], axis=-1)
    constant_density = jnp.all(density_score == density_score[..., :1], axis=-1)
    selected_beta = jnp.where(
        flat_q, jnp.where(constant_density, one, zero), selected_beta
    )
    selected_kl = correction_kl(selected_beta)
    return selected_beta, selected_kl, kl_budget


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
