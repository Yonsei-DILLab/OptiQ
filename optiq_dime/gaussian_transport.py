"""Fresh Gaussian-likelihood OT. Potentials are dimensionless f / alpha."""
import math
from numbers import Real

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from .semi_implicit import tanh_log_jacobian


def source_log_density(query_u, source_mu, source_log_std, action_scale=1.):
    """Actual tanh Gaussian density [B,K,H], with a live query derivative."""
    if query_u.ndim != 3 or source_mu.ndim != 3 or source_mu.shape != source_log_std.shape:
        raise ValueError('Expected query[B,K,D], source mu/logstd[B,H,D]')
    if source_mu.shape[0] not in (1, query_u.shape[0]) or source_mu.shape[-1] != query_u.shape[-1]:
        raise ValueError('Source and query batch/dimension mismatch')
    delta = (query_u[:, :, None, :] - source_mu[:, None, :, :]) * jnp.exp(-source_log_std[:, None])
    normal = (-.5 * delta**2 - source_log_std[:, None] - .5 * math.log(2 * math.pi)).sum(-1)
    return normal - tanh_log_jacobian(query_u)[..., None] - query_u.shape[-1] * jnp.log(action_scale)


def gaussian_assignment_log_probs(query_u, source_mu, source_log_std, source_potential, action_scale=1.):
    densities = source_log_density(query_u, source_mu, source_log_std, action_scale)
    return jax.nn.log_softmax(densities + source_potential[:, None, :], axis=-1)


def fresh_balanced_sinkhorn(log_kernel, teacher_weights, *, max_iterations=500,
                           min_iterations=10, relative_tolerance=1e-3):
    """Solve each batch member independently, initially without a warm start.

    log_kernel[B,H,K] = log pi_old,i(teacher_j), equal to -C/alpha.
    Final columns are normalized exactly up to arithmetic; row tolerance is
    relative to 1/H, so tiny absolute errors cannot hide imbalance. A failed
    solve is returned with converged=False; the actor must reject its update.
    """
    if log_kernel.ndim != 3 or teacher_weights.shape != (log_kernel.shape[0], log_kernel.shape[2]):
        raise ValueError('Expected log kernel[B,H,K] and teacher weights[B,K]')
    if not 1 <= min_iterations <= max_iterations:
        raise ValueError('Require 1 <= min_iterations <= max_iterations')
    if isinstance(relative_tolerance, Real) and not 0 < relative_tolerance < 1:
        raise ValueError('relative_tolerance must lie between zero and one')
    kernel, weights = jax.lax.stop_gradient((log_kernel, teacher_weights))
    batch, sources, teachers = kernel.shape
    # A column constant affects neither the optimal plan nor query assignment.
    kernel = kernel - jnp.max(kernel, axis=1, keepdims=True)
    log_b = jnp.log(weights)
    log_a = -math.log(sources)
    initial_f = jnp.zeros((batch, sources), dtype=kernel.dtype)
    initial_g = jnp.zeros((batch, teachers), dtype=kernel.dtype)
    errors = jnp.full((batch,), jnp.inf, dtype=kernel.dtype)
    steps = jnp.zeros((batch,), dtype=jnp.int32)

    def condition(carry):
        iteration, _, _, error, _ = carry
        return (iteration < max_iterations) & ((iteration < min_iterations) | jnp.any(error > relative_tolerance))

    def iteration(carry):
        count, f, g, error, n = carry
        active = (count < min_iterations) | (error > relative_tolerance)
        new_f = log_a - logsumexp(kernel + g[:, None, :], axis=-1)
        new_g = log_b - logsumexp(kernel + new_f[:, :, None], axis=1)
        # Gauge centering keeps the two scaling factors numerically bounded.
        shift = new_f.mean(-1, keepdims=True)
        new_f, new_g = new_f - shift, new_g + shift
        log_plan = kernel + new_f[:, :, None] + new_g[:, None, :]
        row = jnp.exp(logsumexp(log_plan, axis=-1))
        new_error = jnp.max(jnp.abs(row * sources - 1.), axis=-1)
        new_error = jnp.where(jnp.isfinite(new_error), new_error, jnp.inf)
        return (count + 1, jnp.where(active[:, None], new_f, f),
                jnp.where(active[:, None], new_g, g), jnp.where(active, new_error, error),
                n + active.astype(jnp.int32))

    _, f, _, _, steps = jax.lax.while_loop(condition, iteration,
                                           (jnp.asarray(0), initial_f, initial_g, errors, steps))
    # Same softmax convention as the fresh-action assignment query.
    log_assignment = jax.nn.log_softmax(kernel + f[:, :, None], axis=1)
    plan = weights[:, None, :] * jnp.exp(log_assignment)
    row, column = plan.sum(-1), plan.sum(-2)
    row_relative = jnp.max(jnp.abs(row * sources - 1.), axis=-1)
    column_relative = jnp.max(jnp.abs(column - weights) / jnp.maximum(weights, jnp.finfo(weights.dtype).tiny), axis=-1)
    finite = jnp.all(jnp.isfinite(plan), axis=(1, 2)) & jnp.all(jnp.isfinite(f), axis=-1)
    valid_weights = jnp.all(weights > 0, axis=-1) & (jnp.abs(weights.sum(-1) - 1.) < 1e-5)
    converged = finite & valid_weights & (row_relative <= relative_tolerance) & (column_relative <= relative_tolerance)
    return jax.lax.stop_gradient(dict(
        plan=plan, source_potential=f, source_mass=row, teacher_mass=column,
        log_assignment=log_assignment, iterations=steps, converged=converged,
        row_relative_error=row_relative, column_relative_error=column_relative,
        row_error=jnp.max(jnp.abs(row - 1/sources), axis=-1),
        column_error=jnp.max(jnp.abs(column - weights), axis=-1),
        source_tv=.5*jnp.abs(row-1/sources).sum(-1)))
