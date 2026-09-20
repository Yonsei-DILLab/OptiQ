"""Fresh latent-to-teacher OT and differentiable conditional assignments.

The source is a finite bank of latent coordinates with uniform mass. Teacher
weights already include value weighting and proposal-density correction; this
module does not apply either correction again. Unlike the v5 action-space cost,
the v7 cost compares latent coordinates with pre-tanh teacher coordinates.
"""

from numbers import Real

import jax
import jax.numpy as jnp
import jax.scipy as jsp


def _check_point_shapes(left, right):
    if left.ndim != 3 or right.ndim != 3:
        raise ValueError("Points must have shape [batch, count, dimension]")
    if left.shape[0] != right.shape[0] or left.shape[-1] != right.shape[-1]:
        raise ValueError("Point banks must have the same batch and coordinate dimension")
    if left.shape[1] == 0 or right.shape[1] == 0:
        raise ValueError("Point banks must be nonempty")


def _check_epsilon(epsilon):
    # Numeric hyperparameters can be checked before JIT. A traced epsilon remains
    # a runtime input, as in the existing RL training function.
    if isinstance(epsilon, Real) and not 0.0 < epsilon < float("inf"):
        raise ValueError("epsilon must be positive and finite")


def latent_transport_cost(anchors: jax.Array, teacher_u: jax.Array) -> jax.Array:
    """Return unnormalized squared distances, shape ``[B, N, M]``.

    Inputs have shapes ``anchors[B,N,D]`` and ``teacher_u[B,M,D]``. Explicit
    differences avoid cancellation in the squared-norm/dot-product identity and
    avoid reduced-precision GPU matrix multiplication at small OT epsilon.
    This cost helper itself is differentiable; the solve below freezes its inputs.
    """
    anchors, teacher_u = jnp.asarray(anchors), jnp.asarray(teacher_u)
    _check_point_shapes(anchors, teacher_u)
    return jnp.sum(jnp.square(anchors[:, :, None, :] - teacher_u[:, None, :, :]), axis=-1)


def sinkhorn_with_source_potential(
    costs: jax.Array,
    teacher_weights: jax.Array,
    epsilon: float,
    iterations: int,
) -> dict[str, jax.Array]:
    """Solve balanced entropic OT and return a frozen source potential.

    ``costs`` is ``[B,N,M]``; ``teacher_weights`` is ``[B,M]``. The row target is
    ``1/N``. Teacher weights receive the same floor and normalization as the v5
    solver. Both inputs and all outputs are stop-gradient labels for the actor.

    Returned fields:
      * ``plan[B,N,M]`` and ``source_potential[B,N]``;
      * actual ``source_mass[B,N]`` and ``teacher_mass[B,M]``;
      * maximum absolute ``row_error[B]`` and ``column_error[B]``;
      * ``source_tv[B]``, half the row-marginal L1 error.

    The final column projection makes teacher marginals exact up to roundoff.
    A finite iteration budget can still leave source-marginal error, which must
    be monitored rather than interpreting "balanced" as an exact finite solve.
    The potential includes source-marginal scaling: do not add log(1/N) to it.
    """
    costs, teacher_weights = jnp.asarray(costs), jnp.asarray(teacher_weights)
    if costs.ndim != 3 or teacher_weights.shape != (costs.shape[0], costs.shape[-1]):
        raise ValueError("Expected costs[B,N,M] and teacher_weights[B,M]")
    if costs.shape[1] == 0 or costs.shape[2] == 0:
        raise ValueError("Both transport marginals must be nonempty")
    if isinstance(iterations, int) and iterations < 1:
        raise ValueError("At least one Sinkhorn iteration is required")
    _check_epsilon(epsilon)

    costs = jax.lax.stop_gradient(costs)
    teacher_weights = jax.lax.stop_gradient(teacher_weights)
    epsilon = jax.lax.stop_gradient(jnp.asarray(epsilon, costs.dtype))
    batch_size, num_rows, _ = costs.shape
    row_weights = jnp.full((batch_size, num_rows), 1.0 / num_rows, costs.dtype)
    teacher_weights = jnp.clip(teacher_weights, 1.0e-20, 1.0)
    teacher_weights = teacher_weights / teacher_weights.sum(axis=-1, keepdims=True)
    log_kernel = -costs / epsilon
    log_rows = jnp.log(row_weights)
    log_columns = jnp.log(teacher_weights)

    def iteration(_, potentials):
        _, log_v = potentials
        log_u = log_rows - jsp.special.logsumexp(log_kernel + log_v[:, None, :], axis=-1)
        log_v = log_columns - jsp.special.logsumexp(log_kernel + log_u[:, :, None], axis=-2)
        return log_u, log_v

    log_u, _ = jax.lax.fori_loop(
        0, iterations, iteration, (jnp.zeros_like(log_rows), jnp.zeros_like(log_columns))
    )
    # Choose a gauge with zero-mean source potential. Constructing the final
    # column projection as W * softmax avoids cancellation with a large log_v.
    centered_log_u = log_u - jnp.mean(log_u, axis=-1, keepdims=True)
    source_potential = epsilon * centered_log_u
    # Use the same arithmetic as the out-of-sample assignment helper. In sharp
    # float32 solves, (f - c) / epsilon need not round identically to f/epsilon
    # - c/epsilon; both the training labels and actor should use one conditional.
    log_assignments = jax.nn.log_softmax((source_potential[:, :, None] - costs) / epsilon, axis=1)
    plan = teacher_weights[:, None, :] * jnp.exp(log_assignments)
    source_mass, teacher_mass = plan.sum(axis=-1), plan.sum(axis=-2)
    result = {
        "plan": plan,
        "source_potential": source_potential,
        "source_mass": source_mass,
        "teacher_mass": teacher_mass,
        "row_error": jnp.max(jnp.abs(source_mass - row_weights), axis=-1),
        "column_error": jnp.max(jnp.abs(teacher_mass - teacher_weights), axis=-1),
        "source_tv": 0.5 * jnp.sum(jnp.abs(source_mass - row_weights), axis=-1),
    }
    return jax.tree_util.tree_map(jax.lax.stop_gradient, result)


def latent_assignment_log_probs(
    query_u: jax.Array,
    anchors: jax.Array,
    source_potential: jax.Array,
    epsilon: float,
) -> jax.Array:
    """Return ``log r_i(u)`` with live query gradients, shape ``[B,K,N]``.

    The anchors ``[B,N,D]`` and OT potential ``[B,N]`` are fixed labels; actions
    ``query_u[B,K,D]`` retain their reparameterized actor gradient. K can differ
    from N, and queries need not be teacher samples used in the solve. Coordinates
    are pre-tanh, so callers should keep the sampled u rather than invert tanh.
    """
    query_u, anchors = jnp.asarray(query_u), jnp.asarray(anchors)
    source_potential = jnp.asarray(source_potential)
    _check_point_shapes(query_u, anchors)
    if source_potential.shape != anchors.shape[:2]:
        raise ValueError("source_potential must have shape [batch, anchor_count]")
    _check_epsilon(epsilon)
    anchors = jax.lax.stop_gradient(anchors)
    source_potential = jax.lax.stop_gradient(source_potential)
    epsilon = jax.lax.stop_gradient(jnp.asarray(epsilon, query_u.dtype))
    costs = latent_transport_cost(query_u, anchors)
    return jax.nn.log_softmax((source_potential[:, None, :] - costs) / epsilon, axis=-1)
