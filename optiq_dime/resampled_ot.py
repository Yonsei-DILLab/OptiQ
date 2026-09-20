"""Importance resampling followed by uniform-marginal entropic OT.

Candidates and weights remain the original 64-point importance estimate.
The solver sees 16 resampled slots, including duplicates. Aggregating duplicate
columns back to the original candidate indices is algebraically identical for
the full-row conditional NLL, but avoids misleading duplicate-slot diagnostics.
"""
import jax
import jax.numpy as jnp
from .transport import sinkhorn

METRICS = (
    'resample_unique_candidates', 'resample_mass_tv', 'resample_mass_ess',
    'resample_count', 'ot_slot_row_ess', 'ot_unique_row_ess',
    'ot_slot_row_max', 'ot_unique_row_max', 'ot_unique_row_entropy',
    'ot_slot_row_marginal_error', 'ot_slot_col_marginal_error',
    'ot_original_weight_col_tv', 'ot_row_overlap',
)


def multinomial_resample(key, weights, count):
    """K independent categorical draws with replacement using inverse CDF."""
    weights = jax.lax.stop_gradient(weights)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    cdf = jnp.cumsum(weights, axis=-1).at[:, -1].set(1.0)
    u = jax.random.uniform(key, (weights.shape[0], count), dtype=weights.dtype)
    indices = (u[:, :, None] >= cdf[:, None, :]).sum(axis=-1)
    return jnp.minimum(indices, weights.shape[-1] - 1).astype(jnp.int32)


def resampled_transport(key, positions, candidates, weights, epsilon, iterations,
                        normalize_cost):
    count = positions.shape[1]
    indices = multinomial_resample(key, weights, count)
    slots = jax.vmap(lambda b, idx: b[idx])(candidates, indices)
    raw_cost = jnp.square(positions[:, :, None] - slots[:, None]).sum(axis=-1)
    cost = raw_cost / (raw_cost.mean(axis=(-2, -1), keepdims=True) + 1e-8) if normalize_cost else raw_cost
    uniform = jnp.full((positions.shape[0], count), 1.0/count, dtype=weights.dtype)
    slot_plan = sinkhorn(cost, uniform, epsilon, iterations)
    incidence = jax.nn.one_hot(indices, candidates.shape[1], dtype=weights.dtype)
    plan = jnp.einsum('bnk,bkm->bnm', slot_plan, incidence)
    resampled_mass = incidence.mean(axis=1)
    rows = plan / jnp.maximum(plan.sum(axis=-1, keepdims=True), 1e-20)
    slot_rows = slot_plan / jnp.maximum(slot_plan.sum(axis=-1, keepdims=True), 1e-20)
    row_dot = jnp.einsum('bnm,bkm->bnk', rows, rows)
    norms = jnp.sqrt(jnp.square(rows).sum(axis=-1))
    cosine = row_dot / jnp.maximum(norms[:, :, None]*norms[:, None, :], 1e-20)
    off_diagonal = 1.0-jnp.eye(count)[None]
    metrics = dict(
        resample_unique_candidates=(resampled_mass > 0).sum(axis=-1).mean(),
        resample_mass_tv=(.5*jnp.abs(resampled_mass-weights).sum(axis=-1)).mean(),
        resample_mass_ess=(1/jnp.square(resampled_mass).sum(axis=-1)).mean(),
        resample_count=jnp.asarray(count, dtype=weights.dtype),
        ot_slot_row_ess=(1/jnp.square(slot_rows).sum(axis=-1)).mean(),
        ot_unique_row_ess=(1/jnp.square(rows).sum(axis=-1)).mean(),
        ot_slot_row_max=slot_rows.max(axis=-1).mean(),
        ot_unique_row_max=rows.max(axis=-1).mean(),
        ot_unique_row_entropy=-(rows*jnp.log(jnp.maximum(rows, 1e-30))).sum(axis=-1).mean(),
        ot_slot_row_marginal_error=jnp.abs(slot_plan.sum(axis=-1)-uniform).mean(),
        ot_slot_col_marginal_error=jnp.abs(slot_plan.sum(axis=-2)-uniform).mean(),
        ot_original_weight_col_tv=(.5*jnp.abs(plan.sum(axis=-2)-weights).sum(axis=-1)).mean(),
        ot_row_overlap=(cosine*off_diagonal).sum()/ (positions.shape[0]*count*(count-1)),
    )
    return jax.lax.stop_gradient(plan), jax.lax.stop_gradient(resampled_mass), raw_cost, metrics
