"""Opt-in fixed-Q ablation: balance log-sigma gradients across OT rows."""
import jax
import jax.numpy as jnp

from optiq_dime.distillation import conditional_ot_nll


def sigma_row_balanced_ot_nll(mu, log_std, teacher_u, row_probabilities):
    """Preserve NLL value and dL/dmu; multiply dL/dlog_std by stopped weights.

    A_id = E_row[(u_jd-mu_id)^2], rho_i = mean_d(A_id/sigma_id^2).
    a_i = 1/max(1,rho_i); w_i = a_i/mean_rows(a_i), independently per cloud.

    This is a gradient intervention, not a new ordinary joint NLL objective.
    The shared actor trunk receives the modified sigma-path gradient, so later
    mean outputs/updates need not coincide with the baseline trajectory.
    """
    targets = jax.lax.stop_gradient(teacher_u)
    rows = jax.lax.stop_gradient(row_probabilities)
    target_mean = jnp.einsum('bnm,bmd->bnd', rows, targets)
    variance = jnp.maximum(jnp.einsum('bnm,bmd->bnd', rows, targets**2)-target_mean**2, 0)
    residual = (mu-target_mean)**2+variance
    ratios = jax.lax.stop_gradient(residual*jnp.exp(-2*log_std))
    raw_weight = 1/jnp.maximum(1, ratios.mean(-1))
    weight = jax.lax.stop_gradient(raw_weight/jnp.maximum(raw_weight.mean(-1, keepdims=True), 1e-20))
    # Exact forward identity; the only backward path through log_std has w_i.
    detached_ls = jax.lax.stop_gradient(log_std)
    balanced_ls = detached_ls+weight[..., None]*(log_std-detached_ls)
    value = conditional_ot_nll(mu, balanced_ls, teacher_u, row_probabilities)
    original_gradient = 1-ratios
    expansion = jnp.maximum(-original_gradient, 0)
    diagnostics = dict(
        sigma_row_weight_mean=weight.mean(),
        sigma_row_weight_min=weight.min(),
        sigma_row_weight_max=weight.max(),
        sigma_row_downweight_fraction=(weight<1).astype(mu.dtype).mean(),
        sigma_row_weight_ess_fraction=(1/jnp.mean(weight**2, axis=-1)).mean(),
        sigma_raw_scale_mean=raw_weight.mean(),
        sigma_gradient_original_mean=original_gradient.mean(),
        sigma_gradient_balanced_mean=(weight[..., None]*original_gradient).mean(),
        sigma_expansion_pressure_original=expansion.mean(),
        sigma_expansion_pressure_balanced=(weight[..., None]*expansion).mean(),
    )
    return value, diagnostics
