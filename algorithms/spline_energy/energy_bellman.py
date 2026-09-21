"""Relative energy Bellman regression; the spline model/sampler are unchanged.

Derivation and limits: THEORY.md. This is a target-relative generalized KL
adaptation, not the original GMM40 importance-weighted integrated objective,
and not an implementation of XQL. Temperature alpha defines the policy;
scale eta defines regression geometry in reward units.
"""
import math

import jax.numpy as jnp


def relative_energy_loss(residual, scale=10.0, tail_start=4.0):
    """eta^2 phi_k((Q - stop_gradient(target))/eta), elementwise.

    phi(x)=exp(x)-x-1 up to k; beyond k use its quadratic Taylor
    continuation. Both value and the first two derivatives match at k.
    Exponentiation is bounded by exp(k), without clipping the residual or
    losing gradients in the positive tail. The caller freezes the target.
    """
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")
    if not math.isfinite(tail_start) or not 0 < tail_start <= 20:
        raise ValueError("tail_start must lie in (0, 20]")
    x = jnp.asarray(residual) / scale
    # The unselected branch is finite too, so autodiff cannot see exp(large x).
    base_x = jnp.where(x <= tail_start, x, tail_start)
    tail = x - base_x
    phi = (jnp.expm1(base_x) - base_x
           + math.expm1(tail_start) * tail
           + .5 * math.exp(tail_start) * tail**2)
    return scale**2 * phi


def residual_bound_from_uniform_loss(loss_upper_bound, scale=10.0):
    """Invert phi(-t) to bound |Q-y| given a TRUE uniform loss bound.

    A minibatch mean/max is not a uniform loss bound. This routine computes
    the implication, not a certificate that the premise holds in MuJoCo.
    """
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")
    if not math.isfinite(loss_upper_bound) or loss_upper_bound < 0:
        raise ValueError("loss_upper_bound must be finite and nonnegative")
    u = loss_upper_bound / scale**2
    if u == 0:
        return 0.
    lo, hi = 0., u + 1.
    for _ in range(100):
        mid = .5 * (lo + hi)
        if mid + math.expm1(-mid) < u:
            lo = mid
        else:
            hi = mid
    return scale * hi


def bounds_from_uniform_residual(residual_upper_bound, gamma, temperature):
    """Conditional bounds for the ordinary EXPECTED soft Bellman residual.

    No neural-network convergence or empirical coverage is certified here.
    """
    if not math.isfinite(residual_upper_bound) or residual_upper_bound < 0:
        raise ValueError("residual bound must be finite and nonnegative")
    if not 0 <= gamma < 1 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("need gamma in [0,1) and positive finite temperature")
    q_error = residual_upper_bound / (1. - gamma)
    return {"q_sup_error_bound": q_error,
            "soft_value_suboptimality_bound": 2. * q_error,
            "policy_kl_either_direction_bound": 2. * q_error / temperature}
