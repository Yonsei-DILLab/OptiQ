"""Sampling-only proximal soft-policy target, for proposals equal to the policy.

At a fixed state the ideal target is pi_old**(1-eta) * exp(eta*Q/T).
Sampling from pi_old therefore uses weights exp(eta*(Q/T-log pi_old)).
This changes the policy step, not the temperature of the soft Bellman objective.
The ESS constraint is a sample-budget heuristic, not an improvement certificate.
"""
import jax
import jax.numpy as jnp


def proximal_policy_weights(q_values, log_policy_density, temperature, minimum_ess):
    """Return stopped weights and the largest eta in [0,1] meeting sample ESS.

    Inputs have shape [batch, candidates]. Candidates MUST follow the actual
    old policy; a floored/broadened proposal requires a different IS ratio.
    Configuration validation must enforce finite positive T and 1<=ESS<=K.
    """
    scores = jax.lax.stop_gradient(q_values / temperature - log_policy_density)
    scores = scores - scores.max(axis=-1, keepdims=True)

    def weights_at(fraction):
        return jax.nn.softmax(fraction[:, None] * scores, axis=-1)

    def ess_at(fraction):
        return 1.0 / jnp.square(weights_at(fraction)).sum(axis=-1)

    lower = jnp.zeros(scores.shape[0], dtype=scores.dtype)
    upper = jnp.ones_like(lower)

    def bisect(_, interval):
        lo, hi = interval
        mid = (lo + hi) / 2
        feasible = ess_at(mid) >= minimum_ess
        return jnp.where(feasible, mid, lo), jnp.where(feasible, hi, mid)

    lower, _ = jax.lax.fori_loop(0, 24, bisect, (lower, upper))
    fraction = jnp.where(ess_at(upper) >= minimum_ess, upper, lower)
    fraction = jnp.where(minimum_ess >= scores.shape[-1], 0., fraction)
    return jax.lax.stop_gradient(weights_at(fraction)), jax.lax.stop_gradient(fraction)
