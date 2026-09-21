"""EM-auxiliary shared-actor transfer for the fixed-weight SMEM teacher."""
import jax
import jax.numpy as jnp
import optax

from smem_tr import (box_kl, fit_teacher, log_components, responsibilities,
                     validate_config, weighted_nll)


def distill_actor(state, z, old, fitted, x, w, cfg):
    """Take one backtracked generalized-EM actor step.

    Teacher responsibilities and SNIS weights are stopped. Acceptance uses the
    actual marginal mixture NLL and the same joint component KL radius as the
    original SMEM implementation.
    """
    initial_nll = weighted_nll(old, x, w)
    posterior = jax.lax.stop_gradient(responsibilities(fitted, x))
    weighted_posterior = w[None] * posterior

    def outputs(params):
        return state.apply_fn({"params": params}, z, return_raw=True)

    def objective(params):
        mu, ls, _ = outputs(params)
        return -(weighted_posterior * log_components((mu, ls), x)).sum()

    def evaluate(params):
        mu, ls, _ = outputs(params)
        return weighted_nll((mu, ls), x, w), box_kl(mu, ls, *old).mean()

    loss, grad = jax.value_and_grad(objective)(state.params)

    def search(start):
        proposed = start.apply_gradients(grads=grad)
        finite_state = jnp.all(jnp.stack([
            jnp.all(jnp.isfinite(v))
            for v in jax.tree.leaves((proposed.params, proposed.opt_state))
        ]))
        delta = jax.tree.map(lambda new, previous: new-previous,
                             proposed.params, state.params)

        def attempt(i, best):
            params, nll, taken, fraction, kl = best

            def evaluate_candidate(_):
                rate = .5**i
                candidate = jax.tree.map(
                    lambda previous, change: previous + rate*change,
                    state.params, delta)
                candidate_nll, candidate_kl = evaluate(candidate)
                valid = finite_state & jnp.isfinite(candidate_nll) & jnp.isfinite(candidate_kl)
                valid &= (candidate_nll < initial_nll) & (candidate_kl <= cfg["tr_kl"])
                return (jax.tree.map(lambda new, old: jnp.where(valid, new, old),
                                     candidate, params),
                        jnp.where(valid, candidate_nll, nll), valid,
                        jnp.where(valid, rate, fraction),
                        jnp.where(valid, candidate_kl, kl))

            return jax.lax.cond(taken, lambda _: best, evaluate_candidate, None)

        params, nll, accepted, fraction, kl = jax.lax.fori_loop(
            0, cfg["actor_backtracks"], attempt,
            (state.params, initial_nll, jnp.array(False), jnp.array(0.), jnp.array(0.)))
        selected = jax.tree.map(lambda new, old: jnp.where(accepted, new, old),
                                proposed, state)
        return selected.replace(params=params), nll, accepted, fraction, kl

    candidate, nll, accepted, fraction, kl = search(state)
    first_accepted = accepted

    def restart(_):
        reset = state.replace(opt_state=state.tx.init(state.params))
        return search(reset)

    best, nll, accepted, fraction, kl = jax.lax.cond(
        accepted, lambda _: (candidate, nll, accepted, fraction, kl), restart, None)
    momentum_reset = (~first_accepted & accepted).astype(float)
    best = best.replace(step=state.step+1)
    mu, ls, raw_ls = outputs(best.params)
    final_nll = weighted_nll((mu, ls), x, w)
    final_kl = box_kl(mu, ls, *old).mean()
    valid = jnp.isfinite(final_nll) & jnp.isfinite(final_kl)
    valid &= (final_nll < initial_nll) & (final_kl <= cfg["tr_kl"])
    accepted_before_final_gate = accepted
    best = jax.tree.map(lambda new, old: jnp.where(valid, new, old), best, state)
    best = best.replace(step=state.step+1)
    mu, ls, raw_ls = outputs(best.params)
    final_nll = jnp.where(valid, final_nll, initial_nll)
    final_kl = jnp.where(valid, final_kl, 0.)
    accepted = accepted & valid
    return best, dict(
        loss=final_nll,
        actor_nll_gain=initial_nll-final_nll,
        actor_kl_bound=final_kl,
        actor_accepted=accepted.astype(float),
        final_actor_rejected=(accepted_before_final_gate & ~valid).astype(float),
        actor_momentum_resets=jnp.where(accepted, momentum_reset, 0.),
        actor_step_fraction=jnp.where(accepted, fraction, 0.),
        projection_loss=loss,
        grad_norm=optax.global_norm(grad),
        mean_log_std=ls.mean(),
        raw_scale_clipped_fraction=((raw_ls < -5.) | (raw_ls > -1.)).mean())


def make_smem_repair_update(target, cfg):
    from core import teacher
    validate_config(cfg)

    @jax.jit
    def update(state, z, key):
        key, teacher_key = jax.random.split(key)
        old = jax.tree.map(jax.lax.stop_gradient,
                           state.apply_fn({"params": state.params}, z))
        x, w, sampler_info = teacher(teacher_key, *old, target, cfg)
        x = jax.lax.stop_gradient(x.reshape(-1, old[0].shape[-1]))
        w = jax.lax.stop_gradient(w.reshape(-1)/cfg["populations"])
        fitted, fit_info = fit_teacher(old, x, w, state.step, cfg)
        new, actor_info = distill_actor(state, z, old, fitted, x, w, cfg)
        return new, key, dict(sampler_info, **fit_info, **actor_info)

    return update
