"""Shared-actor generalized M-step with a backtracked KL-constrained update."""
import jax
import jax.numpy as jnp
import optax

from .smem_tr import batch_nll, box_kl, log_components, responsibilities


def distill_actor(actor_state, observations, latents, old, fitted, actions, weights, cfg):
    batch, components, dims = old[0].shape
    obs = jnp.broadcast_to(observations[:, None], (batch, components, observations.shape[-1])).reshape(-1, observations.shape[-1])
    z = latents.reshape(-1, dims)
    initial_nll = batch_nll(old, actions, weights).mean()
    posterior = jax.lax.stop_gradient(jax.vmap(responsibilities, in_axes=((0, 0), 0))(fitted, actions))
    weighted_posterior = weights[:, None, :] * posterior

    def outputs(params):
        return tuple(x.reshape(batch, components, dims) for x in actor_state.apply_fn({'params': params}, obs, z, return_raw=True))

    def objective(params):
        mu, scale, raw_scale = outputs(params)
        if cfg['projection_objective'] == 'parameter_regression':
            return (.5*((mu-fitted[0])*jnp.exp(-old[1]))**2 + (raw_scale-fitted[1])**2).mean()
        # Generalized EM auxiliary objective: responsibilities come from the
        # projected SMEM teacher. Retain component responsibility mass instead
        # of giving each independently fitted component equal regression weight.
        logp = jax.vmap(log_components, in_axes=((0, 0), 0))((mu, scale), actions)
        return -(weighted_posterior * logp).sum((1, 2)).mean()

    def evaluate(params):
        mu, scale, _ = outputs(params)
        return batch_nll((mu, scale), actions, weights).mean(), box_kl(mu, scale, *old).mean()

    def inner(_, carry):
        current, current_nll, accepted_count, reset_count, frac_sum, grad_norm = carry
        loss, grad = jax.value_and_grad(objective)(current.params)

        def search(start):
            proposed = start.apply_gradients(grads=grad)
            finite_state = jnp.all(jnp.stack([
                jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves((proposed.params, proposed.opt_state))
            ]))
            delta = jax.tree.map(lambda new, prev: new-prev, proposed.params, current.params)

            def body(i, best):
                params, nll, taken, fraction = best

                def attempt(_):
                    rate = .5**i
                    candidate = jax.tree.map(lambda p, d: p+rate*d, current.params, delta)
                    candidate_nll, kl = evaluate(candidate)
                    ok = finite_state & jnp.isfinite(candidate_nll) & jnp.isfinite(kl)
                    ok &= (candidate_nll < current_nll) & (kl <= cfg['tr_kl'])
                    return (
                        jax.tree.map(lambda n, p: jnp.where(ok, n, p), candidate, params),
                        jnp.where(ok, candidate_nll, nll), ok,
                        jnp.where(ok, rate, fraction),
                    )
                return jax.lax.cond(taken, lambda _: best, attempt, None)

            params, nll, taken, fraction = jax.lax.fori_loop(
                0, cfg['actor_backtracks'], body,
                (current.params, current_nll, jnp.array(False), jnp.array(0.)),
            )
            selected = jax.tree.map(lambda new, previous: jnp.where(taken, new, previous), proposed, current)
            return selected.replace(params=params), nll, taken, fraction

        proposed, nll, accepted, fraction = search(current)

        def restart(_):
            # On rejection, stale momentum is discarded only for the retry.
            # A rejected retry restores both original params and optimizer state.
            restarted = current.replace(opt_state=current.tx.init(current.params))
            return search(restarted)

        retried, retry_nll, retry_ok, retry_fraction = jax.lax.cond(
            accepted, lambda _: (proposed, nll, accepted, fraction), restart, None,
        )
        reset_used = (~accepted & retry_ok).astype(float)
        return retried, retry_nll, accepted_count+retry_ok, reset_count+reset_used, frac_sum+retry_fraction, optax.global_norm(grad)

    best, final_nll, accepts, resets, fractions, grad_norm = jax.lax.fori_loop(
        0, cfg['projection_steps'], inner,
        (actor_state, initial_nll, jnp.array(0.), jnp.array(0.), jnp.array(0.), jnp.array(0.)),
    )
    best = best.replace(step=actor_state.step+1)
    mu, scale, raw_scale = outputs(best.params)
    per_state_kl = box_kl(mu, scale, *old).mean(1)
    return best, dict(
        actor_loss=final_nll, actor_nll_gain=initial_nll-final_nll,
        actor_kl_bound=per_state_kl.mean(), actor_kl_state_max=per_state_kl.max(),
        actor_accepted=(accepts > 0).astype(float), final_actor_rejected=jnp.array(0.),
        actor_inner_accepts=accepts, actor_momentum_resets=resets,
        actor_step_fraction=fractions/jnp.maximum(accepts, 1.),
        projection_loss=objective(best.params), actor_grad_norm=grad_norm,
        actor_log_std_mean=scale.mean(), actor_log_std_min=scale.min(), actor_log_std_max=scale.max(),
        actor_std_mean=jnp.exp(scale).mean(), actor_std_min=jnp.exp(scale).min(), actor_std_max=jnp.exp(scale).max(),
        raw_scale_clipped_fraction=((raw_scale < -5.) | (raw_scale > -1.)).mean(),
    )
