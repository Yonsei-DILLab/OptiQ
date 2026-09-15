"""Best-of-k pilot guidance with exact, defensive proposal-density correction.

Winners define Gaussian proposal centers; they are NOT the final teacher samples.
Fresh draws from the explicit mixture can still target exp(Q/T) using 1/q_mix.
The original proposal has positive mixture mass to preserve tail coverage.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .semi_implicit import ConditionalGaussianProposal, conditional_mixture_log_prob


class BestKGuidedProposal(NamedTuple):
    base: ConditionalGaussianProposal
    guide_means: jax.Array
    guide_log_std: jax.Array
    guide_origins: jax.Array
    guided_fraction: float

    def log_prob(self, u):
        return jnp.logaddexp(
            jnp.log1p(-self.guided_fraction) + self.base.log_prob(u),
            jnp.log(self.guided_fraction)
            + conditional_mixture_log_prob(u, self.guide_means, self.guide_log_std),
        )

    def sample(self, key, repeats, mode):
        if mode != "exact":
            raise ValueError("Best-k guided proposal requires IID exact mixture sampling")
        batch, components, dim = self.base.means.shape
        count = components * repeats
        branch_key, component_key, noise_key = jax.random.split(key, 3)
        guided = jax.random.bernoulli(branch_key, self.guided_fraction, (batch, count))
        index = jax.random.randint(component_key, (batch, count), 0, components)
        mixture_index = index + guided.astype(index.dtype) * components
        means = jnp.concatenate((self.base.means, self.guide_means), axis=1)
        log_std = jnp.concatenate((self.base.effective_log_std(), self.guide_log_std), axis=1)
        selected_mu = jnp.take_along_axis(means, mixture_index[..., None], axis=1)
        selected_ls = jnp.take_along_axis(log_std, mixture_index[..., None], axis=1)
        u = selected_mu + jnp.exp(selected_ls) * jax.random.normal(
            noise_key, (batch, count, dim), dtype=selected_mu.dtype)
        guide_origins = jnp.take_along_axis(self.guide_origins, index, axis=1)
        # Preserve the original component lineage for the existing local diagnostics.
        origins = jnp.where(guided, guide_origins, index)
        return jnp.tanh(u), u, origins


def make_bestk_proposal(base, qf_state, observations, key, k, guided_fraction,
                       source_q_eval, pilot_selection="best"):
    """Condition on independent selected pilots, then define an explicit mixture.

There are M independent groups of k draws from the original floored Gaussian
mixture. Each selected pilot inherits its generating component's Gaussian scale.
Selection is best-Q by default, or the first IID draw for a random-pilot control.
The final teacher sampler uses a different key, AFTER this proposal is fixed.
"""
    batch, components, dim = base.means.shape
    sample_key, dropout_key = jax.random.split(key)
    pilot_a, pilot_u, origins = base.sample(sample_key, k, "exact")
    obs = jnp.broadcast_to(observations[:, None, :],
                           (batch, components * k, observations.shape[-1]))
    twin_q = qf_state.apply_fn(
        {"params": qf_state.params, "batch_stats": qf_state.batch_stats},
        obs.reshape(-1, observations.shape[-1]), pilot_a.reshape(-1, dim),
        rngs={"dropout": dropout_key}, train=False,
    )
    if twin_q.shape != (2, batch * components * k, 1):
        raise ValueError("Best-k proposal guidance requires two scalar critics")
    twin_q = twin_q.reshape(2, batch, components, k)
    if source_q_eval == "mean":
        score = twin_q.mean(axis=0)
    elif source_q_eval == "min":
        score = twin_q.min(axis=0)
    else:
        raise ValueError("source_q_eval must be mean or min")
    if pilot_selection == "best":
        index = score.argmax(axis=-1)
    elif pilot_selection == "first":
        # The first member of each IID pool is an unbiased random pilot.
        # Keep the same pools and Q evaluations for a controlled ablation.
        index = jnp.zeros(score.shape[:-1], dtype=jnp.int32)
    else:
        raise ValueError("pilot_selection must be best or first")
    winner_u = jnp.take_along_axis(pilot_u.reshape(batch, components, k, dim),
                                 index[..., None, None], axis=2).squeeze(2)
    winner_origins = jnp.take_along_axis(origins.reshape(batch, components, k),
                                       index[..., None], axis=2).squeeze(2)
    winner_ls = jnp.take_along_axis(base.effective_log_std(), winner_origins[..., None], axis=1)
    proposal = BestKGuidedProposal(base, winner_u, winner_ls, winner_origins, guided_fraction)
    proposal = jax.tree.map(jax.lax.stop_gradient, proposal)
    metrics = {
        "proposal_best_of_k": jnp.asarray(float(k)),
        "proposal_guided_fraction": jnp.asarray(guided_fraction),
        "proposal_pilot_count": jnp.asarray(float(components * k)),
        "proposal_pilot_winner_q_gain": (score.max(-1) - score.mean(-1)).mean(),
        "proposal_pilot_selected_q_gain": (
            jnp.take_along_axis(score, index[..., None], axis=-1).squeeze(-1)
            - score.mean(-1)).mean(),
        "proposal_pilot_selects_best": jnp.asarray(float(pilot_selection == "best")),
    }
    return proposal, jax.tree.map(jax.lax.stop_gradient, metrics)
