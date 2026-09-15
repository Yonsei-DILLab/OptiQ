"""Collection-only best-of-k sampling from the current Gaussian policy."""
from functools import partial
import math

import jax
import jax.numpy as jnp

from .policy import OptiQPolicy


def parse_behavior_best_of_k(alg):
    value = alg.get("behavior_best_of_k", 1)
    number = float(value)
    if isinstance(value, bool) or not math.isfinite(number) or number < 1 or int(number) != number:
        raise ValueError("behavior_best_of_k must be a positive integer")
    if number > 1:
        if alg.actor.get("type") != "semi_implicit" or alg.actor.get("latent_prior", "normal") != "normal":
            raise ValueError("best-of-k collection requires a continuous Gaussian latent actor")
        if alg.critic.n_atoms != 1 or alg.critic.n_critics != 2:
            raise ValueError("best-of-k collection requires scalar twin critics")
    return int(number)


@partial(jax.jit, static_argnames=["k"])
def select_best_of_k(actor_state, critic_state, observations, first_action, key, k):
    """Keep the baseline draw as candidate 0; add k-1 independent policy draws.

    Critic inference uses live parameters, frozen statistics, and no gradients.
    Both latent z and conditional Gaussian epsilon vary across added candidates.
    This helper is called exclusively by the environment collector.
    """
    sample_key, critic_key = jax.random.split(key)
    batch, action_dim = first_action.shape
    extra = OptiQPolicy.sample_action(
        actor_state, jnp.repeat(observations, k - 1, axis=0), sample_key,
        deterministic=False, sample_conditional_noise=True,
    ).reshape(batch, k - 1, action_dim)
    candidates = jnp.concatenate((first_action[:, None, :], extra), axis=1)
    predictions = critic_state.apply_fn(
        {"params": critic_state.params, "batch_stats": critic_state.batch_stats},
        jnp.repeat(observations, k, axis=0), candidates.reshape(batch * k, action_dim),
        rngs={"dropout": critic_key}, train=False,
    )
    scores = predictions[..., 0].reshape(2, batch, k).min(axis=0)
    indices = jnp.argmax(scores, axis=1)
    chosen = candidates[jnp.arange(batch), indices]
    selected_q = scores[jnp.arange(batch), indices]
    metrics = {
        "best_of_k_selected_q": selected_q.mean(),
        "best_of_k_base_q": scores[:, 0].mean(),
        "best_of_k_q_gain": (selected_q - scores[:, 0]).mean(),
        "best_of_k_kept_base_fraction": (indices == 0).mean(),
    }
    return chosen, metrics
