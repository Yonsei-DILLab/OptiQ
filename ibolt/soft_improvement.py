"""Black-box soft-value checks for OT-proposed policy updates.

Monte Carlo outputs are estimates, not certified statewise bounds. The separate
margin function requires externally justified critic/sampling error bounds.
No derivatives through Q or an entropy objective are used to propose updates.
"""

import jax
import jax.numpy as jnp

from .semi_implicit import actor_components, conditional_mixture_log_prob, finite_policy_action_and_log_density
from .latent import FiniteMixtureTrainState
from .critic_utils import critic_expectation


def entropy_bracket_sample(actor_state, observations, key, components):
    """One true-policy action, with lower/upper entropy samples in expectation.

    The lower estimate uses the generating component + M-1 auxiliaries. The
    upper estimate uses M auxiliaries independent of the action. Their bounds
    hold after expectation over actions and components, not pointwise.
    """
    if isinstance(actor_state, FiniteMixtureTrainState):
        action, u, log_pi = finite_policy_action_and_log_density(actor_state, observations, key, components)
        return action, u, -log_pi, -log_pi
    latent_key, noise_key = jax.random.split(key)
    mu, ls = actor_components(actor_state, observations, latent_key, components+1)
    u = mu[:,0] + jnp.exp(ls[:,0])*jax.random.normal(noise_key,mu[:,0].shape)
    lower = -conditional_mixture_log_prob(u[:,None],mu[:,:components],ls[:,:components])[:,0]
    upper = -conditional_mixture_log_prob(u[:,None],mu[:,1:],ls[:,1:])[:,0]
    return jnp.tanh(u), u, lower, upper


def soft_value_bracket_samples(actor_state, critic_state, observations, key,
                               temperature, z_atoms, components=16, draws=16):
    keys = jax.random.split(key,draws)
    actions, _, lower, upper = jax.vmap(
        lambda k: entropy_bracket_sample(actor_state,observations,k,components)
    )(keys)
    n_draws,batch,action_dim = actions.shape
    obs = jnp.broadcast_to(observations[None],(n_draws,*observations.shape))
    q_dist = critic_state.apply_fn(
        {'params':critic_state.params,'batch_stats':critic_state.batch_stats},
        obs.reshape(n_draws*batch,-1),actions.reshape(n_draws*batch,action_dim),
        rngs={'dropout':key},train=False,
    )
    q = critic_expectation(q_dist,z_atoms).min(axis=0).reshape(n_draws,batch)
    return jax.lax.stop_gradient(q+temperature*lower), jax.lax.stop_gradient(q+temperature*upper)


def candidate_gap_samples(old_actor, candidate_actor, critic_state, observations,
                          key, temperature, z_atoms, components=16, draws=16):
    """Paired candidate-lower minus old-upper soft scores, [draws, states].

    Common random numbers reduce variance; both sides use the same frozen Q.
    Means/standard errors of these samples do not bound critic error or certify
    unobserved states. A positive sample mean alone is not a theorem.
    """
    _, old_upper = soft_value_bracket_samples(old_actor,critic_state,observations,
        key,temperature,z_atoms,components,draws)
    new_lower, _ = soft_value_bracket_samples(candidate_actor,critic_state,observations,
        key,temperature,z_atoms,components,draws)
    return jax.lax.stop_gradient(new_lower-old_upper)


def sampled_soft_update(old_actor, candidate_actor, critic_state, observations,
                        key, temperature, z_atoms, components=16, draws=8,
                        standard_error_multiplier=2.0):
    """Accept an OT candidate using a held-out replay *estimate* of soft gain.

    This is an empirical filter, not a certified policy-improvement test. The
    standard error treats each replay state as a cluster, since its multiple
    action draws share that state. Neither this error estimate nor the entropy
    bracket supplies a uniform learned-critic error or global state coverage.
    Rejection restores the entire old TrainState, including Adam and step.
    """
    samples = candidate_gap_samples(
        old_actor, candidate_actor, critic_state, observations, key,
        temperature, z_atoms, components, draws,
    )
    state_gaps = samples.mean(axis=0)
    mean = state_gaps.mean()
    standard_error = state_gaps.std(ddof=1) / jnp.sqrt(state_gaps.size)
    margin = mean - standard_error_multiplier * standard_error
    accepted = jnp.all(jnp.isfinite(samples)) & jnp.isfinite(margin) & (margin > 0.0)
    selected = jax.lax.cond(accepted, lambda: candidate_actor, lambda: old_actor)
    metrics = {
        "soft_guard_accepted": accepted.astype(jnp.float32),
        "soft_guard_gap_mean": mean,
        "soft_guard_gap_standard_error": standard_error,
        "soft_guard_estimated_margin": margin,
        "soft_guard_min_state_gap": state_gaps.min(),
        "soft_guard_negative_state_fraction": (state_gaps < 0.0).mean(),
    }
    return selected, metrics


def conservative_soft_margin(q_gain, entropy_lower_new, entropy_upper_old,
                             temperature, *, critic_error_bound, sampling_error_bound):
    """Per-state sufficient margin under externally verified error bounds.

    If every relevant state's margin is >=0, the true old-policy Q error is
    uniformly <= critic_error_bound, and all supplied bounds are valid, the
    soft-policy-improvement sufficient condition holds. The API deliberately
    supplies no default zero error bounds for a learned critic.
    """
    if temperature <= 0 or critic_error_bound < 0 or sampling_error_bound < 0:
        raise ValueError('Positive temperature and nonnegative error bounds required')
    return (q_gain + temperature*(entropy_lower_new-entropy_upper_old)
            -2*critic_error_bound-sampling_error_bound)
