"""Persistent state-conditioned semi-dual, one stochastic Adam update per actor.

Restores the historical potential optimizer while keeping v7's conditional
actor objective. Teacher/actor/source sampling use the same PRE-update f.
The dual is updated with the exact analytical potential gradient m-h; no
backward pass through the large transport matrix is needed for the dual.
"""
import math

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from . import conditional_sac


class DualPotentialNetwork(nn.Module):
    num_sources: int
    hidden_dims: tuple = (256, 256)

    @nn.compact
    def __call__(self, observations):
        x = observations
        for width in self.hidden_dims:
            x = nn.relu(nn.Dense(width)(x))
        potential = nn.Dense(self.num_sources, kernel_init=nn.initializers.zeros_init(),
                             bias_init=nn.initializers.zeros_init())(x)
        return potential-potential.mean(axis=-1, keepdims=True)


def create_dual_state(key, observations, *, num_sources=4096,
                      hidden_dims=(256, 256), learning_rate=1e-4):
    if num_sources < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('Positive source count and finite dual learning rate required')
    if observations.ndim != 2 or observations.shape[-1] < 1:
        raise ValueError('Dual observations must have shape [batch, observation_dim]')
    network = DualPotentialNetwork(num_sources, tuple(hidden_dims))
    params = network.init(key, observations[:1])['params']
    return TrainState.create(apply_fn=network.apply, params=params, tx=optax.adam(learning_rate))


def semidual_objective(cost, teacher_weights, potential, epsilon):
    """Mean semi-dual objective; cost[B,H,K], weights[B,K], f[B,H] or f[1,H]."""
    if cost.ndim != 3 or teacher_weights.shape != (cost.shape[0], cost.shape[2]):
        raise ValueError('Expected cost[B,H,K] and weights[B,K]')
    if potential.shape not in ((1, cost.shape[1]), cost.shape[:2]):
        raise ValueError('Potential must have shape [1,H] or [B,H]')
    logits = (potential[:, :, None]-cost)/epsilon-math.log(cost.shape[1])
    log_z = jax.scipy.special.logsumexp(logits, axis=1)
    return epsilon*(teacher_weights*log_z).sum(-1).mean()-potential.mean()


def update_actor_persistent(actor_state, dual_state, observations, key, q_fn, *,
                            dual_observations=None, **settings):
    """Return actor, dual, loss, RNG, metrics after one simultaneous update.

    RL uses each replay observation. A fixed-Q single-state adapter may pass
    one explicit constant observation, sharing one f over its Monte Carlo lanes.
    Source importance h/m is retained without clipping or self-normalization.
    Unlike an iterated Sinkhorn map, a stochastic dual has no h/m <= K bound.
    """
    dual_obs = observations if dual_observations is None else dual_observations
    batch = observations.shape[0]
    if dual_obs.ndim != 2 or dual_obs.shape[0] not in (1, batch):
        raise ValueError('Dual observations need one common state or one state per lane')

    def evaluate_potential(params):
        return dual_state.apply_fn({'params': params}, dual_obs)

    potential, potential_vjp = jax.vjp(evaluate_potential, dual_state.params)
    actor, loss, next_key, metrics, data = conditional_sac.update_actor(
        actor_state, observations, key, q_fn,
        source_potential=jax.lax.stop_gradient(potential), return_batch=True, **settings)
    mass = data['ot']['source_mass']
    num_sources = mass.shape[-1]
    if dual_obs.shape[0] == 1:
        potential_gradient = mass.mean(axis=0, keepdims=True)-1/num_sources
    else:
        potential_gradient = (mass-1/num_sources)/batch
    potential_gradient = jax.lax.stop_gradient(potential_gradient)
    dual_gradient = potential_vjp(potential_gradient)[0]
    updated_dual = dual_state.apply_gradients(grads=dual_gradient)
    metrics.update(
        ot_persistent_dual=jnp.asarray(1.),
        ot_dual_updates=jnp.asarray(1.),
        ot_dual_loss=semidual_objective(data['cost'], data['ot_weights'],
                                      jax.lax.stop_gradient(potential), settings.get('epsilon', .1)),
        ot_dual_potential_rms=jnp.sqrt(jnp.mean(potential**2)),
        ot_dual_potential_gradient_rms=jnp.sqrt(jnp.mean(potential_gradient**2)),
        ot_dual_gradient_norm=jnp.sqrt(sum(jnp.sum(g*g) for g in jax.tree.leaves(dual_gradient))),
        ot_dual_step=jnp.asarray(updated_dual.step, dtype=jnp.float32),
        ot_shared_state_dual=jnp.asarray(float(dual_obs.shape[0] == 1)),
        ot_batch_mean_source_mass_tv=.5*jnp.abs(mass.mean(0)-1/num_sources).sum(),
    )
    return actor, updated_dual, loss, next_key, metrics
