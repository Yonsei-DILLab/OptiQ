"""Reward-only single-target-critic Monte Carlo policy evaluation.

All K actions are independent full (latent + conditional noise) policy draws.
No candidate importance weights, proposal floor, entropy bonus, or critic min.
"""
from functools import partial
import jax
import jax.numpy as jnp
from .policy import OptiQPolicy


@partial(jax.jit, static_argnames=['samples'])
def update(actor, critic, observations, actions, next_observations, rewards,
           dones, gamma, key, samples):
    key, actor_key, _, target_dropout, current_dropout, _ = jax.random.split(key, 6)
    batch, obs_dim = next_observations.shape
    repeated = jnp.broadcast_to(next_observations[:, None, :],
                                (batch, samples, obs_dim)).reshape(batch * samples, obs_dim)
    next_actions = jax.lax.stop_gradient(OptiQPolicy.sample_action(
        actor, repeated, actor_key, deterministic=False))
    values = critic.apply_fn(
        {'params': critic.target_params, 'batch_stats': critic.target_batch_stats},
        repeated, next_actions, rngs={'dropout': target_dropout}, train=False)
    if values.shape != (1, batch * samples, 1):
        raise ValueError('single_mc requires exactly one scalar critic')
    samples_q = values[0, :, 0].reshape(batch, samples)
    expected_q = samples_q.mean(axis=1)
    target = jax.lax.stop_gradient(rewards + gamma * (1.0 - dones) * expected_q)

    def loss_fn(params):
        current, updates = critic.apply_fn(
            {'params': params, 'batch_stats': critic.batch_stats}, observations, actions,
            rngs={'dropout': current_dropout}, mutable=['batch_stats'], train=True)
        if current.shape != (1, batch, 1):
            raise ValueError('single_mc current critic shape mismatch')
        prediction = current[0, :, 0]
        return jnp.square(prediction - target).mean(), (updates, prediction.mean())

    (loss, (updates, current_mean)), grads = jax.value_and_grad(loss_fn, has_aux=True)(critic.params)
    critic = critic.apply_gradients(grads=grads)
    critic = critic.replace(batch_stats=updates.get('batch_stats', critic.batch_stats))
    zero = jnp.asarray(0.0)
    std = samples_q.std(axis=1, ddof=1 if samples > 1 else 0)
    metrics = dict(critic_loss=loss, current_q_values=current_mean,
        next_q_values=target.mean(), entrQ_1=zero, entrQ_2=zero, ent_coef=zero,
        backup_entropy_term=zero, backup_discounted_entropy_term=zero,
        backup_samples=jnp.asarray(samples), critic_count=jnp.asarray(1),
        backup_q_mean=expected_q.mean(), backup_q_sample_std=std.mean(),
        backup_q_mc_se=(std / jnp.sqrt(float(samples))).mean())
    return critic, metrics, key
