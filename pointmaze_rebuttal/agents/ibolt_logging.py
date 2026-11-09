"""Defer host-only metric reads; keep the frozen iBOLT update untouched."""
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from ..ibolt_core.algorithm import IBOLT


@jax.jit
def unpack_pointmaze_batch(packed):
    # Slices/copies only: one host transfer, then the same float32 input bits.
    return (packed[:, :4], packed[:, 4:6], packed[:, 6:10],
            packed[:, 10], packed[:, 11])


@partial(jax.jit, static_argnames=('n', 'repeats', 'do_actor'))
def sequential_updates(gamma, tau, actor, critic, batches, key,
                       n, repeats, temperature, beta, do_actor):
    def body(carry, packed):
        critic, actor, key = carry
        obs, actions, next_obs, dones, rewards = unpack_pointmaze_batch(packed)
        critic, actor, metrics, key = IBOLT._step(
            gamma, tau, actor, critic, obs, actions, next_obs, dones, rewards,
            key, n, repeats, temperature, beta, do_actor)
        return (critic, actor, key), metrics
    (critic, actor, key), metrics = jax.lax.scan(body, (critic, actor, key), batches)
    return critic, actor, {name: value[-1] for name, value in metrics.items()}, key


class DeferredMetricIBOLT(IBOLT):
    def device_constants(self):
        a = self.cfg.alg.actor
        values = (self.gamma, self.tau, float(a.temperature), float(a.density_correction_beta))
        if getattr(self, '_constant_values', None) != values:
            self._device_constants = tuple(jax.device_put(value) for value in values)
            self._constant_values = values
        return self._device_constants

    def train_collection(self, batch_size, count):
        if self.model_save_path or int(self.cfg.diagnostic_interval) != 0:
            raise ValueError('Sequential execution requires the isolated runner callbacks')
        batches = []
        for _ in range(count):
            data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            obs, actions, next_obs, dones, rewards = [x.numpy() for x in (
                data.observations, data.actions, data.next_observations, data.dones, data.rewards)]
            if obs.shape[1:] != (4,) or actions.shape[1:] != (2,):
                raise ValueError('Packed transport is restricted to PointMaze')
            batches.append(np.concatenate((obs, actions, next_obs, dones, rewards), axis=1))
        a = self.cfg.alg.actor
        gamma, tau, temperature, beta = self.device_constants()
        self.policy.qf_state, self.policy.actor_state, self._pending_metrics, self.key = sequential_updates(
            gamma, tau, self.policy.actor_state, self.policy.qf_state,
            jax.device_put(np.stack(batches)), self.key,
            int(a.num_policy_samples), int(a.proposals_per_policy_sample),
            temperature, beta, self.num_timesteps > a.learning_starts)
        self._n_updates += count
        self.logger.record('train/n_updates', self._n_updates, exclude='tensorboard')

    def train(self, batch_size, gradient_steps):
        # Preserve reference sampling, minibatches, counters, and random-key order.
        data = self.replay_buffer.sample(batch_size * gradient_steps, env=self._vec_normalize_env)
        arrays = [x.numpy() for x in (data.observations, data.actions, data.next_observations,
                                      data.dones, data.rewards)]
        arrays[3], arrays[4] = arrays[3].ravel(), arrays[4].ravel()
        a = self.cfg.alg.actor
        gamma, tau, temperature, beta = self.device_constants()
        for i in range(gradient_steps):
            obs, actions, next_obs, dones, rewards = [x[i*batch_size:(i+1)*batch_size] for x in arrays]
            if obs.shape[1:] != (4,) or actions.shape[1:] != (2,):
                raise ValueError('Packed transport is restricted to the frozen PointMaze spaces')
            packed = np.concatenate((obs, actions, next_obs,
                                     dones[:, None], rewards[:, None]), axis=1)
            obs, actions, next_obs, dones, rewards = unpack_pointmaze_batch(jax.device_put(packed))
            update_actor = self.num_timesteps > a.learning_starts
            self.policy.qf_state, self.policy.actor_state, metrics, self.key = IBOLT._step(
                gamma, tau, self.policy.actor_state, self.policy.qf_state,
                obs, actions, next_obs, dones, rewards, self.key,
                int(a.num_policy_samples), int(a.proposals_per_policy_sample),
                temperature, beta, update_actor)
        self._n_updates += gradient_steps
        if self.model_save_path and (self.num_timesteps % self.save_every_n_steps == 0
                                   or self.num_timesteps == self.learning_starts + 1):
            self._save_model()
        self.logger.record('train/n_updates', self._n_updates, exclude='tensorboard')
        # The reference logger overwrites these keys until dump(). Keep only the
        # same last values, on device, rather than synchronizing every update.
        self._pending_metrics = metrics
        interval = int(self.cfg.diagnostic_interval)
        if interval > 0 and self.num_timesteps % interval == 0:
            self.flush_training_metrics()
            self.logger.dump(self.num_timesteps)

    def flush_training_metrics(self):
        metrics = getattr(self, '_pending_metrics', None)
        if metrics is None:
            return
        names = tuple(metrics)
        values = jax.device_get(jnp.stack([metrics[name] for name in names]))
        for name, value in zip(names, values):
            self.logger.record('train/' + name, float(value))
        self._pending_metrics = None
