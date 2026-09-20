"""Thin fixed-Q adapter: all actor learning executes the shared RL v7 core."""
import math
import json
from pathlib import Path

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

from optiq_dime import conditional_sac
from optiq_dime.optimizers import adam_with_grad_clip
from optiq_dime.persistent_transport import create_dual_state, update_actor_persistent
from optiq_dime.policy import SemiImplicitActor
from .target import Target


class GMM40V7:
    def __init__(self, seed=0, *, batch=256, num_students=4096,
                 proposal_components=256, proposals_per_component=1,
                 teacher_sampling_mode='stratified', proposal_std=.05,
                 temperature=1., epsilon=.1, iterations=100,
                 actor_samples=16, teacher_resample_count=16,
                 latent_seed=0, hidden_dims=(256, 256),
                 potential_solver='persistent_dual', dual_learning_rate=1e-4,
                 dual_hidden_dims=(256, 256), actor_max_grad_norm=None,
                 source_importance_correction=True):
        if potential_solver not in ('persistent_dual', 'fresh_sinkhorn'):
            raise ValueError('Unknown potential solver')
        if not isinstance(source_importance_correction, (bool, np.bool_)):
            raise ValueError('source_importance_correction must be a boolean')
        self.source_importance_correction = bool(source_importance_correction)
        if actor_max_grad_norm is not None:
            actor_max_grad_norm = float(actor_max_grad_norm)
            if not math.isfinite(actor_max_grad_norm) or actor_max_grad_norm <= 0:
                raise ValueError('actor_max_grad_norm must be null or positive and finite')
        self.actor_max_grad_norm = actor_max_grad_norm
        self.potential_solver = potential_solver
        self.target = Target()
        self.batch = int(batch)
        self.observations = jnp.zeros((self.batch, 1), dtype=jnp.float32)
        self.settings = dict(num_students=num_students, proposal_components=proposal_components,
                             proposals_per_component=proposals_per_component,
                             teacher_sampling_mode=teacher_sampling_mode, proposal_std=proposal_std,
                             temperature=temperature, epsilon=epsilon, iterations=iterations,
                             actor_samples=actor_samples, teacher_resample_count=teacher_resample_count,
                             student_latents=None, latent_seed=latent_seed, action_scale=40.)
        # The opt-in ablation disables only the actor source correction. Omit
        # the default flag so historical no-clip and clipped signatures survive.
        if not self.source_importance_correction:
            self.settings['source_importance_correction'] = False
        signature = dict(seed=seed, batch=batch,
            hidden_dims=list(hidden_dims), actor_learning_rate=3e-4,
            actor_log_std_bounds=[-5., 1.], actor_initial_sigma=.5,
            **self.settings)
        if potential_solver == 'persistent_dual':
            signature.update(potential_solver=potential_solver,
                             dual_learning_rate=dual_learning_rate,
                             dual_hidden_dims=list(dual_hidden_dims),
                             dual_observation='ones[1,1]', dual_updates_per_actor=1)
        # Preserve old no-clip signatures and checkpoints exactly. Clipped runs
        # must never silently resume as a different optimizer experiment.
        if actor_max_grad_norm is not None:
            signature['actor_max_grad_norm'] = actor_max_grad_norm
        # The explicit fresh control retains the exact historical signature.
        self.settings_signature = json.dumps(signature, sort_keys=True)
        self.actor = SemiImplicitActor(2, tuple(hidden_dims), -5., 1., math.log(.5),
                                      mean_output_init_scale=1e-4)
        self.key, init = jax.random.split(jax.random.PRNGKey(seed))
        params = self.actor.init(init, jnp.zeros((1, 1)), jnp.zeros((1, 2)))['params']
        self.state = TrainState.create(
            apply_fn=self.actor.apply, params=params,
            tx=adam_with_grad_clip(3e-4, .9, .999, actor_max_grad_norm))
        self.dual_observations = jnp.ones((1, 1), dtype=jnp.float32)
        self.dual_state = None
        if potential_solver == 'persistent_dual':
            self.dual_state = create_dual_state(
                jax.random.fold_in(jax.random.PRNGKey(seed), 0x4455414C),
                self.dual_observations, num_sources=num_students,
                hidden_dims=tuple(dual_hidden_dims), learning_rate=dual_learning_rate)
        self.updates = 0
        self.advance_fn = jax.jit(self._advance, static_argnums=2)
        self.advance_persistent_fn = jax.jit(self._advance_persistent, static_argnums=3)
        self.sample_fn = jax.jit(self._sample, static_argnums=2)

    def q_fn(self, observations, actions):
        del observations
        return self.target.jax_log_prob(40.*actions)

    def prepare(self):
        """Expose the exact shared teacher/assignment preparation for diagnostics."""
        return self.prepare_for(self.state, self.key, self.dual_state)

    def prepare_for(self, state, key, dual_state=None):
        potential = None
        if self.potential_solver == 'persistent_dual':
            potential = dual_state.apply_fn({'params': dual_state.params}, self.dual_observations)
        return conditional_sac.prepare_batch(state, self.observations, key,
                    self.q_fn, source_potential=potential, **self.settings)

    def _advance(self, state, key, count):
        def one(carry, _):
            current, current_key = carry
            updated, loss, next_key, metrics = conditional_sac.update_actor(
                current, self.observations, current_key, self.q_fn, **self.settings)
            metrics = self._optimizer_metrics(metrics, current, updated)
            return (updated, next_key), dict(metrics, loss=loss)
        (state, key), metrics = jax.lax.scan(one, (state, key), None, length=count)
        return state, key, self._aggregate_metrics(metrics)

    def _advance_persistent(self, state, dual_state, key, count):
        def one(carry, _):
            current, dual, current_key = carry
            updated, updated_dual, loss, next_key, metrics = update_actor_persistent(
                current, dual, self.observations, current_key, self.q_fn,
                dual_observations=self.dual_observations, **self.settings)
            metrics = self._optimizer_metrics(metrics, current, updated)
            return (updated, updated_dual, next_key), dict(metrics, loss=loss)
        (state, dual_state, key), metrics = jax.lax.scan(
            one, (state, dual_state, key), None, length=count)
        return state, dual_state, key, self._aggregate_metrics(metrics)

    def _aggregate_metrics(self, metrics):
        averaged = jax.tree_util.tree_map(jnp.mean, metrics)
        if self.actor_max_grad_norm is not None or not self.source_importance_correction:
            averaged.update(
                actor_source_importance_block_max=jnp.max(metrics['actor_source_importance_max']),
                actor_gradient_block_max=jnp.max(metrics['actor_gradient_norm']),
                actor_parameter_step_block_max=jnp.max(metrics['actor_parameter_delta_norm']))
        return averaged

    def _optimizer_metrics(self, metrics, previous, updated):
        """Describe per-update clipping before scan averaging, without loss changes."""
        if self.actor_max_grad_norm is None and self.source_importance_correction:
            return metrics
        raw_norm = metrics['actor_gradient_norm']
        delta = jax.tree_util.tree_map(lambda new, old: new - old,
                                      updated.params, previous.params)
        delta_norm = jnp.sqrt(sum(jnp.sum(value * value) for value in jax.tree.leaves(delta)))
        if self.actor_max_grad_norm is None:
            return dict(metrics, actor_parameter_delta_norm=delta_norm)
        limit = jnp.asarray(self.actor_max_grad_norm, dtype=raw_norm.dtype)
        scale = jnp.minimum(1., limit / jnp.maximum(raw_norm, jnp.finfo(raw_norm.dtype).tiny))
        return dict(metrics,
                    actor_gradient_norm_clipped=jnp.minimum(raw_norm, limit),
                    actor_gradient_clip_scale=scale,
                    actor_gradient_clip_fraction=(raw_norm > limit).astype(raw_norm.dtype),
                    actor_gradient_clip_limit=limit,
                    actor_parameter_delta_norm=delta_norm)

    def advance(self, count):
        if count < 1:
            raise ValueError('Positive update count required')
        if self.potential_solver == 'persistent_dual':
            self.state, self.dual_state, self.key, metrics = self.advance_persistent_fn(
                self.state, self.dual_state, self.key, count)
        else:
            self.state, self.key, metrics = self.advance_fn(self.state, self.key, count)
        jax.block_until_ready((self.state, self.dual_state, self.key, metrics))
        self.updates += count
        return {key: float(value) for key, value in metrics.items()}

    def _sample(self, params, key, n):
        zkey, ekey = jax.random.split(key)
        z = jax.random.normal(zkey, (n, 2))
        noise = jax.random.normal(ekey, (n, 2))
        mu, ls = self.actor.apply({'params': params}, jnp.zeros((n, 1)), z)
        return 40*jnp.tanh(mu+jnp.exp(ls)*noise), 40*jnp.tanh(mu), jnp.exp(ls)

    def evaluate_samples(self, n, seed):
        x, means, _ = self.sample_fn(self.state.params, jax.random.PRNGKey(seed), n)
        return np.asarray(x), {'mu_only': np.asarray(means)}

    def checkpoint(self):
        saved = dict(state=self.state, key=self.key, updates=self.updates,
                     settings_signature=self.settings_signature)
        if self.dual_state is not None:
            saved['dual_state'] = self.dual_state
        return saved

    def save(self, path):
        Path(path).write_bytes(flax.serialization.to_bytes(self.checkpoint()))

    def restore(self, path):
        payload = Path(path).read_bytes()
        raw_saved = flax.serialization.msgpack_restore(payload)
        has_dual = 'dual_state' in raw_saved
        if has_dual != (self.potential_solver == 'persistent_dual'):
            raise ValueError('Checkpoint potential solver differs from the requested solver')
        # Check before decoding optimizer state: plain Adam and clip+Adam use
        # different state trees, and the mismatch should identify the setting.
        if raw_saved['settings_signature'] != self.settings_signature:
            raise ValueError('Checkpoint settings/seed mismatch, including fixed latent bank')
        saved = flax.serialization.from_bytes(self.checkpoint(), payload)
        if saved['settings_signature'] != self.settings_signature:
            raise ValueError('Checkpoint settings/seed mismatch, including fixed latent bank')
        self.state, self.key, self.updates = saved['state'], jnp.asarray(saved['key']), int(saved['updates'])
        if has_dual:
            self.dual_state = saved['dual_state']
