"""Fixed-Q adapter using the same v8 actor core as the RL route."""
import json

import jax
import jax.numpy as jnp
import numpy as np

from optiq_dime import conditional_sac_v8
from .v7 import GMM40V7


class OTConvergenceError(RuntimeError):
    pass


class GMM40V8(GMM40V7):
    def __init__(self, seed=0, *, batch=256, num_students=4096, proposal_components=256,
                 teacher_resample_count=16, actor_samples=16, proposal_std=.05,
                 temperature=1., latent_seed=0, hidden_dims=(256,256),
                 max_iterations=500, min_iterations=10, relative_tolerance=1e-3,
                 actor_max_grad_norm=None):
        super().__init__(seed, batch=batch, num_students=num_students,
            proposal_components=proposal_components, proposals_per_component=1,
            teacher_sampling_mode='stratified', proposal_std=proposal_std,
            temperature=temperature, epsilon=1., iterations=max_iterations,
            actor_samples=actor_samples, teacher_resample_count=teacher_resample_count,
            latent_seed=latent_seed, hidden_dims=hidden_dims, potential_solver='fresh_sinkhorn',
            actor_max_grad_norm=actor_max_grad_norm, source_importance_correction=False)
        self.settings.pop('epsilon')
        self.settings.pop('iterations')
        self.settings.pop('source_importance_correction')
        self.settings.update(max_iterations=max_iterations, min_iterations=min_iterations,
                             relative_tolerance=relative_tolerance, shared_source=True)
        self.settings_signature = json.dumps(dict(version=8, seed=seed, batch=batch,
            actor_learning_rate=3e-4, actor_initial_sigma=.5, actor_log_std_bounds=[-5.,1.],
            hidden_dims=list(hidden_dims), actor_max_grad_norm=actor_max_grad_norm,
            **self.settings), sort_keys=True)
        self.last_metrics = {}
        self.advance_fn = jax.jit(self._advance, static_argnums=2)

    def prepare_for(self, state, key, dual_state=None):
        if dual_state is not None:
            raise ValueError('v8 has no persistent dual state')
        return conditional_sac_v8.prepare_batch(state, self.observations, key, self.q_fn, **self.settings)

    def _advance(self, state, key, count):
        def one(current, current_key):
            updated, loss, next_key, metrics = conditional_sac_v8.update_actor(
                current, self.observations, current_key, self.q_fn, **self.settings)
            delta = jax.tree.map(lambda a,b: a-b, updated.params, current.params)
            metrics['actor_parameter_delta_norm'] = jnp.sqrt(sum(jnp.sum(x*x) for x in jax.tree.leaves(delta)))
            if self.actor_max_grad_norm is not None:
                metrics['actor_gradient_clip_fraction'] = (metrics['actor_gradient_norm']>self.actor_max_grad_norm).astype(jnp.float32)
            return updated, next_key, metrics

        maximum_names = ('ot_row_relative_error', 'ot_col_relative_error',
                         'ot_iterations_max', 'actor_gradient_norm')
        minimum_names = ('actor_update_accepted', 'ot_converged_fraction')
        starting_step = state.step
        # The first attempt supplies the metric tree without a dummy solve.
        # On rejection the core preserves actor, Adam and RNG. Stop here instead
        # of repeating the same expensive failed solve for the rest of a chunk.
        state, key, first = one(state, key)
        maximum = {name: first[name] for name in maximum_names}
        minimum = {name: first[name] for name in minimum_names}

        def condition(carry):
            _, _, attempts, _, _, smallest = carry
            return (attempts < count) & (smallest['actor_update_accepted'] == 1.)

        def advance_one(carry):
            current, current_key, attempts, totals, largest, smallest = carry
            updated, next_key, current_metrics = one(current, current_key)
            totals = jax.tree.map(jnp.add, totals, current_metrics)
            largest = {name: jnp.maximum(largest[name], current_metrics[name])
                       for name in maximum_names}
            smallest = {name: jnp.minimum(smallest[name], current_metrics[name])
                        for name in minimum_names}
            return updated, next_key, attempts + 1, totals, largest, smallest

        state, key, attempts, totals, maximum, minimum = jax.lax.while_loop(
            condition, advance_one,
            (state, key, jnp.asarray(1, dtype=jnp.int32), first, maximum, minimum))
        metrics = jax.tree.map(lambda value: value / attempts, totals)
        metrics.update(maximum)
        metrics.update(minimum)
        metrics['actor_updates_attempted'] = attempts.astype(jnp.float32)
        metrics['actor_updates_accepted'] = jnp.asarray(state.step - starting_step, dtype=jnp.float32)
        return state,key,metrics

    def advance(self, count):
        if count < 1:
            raise ValueError('Positive update count required')
        before = self.updates
        self.state,self.key,metrics = self.advance_fn(self.state,self.key,count)
        jax.block_until_ready((self.state,self.key,metrics))
        self.updates = int(self.state.step)
        self.last_metrics = {k:float(v) for k,v in metrics.items()}
        if self.updates != before+count or self.last_metrics['actor_update_accepted'] != 1.:
            raise OTConvergenceError(f'v8 rejected update after {self.updates} accepted updates: {self.last_metrics}')
        if not all(np.isfinite(v) for v in self.last_metrics.values()):
            raise FloatingPointError(f'Nonfinite v8 diagnostics: {self.last_metrics}')
        return self.last_metrics
