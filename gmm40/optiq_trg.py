"""Fixed-Q adapter reusing the current Direct GMM/TRG actor, proposal and NLL."""
import importlib
import math

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from .optiq import OptiQ

SOURCE = 'analysis_tools.experiments.20260920_truncated_mll.optiq_dime'
Actor = importlib.import_module(SOURCE + '.policy').SemiImplicitActor
Proposal = importlib.import_module(SOURCE + '.semi_implicit').ConditionalGaussianProposal
direct_gmm_nll = importlib.import_module(SOURCE + '.distillation').direct_gmm_nll
sample_box = importlib.import_module(SOURCE + '.box_gaussian').sample_box


class OptiQTRG(OptiQ):
    def __init__(self, target, seed=0, n=64, m=64, batch=256,
                 hidden_dims=(256,256), temperature=1.,
                 log_std_max=-1., initial_log_std=-1.):
        if n <= 0 or m <= 0 or m % n or batch <= 0:
            raise ValueError('TRG requires positive batch/N/M and M divisible by N')
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Temperature must be positive and finite')
        if not (math.isfinite(log_std_max) and math.isfinite(initial_log_std)
                and -5. < log_std_max and -5. <= initial_log_std <= log_std_max):
            raise ValueError('TRG requires -5 < log_std_max and initial log std within bounds')
        self.target, self.n, self.m, self.batch = target, n, m, batch
        self.temperature = temperature
        # GMM40-only mean initialization override; retain the RL sigma defaults.
        self.actor = Actor(2, tuple(hidden_dims), -5., log_std_max, initial_log_std,
                           mean_output_init_scale=1.)
        self.key, init = jax.random.split(jax.random.PRNGKey(seed))
        params = self.actor.init(init, jnp.zeros((1,1)), jnp.zeros((1,2)))['params']
        self.state = TrainState.create(apply_fn=self.actor.apply, params=params, tx=optax.adam(3e-4))
        self.updates = 0
        self.advance_fn = jax.jit(self._advance, static_argnums=2)
        self.sample_fn = jax.jit(self._sample, static_argnums=2)

    def _update(self, carry, _):
        state, key = carry
        key, zk, pk = jax.random.split(key, 3)
        obs = jnp.zeros((self.batch*self.n,1))
        z = jax.random.normal(zk, (self.batch*self.n,2))
        def loss(params):
            mu, ls = state.apply_fn({'params':params}, obs, z)
            mu, ls = mu.reshape(self.batch,self.n,2), ls.reshape(self.batch,self.n,2)
            proposal = Proposal(jax.lax.stop_gradient(mu), jax.lax.stop_gradient(ls), math.exp(-5))
            actions, _, _ = proposal.sample(pk, self.m//self.n, 'exact')
            actions = jax.lax.stop_gradient(actions)
            # Physical-coordinate density differs by a constant, canceled by softmax.
            logq = proposal.log_prob(actions)
            q = self.target.jax_log_prob(40*actions)
            w = jax.lax.stop_gradient(jax.nn.softmax(q/self.temperature-logq, axis=-1))
            value, ell = direct_gmm_nll(mu, ls, actions, w)
            usage = (jax.nn.softmax(ell, axis=1)*w[:,None,:]).sum(-1)
            return value, dict(loss=value,teacher_ess=(1/(w*w).sum(-1)).mean(),
                sigma_mean=jnp.exp(ls).mean(),sigma_min=jnp.exp(ls).min(),sigma_max=jnp.exp(ls).max(),
                teacher_Q=q.mean(),weighted_teacher_Q=(w*q).sum(-1).mean(),
                mean_spread=jnp.var(mu,axis=1).mean(),
                component_usage_ess=(1/(usage*usage).sum(-1)).mean())
        (_,info), grads = jax.value_and_grad(loss,has_aux=True)(state.params)
        return (state.apply_gradients(grads=grads),key),info

    def _sample(self, params, key, n):
        zk, ak = jax.random.split(key)
        z = jax.random.normal(zk,(n,2))
        mu, ls = self.actor.apply({'params':params},jnp.zeros((n,1)),z)
        return 40*sample_box(ak,mu,ls),40*mu,40*jnp.tanh(z),jnp.exp(ls)
