"""Keep the legacy OT update; route continuation and evaluation to its EMA."""
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from optiq_dime.algorithm import OptiQDIME
from optiq_dime.policy import OptiQPolicy
from optiq_dime.evaluation import MujocoEvalCallback


class Paused(Exception):
    pass


class ExplorerPolicy(OptiQPolicy):
    evaluation_only = False

    def build(self, *args, **kwargs):
        key = super().build(*args, **kwargs)
        # Evaluation never consumes exploration's random stream.
        self.eval_key = jax.random.PRNGKey(910000 + int(self.cfg.seed))
        return key

    def _predict(self, observation, deterministic=False):
        if not self.evaluation_only:
            return super()._predict(observation, deterministic=deterministic)
        self.eval_key, key = jax.random.split(self.eval_key)
        return self.sample_action(self.target_actor_state, observation, key,
                                  deterministic=deterministic)[0]


class EvaluatorCallback(MujocoEvalCallback):
    def _on_step(self):
        policy = self.model.policy
        assert not policy.evaluation_only
        due = self.n_calls == 1 or self.n_calls % self.eval_freq == 0
        if due:
            policy.evaluation_only = True
            self.logger.record('eval/uses_ema_evaluator', 1)
        try:
            return super()._on_step()
        finally:
            policy.evaluation_only = False


@jax.jit
def continuation_target(evaluator, critic, next_observations, rewards, dones, gamma, key):
    """Exactly r + gamma (1-d) min_k Q_target(s', G_eval(s',z'))."""
    next_actions = jax.lax.stop_gradient(OptiQPolicy.sample_action(
        evaluator, next_observations, key, deterministic=False))
    values = critic.apply_fn({'params':critic.target_params,
                              'batch_stats':critic.target_batch_stats},
                             next_observations, next_actions, train=False)
    target = rewards + gamma*(1-dones)*values[..., 0].min(axis=0)
    return jax.lax.stop_gradient(target), next_actions


class ExplorerOptiQ(OptiQDIME):
    policy_aliases = {'MlpPolicy':ExplorerPolicy, 'MultiInputPolicy':ExplorerPolicy}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        a = self.cfg.alg
        assert a.critic.n_critics == 2 and a.critic.n_atoms == 1
        assert not a.critic.crossq_style and not a.optimizer.bn
        assert a.critic.dropout_rate is None and a.critic.entr_coeff == 0
        assert a.utd == a.policy_delay == 1
        assert a.actor.source_q_eval == 'mean' and a.actor.transport_target_mode == 'argmax'
        assert a.actor.td_noise_std == a.actor.td_noise_clip == 0
        assert 0 < a.policy_tau < 1
        self.stop_requested = False
        self.checkpoint_hook = None

    # The inherited _train calls cls.update_critic, the unchanged legacy
    # update_actor, then inherited soft_update_target_actor after every update.
    @staticmethod
    @partial(jax.jit, static_argnames=['crossq_style', 'use_bnstats_from_live_net',
                                     'num_atoms', 'v_min', 'v_max', 'entr_coeff'])
    def update_critic(crossq_style, use_bnstats_from_live_net, gamma, target_actor_state,
                      qf_state, observations, actions, next_observations, rewards, dones,
                      num_atoms, z_atoms, v_min, v_max, entr_coeff, td_noise_std,
                      td_noise_clip, key):
        assert not crossq_style and not use_bnstats_from_live_net and num_atoms == 1
        assert entr_coeff == 0
        # Preserve the legacy random-key split structure; omit TD perturbation.
        key, actor_key, _, _, dropout_key, _ = jax.random.split(key, 6)
        target, next_actions = continuation_target(target_actor_state, qf_state,
                                                   next_observations, rewards, dones, gamma, actor_key)

        def loss(params):
            q, state_updates = qf_state.apply_fn(
                {'params':params, 'batch_stats':qf_state.batch_stats}, observations,
                actions, train=True, mutable=['batch_stats'], rngs={'dropout':dropout_key})
            current = q[..., 0]
            value = jnp.square(current-target[None, :]).mean(axis=1).sum()
            return value, (state_updates, current.min(axis=0).mean())

        (value, (updates, current)), grads = jax.value_and_grad(loss, has_aux=True)(qf_state.params)
        qf_state = qf_state.apply_gradients(grads=grads).replace(batch_stats=updates.get('batch_stats', {}))
        zero = jnp.asarray(0., dtype=value.dtype)
        return qf_state, dict(critic_loss=value, current_q_values=current,
                             next_q_values=target.mean(), entrQ_1=zero, entrQ_2=zero,
                             td_evaluator_action_abs=jnp.abs(next_actions).mean()), key

    def train(self, batch_size, gradient_steps):
        assert not self.policy.evaluation_only
        super().train(batch_size, gradient_steps)
        if self.num_timesteps % 1000 == 0:
            exp = jax.tree_util.tree_leaves(self.policy.actor_state.params)
            ev = jax.tree_util.tree_leaves(self.policy.target_actor_state.params)
            mse = sum(jnp.square(a-b).sum() for a,b in zip(exp,ev))/sum(a.size for a in exp)
            self.logger.record('policy/explorer_evaluator_parameter_rms', float(jnp.sqrt(mse)))
            self.logger.record('policy/evaluator_gradient_steps', int(self.policy.target_actor_state.step))
            if self._last_obs is not None:
                obs = jnp.asarray(self._last_obs)
                key = jax.random.PRNGKey(930000 + int(self.cfg.seed))
                ea = OptiQPolicy.sample_action(self.policy.actor_state, obs, key)
                va = OptiQPolicy.sample_action(self.policy.target_actor_state, obs, key)
                self.logger.record('policy/matched_latent_action_rms', float(jnp.sqrt(jnp.mean((ea-va)**2))))
        if self.checkpoint_hook is not None:
            self.checkpoint_hook(self)
        if self.stop_requested:
            raise Paused()
