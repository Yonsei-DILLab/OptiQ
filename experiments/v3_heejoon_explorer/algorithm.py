"""Legacy explorer; min-max gated evaluator with stopped positive-advantage weights."""
from functools import partial
import jax
import jax.numpy as jnp
from optiq_dime.algorithm import OptiQDIME
from optiq_dime.policy import OptiQPolicy
from optiq_dime.evaluation import MujocoEvalCallback
from experiments.v1_heejoon_explorer.algorithm import (
    ExplorerOptiQ as EMAExplorerOptiQ, ExplorerPolicy, Paused, continuation_target)


Q_SAMPLES = 16


def latent_actions(state, observations, z):
    return jnp.clip(state.apply_fn({'params':state.params},observations,z),-1.,1.)


def target_min(critic, observations, actions):
    q=critic.apply_fn({'params':critic.target_params,'batch_stats':critic.target_batch_stats},
                      observations,actions,train=False)
    return q[...,0].min(axis=0)


@partial(jax.jit,static_argnames=['q_samples'])
def evaluator_teacher(evaluator,explorer,critic,observations,key,q_samples=Q_SAMPLES):
    """One explorer action/state; independent K-action evaluator MC baseline.

    Explorer uses target twin-min; evaluator baseline uses E[target twin-max].
    Detach the entire teacher, including the
    evaluator baseline; no actor/critic gradient passes through the acceptance gate.
    """
    b=observations.shape[0]
    d=explorer.params[f'Dense_{len(explorer.params)-1}']['bias'].shape[0]
    zk,bk=jax.random.split(key)
    z=jax.random.normal(zk,(b,d),dtype=observations.dtype)
    candidate=latent_actions(explorer,observations,z)
    baseline_z=jax.random.normal(bk,(b,q_samples,d),dtype=observations.dtype)
    baseline_obs=jnp.repeat(observations,q_samples,axis=0)
    baseline_actions=latent_actions(evaluator,baseline_obs,baseline_z.reshape(b*q_samples,d))
    q_exp=target_min(critic,observations,candidate)
    all_q=critic.apply_fn({'params':critic.target_params,'batch_stats':critic.target_batch_stats},
                          baseline_obs,baseline_actions,train=False)[...,0]
    q_eval=all_q.max(axis=0).reshape(b,q_samples)
    q_eval_min=all_q.min(axis=0).reshape(b,q_samples)
    baseline=q_eval.mean(axis=1);advantage=q_exp-baseline
    return jax.tree_util.tree_map(jax.lax.stop_gradient,dict(observations=observations,z=z,
        actions=candidate,baseline_actions=baseline_actions.reshape(b,q_samples,d),
        q_exp=q_exp,q_eval_samples=q_eval,q_eval=baseline,q_eval_min=q_eval_min.mean(axis=1),
        advantage=advantage,weights=jnp.maximum(advantage,0.),
        accepted=(advantage>0).astype(observations.dtype)))


def evaluator_loss(params,evaluator,teacher):
    predicted=latent_actions(evaluator.replace(params=params),teacher['observations'],teacher['z'])
    target=jax.lax.stop_gradient(teacher['actions']);mask=jax.lax.stop_gradient(teacher['accepted'])
    weights=jax.lax.stop_gradient(teacher['weights'])
    errors=jnp.square(predicted-target).sum(axis=-1)
    # Preserve v2 accepted-count normalization; use raw A, not normalized weights.
    return (mask*weights*errors).sum()/jnp.maximum(mask.sum(),1.)


@jax.jit
def fit_evaluator(evaluator,teacher):
    loss,grads=jax.value_and_grad(evaluator_loss)(evaluator.params,evaluator,teacher)
    accepted_count=teacher['accepted'].sum()
    # Applying zero Adam gradients would still move parameters from old momentum.
    new=jax.lax.cond(accepted_count>0,lambda st:st.apply_gradients(grads=grads),lambda st:st,evaluator)
    grad_norm=jnp.sqrt(sum(jnp.square(v).sum() for v in jax.tree_util.tree_leaves(grads)))
    return new,loss,grad_norm


@partial(jax.jit,static_argnames=['q_samples'])
def update_evaluator(evaluator,explorer,critic,observations,key,q_samples=Q_SAMPLES):
    t=evaluator_teacher(evaluator,explorer,critic,observations,key,q_samples)
    new,loss,gn=fit_evaluator(evaluator,t)
    accepted=t['accepted'];den=jnp.maximum(accepted.sum(),1.)
    old_prediction=latent_actions(evaluator,observations,t['z'])
    new_prediction=latent_actions(new,observations,t['z'])
    metrics=dict(evaluator_loss=loss,evaluator_gradient_norm=gn,
        evaluator_acceptance_fraction=accepted.mean(),evaluator_accepted_count=accepted.sum(),
        evaluator_advantage_mean=t['advantage'].mean(),evaluator_advantage_min=t['advantage'].min(),
        evaluator_advantage_max=t['advantage'].max(),
        evaluator_accepted_advantage=(accepted*t['advantage']).sum()/den,
        evaluator_target_q_exp=t['q_exp'].mean(),evaluator_target_q_baseline=t['q_eval'].mean(),
        evaluator_target_q_baseline_min=t['q_eval_min'].mean(),
        evaluator_baseline_twin_gap=(t['q_eval']-t['q_eval_min']).mean(),
        evaluator_minmin_acceptance_fraction=(t['q_exp']>t['q_eval_min']).mean(),
        evaluator_weight_sum=t['weights'].sum(),evaluator_weight_max=t['weights'].max(),
        evaluator_accepted_weight_mean=t['weights'].sum()/den,
        evaluator_weight_ess=jnp.square(t['weights'].sum())/jnp.maximum(jnp.square(t['weights']).sum(),1e-20),
        evaluator_baseline_mc_std=jnp.std(t['q_eval_samples'],axis=1).mean(),
        evaluator_baseline_mc_se=jnp.std(t['q_eval_samples'],axis=1).mean()/jnp.sqrt(float(q_samples)),
        evaluator_update_applied=(accepted.sum()>0).astype(jnp.float32),
        evaluator_action_change_rms=jnp.sqrt(jnp.square(new_prediction-old_prediction).mean()),
        evaluator_action_boundary_fraction=(jnp.abs(old_prediction)>=1.).mean(),
        evaluator_ema_applied=jnp.asarray(0.))
    return new,metrics


class EvaluatorCallback(MujocoEvalCallback):
    def _on_step(self):
        policy=self.model.policy;assert not policy.evaluation_only
        due=self.n_calls==1 or self.n_calls%self.eval_freq==0
        if due:
            policy.evaluation_only=True
            self.logger.record('eval/uses_selective_evaluator',1)
        try:return super()._on_step()
        finally:policy.evaluation_only=False


class ExplorerOptiQ(EMAExplorerOptiQ):
    policy_aliases={'MlpPolicy':ExplorerPolicy,'MultiInputPolicy':ExplorerPolicy}

    def __init__(self,*args,**kwargs):
        # Reuse v1 TD target, routing, train/checkpoint hooks; replace EMA entirely.
        OptiQDIME.__init__(self,*args,**kwargs)
        a=self.cfg.alg
        assert a.critic.n_critics==2 and a.critic.n_atoms==1
        assert not a.critic.crossq_style and not a.optimizer.bn
        assert a.critic.dropout_rate is None and a.critic.entr_coeff==0
        assert a.utd==a.policy_delay==1
        assert a.actor.source_q_eval=='mean' and a.actor.transport_target_mode=='argmax'
        assert a.actor.td_noise_std==a.actor.td_noise_clip==0
        assert a.policy_tau==0 and a.evaluator.q_samples==Q_SAMPLES
        assert a.evaluator.learning_rate==a.optimizer.lr_actor
        assert a.evaluator.loss=='positive_advantage_weighted_action_mse'
        assert a.evaluator.matched_latent and a.evaluator.threshold==0.
        assert a.evaluator.advantage_mode=='min_max'
        assert a.evaluator.weight_mode=='positive_advantage'
        assert a.evaluator.normalization=='accepted_count'
        self.stop_requested=False;self.checkpoint_hook=None

    @staticmethod
    def soft_update_target_actor(tau,actor_state,target_actor_state):
        # Parent legacy loop calls this after explorer training. No EMA in v2.
        return target_actor_state

    @classmethod
    @partial(jax.jit, static_argnames=['cls', 'crossq_style', 'use_bnstats_from_live_net', 'gradient_steps', 'v_min', 'v_max', 'num_atoms', 'entr_coeff', 'num_policy_samples', 'proposals_per_policy_sample', 'proposal_sampling_mode', 'include_anchor', 'density_correction', 'sinkhorn_iterations', 'source_q_eval', 'transport_target_mode', 'adaptive_density_beta', 'density_beta_grid_size'])
    def _train(cls, crossq_style, use_bnstats_from_live_net, gamma, tau, policy_tau, gradient_steps, data, policy_delay_indices, qf_state, actor_state, target_actor_state, ent_coef_state, key, n_env_interacts, v_min, v_max, entr_coeff, num_atoms, num_policy_samples, proposals_per_policy_sample, proposal_sampling_mode, proposal_std, proposal_clip, include_anchor, density_correction, density_beta, adaptive_density_beta, minimum_source_ess, density_beta_grid_size, temperature, sinkhorn_epsilon, sinkhorn_iterations, source_q_eval, transport_target_mode, td_noise_std, td_noise_clip):
        assert gradient_steps == 1
        result = super()._train(crossq_style, use_bnstats_from_live_net, gamma, tau, policy_tau, gradient_steps, data, policy_delay_indices, qf_state, actor_state, target_actor_state, ent_coef_state, key, n_env_interacts, v_min, v_max, entr_coeff, num_atoms, num_policy_samples, proposals_per_policy_sample, proposal_sampling_mode, proposal_std, proposal_clip, include_anchor, density_correction, density_beta, adaptive_density_beta, minimum_source_ess, density_beta_grid_size, temperature, sinkhorn_epsilon, sinkhorn_iterations, source_q_eval, transport_target_mode, td_noise_std, td_noise_clip)
        qf, explorer, evaluator, entropy, returned_key, metrics = result
        if 0 in policy_delay_indices:
            # Extra distillation randomness does not consume the legacy RNG stream.
            distill_key = jax.random.fold_in(returned_key, 20002)
            evaluator, evaluator_metrics = update_evaluator(
                evaluator, explorer, qf, data.observations, distill_key, Q_SAMPLES)
            metrics = {**metrics, **evaluator_metrics}
        return qf, explorer, evaluator, entropy, returned_key, metrics
