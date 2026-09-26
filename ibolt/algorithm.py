"""iBOLT: plain scalar TD and direct value-weighted mixture likelihood."""
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from .learner import JaxLearner
from .policy import IBOLTPolicy
from .box_gaussian import sample_box, mixture_log_prob
from .distillation import direct_gmm_nll

class IBOLT(JaxLearner):
    policy_aliases = {'MlpPolicy': IBOLTPolicy}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backup_mode = 'td'

    def train(self, batch_size, gradient_steps):
        data = self.replay_buffer.sample(batch_size * gradient_steps, env=self._vec_normalize_env)
        arrays = [x.numpy() for x in (data.observations, data.actions, data.next_observations,
                                      data.dones, data.rewards)]
        arrays[3], arrays[4] = arrays[3].ravel(), arrays[4].ravel()
        a = self.cfg.alg.actor
        for i in range(gradient_steps):
            obs, actions, next_obs, dones, rewards = [x[i*batch_size:(i+1)*batch_size] for x in arrays]
            update_actor = self.num_timesteps > a.learning_starts and (self._n_updates+i+1) % self.policy_delay == 0
            self.policy.qf_state, self.policy.actor_state, metrics, self.key = self._step(
                self.gamma, self.tau, self.policy.actor_state, self.policy.qf_state,
                obs, actions, next_obs, dones, rewards, self.key,
                int(a.num_policy_samples), int(a.proposals_per_policy_sample),
                float(a.temperature), float(a.density_correction_beta),
                float(a.log_std_min), float(a.log_std_max), update_actor)
        self._n_updates += gradient_steps
        if self.model_save_path and (self.num_timesteps % self.save_every_n_steps == 0
                                   or self.num_timesteps == self.learning_starts + 1):
            self._save_model()
        self.logger.record('train/n_updates', self._n_updates, exclude='tensorboard')
        self.logger.record('train/actor_updates', int(self.policy.actor_state.step))
        interval = int(self.cfg.diagnostic_interval)
        due = interval > 0 and self.num_timesteps % interval == 0
        core = {'actor_loss','critic_loss','current_q_values','next_q_values','actor_std_mean'}
        for name, value in metrics.items():
            if not interval or due or name in core:
                self.logger.record('train/'+name, float(value))
        self.logger.record('train/temperature', float(a.temperature))
        if due:
            self.logger.record('time/total_timesteps', self.num_timesteps, exclude='tensorboard')
            self.logger.dump(self.num_timesteps)

    @classmethod
    @partial(jax.jit, static_argnames=['cls','n','repeats','do_actor'])
    def _step(cls,gamma,tau,actor,critic,obs,actions,next_obs,dones,rewards,key,
              n,repeats,temperature,beta,min_log_std,max_log_std,do_actor):
        critic,metrics,key = cls.update_critic(gamma,actor,critic,obs,actions,next_obs,dones,rewards,key)
        critic = cls.soft_update(tau,critic)
        if do_actor:
            actor,actor_metrics,key = cls.update_actor(actor,critic,obs,key,n,repeats,
                                                      temperature,beta,min_log_std,max_log_std)
            metrics.update(actor_metrics)
        return critic,actor,metrics,key

    @staticmethod
    @jax.jit
    def update_critic(gamma, actor, critic, obs, actions, next_obs, dones, rewards, key):
        # Keep the reference random-key split, including unused stream slots.
        key, actor_key, _, target_key, current_key, _ = jax.random.split(key, 6)
        next_actions = jax.lax.stop_gradient(IBOLTPolicy.sample_action(actor, next_obs, actor_key))
        def loss_fn(params):
            next_q = critic.apply_fn({'params':critic.target_params, 'batch_stats':critic.target_batch_stats},
                next_obs, next_actions, rngs={'dropout':target_key}, train=False)[...,0].min(axis=0)
            target = jax.lax.stop_gradient(rewards + (1-dones)*gamma*next_q)
            current, updates = critic.apply_fn({'params':params,'batch_stats':critic.batch_stats},
                obs, actions, rngs={'dropout':current_key}, mutable=['batch_stats'], train=True)
            current = current[...,0]
            loss = ((current-target[None])**2).mean(axis=1).sum()
            return loss, (updates, current.min(axis=0).mean(), target.mean())
        (loss,(updates,current,target)), grads = jax.value_and_grad(loss_fn,has_aux=True)(critic.params)
        critic = critic.apply_gradients(grads=grads)
        critic = critic.replace(batch_stats=updates.get('batch_stats',critic.batch_stats))
        return critic, dict(critic_loss=loss,current_q_values=current,next_q_values=target), key

    @staticmethod
    @partial(jax.jit, static_argnames=['n','repeats'])
    def update_actor(actor, critic, observations, key, n, repeats, temperature, beta, min_log_std, max_log_std):
        key, latent_key, proposal_key, dropout_key = jax.random.split(key,4)
        batch, obs_dim = observations.shape
        dim = actor.params['mu']['bias'].shape[0]
        z_key, eps_key = jax.random.split(latent_key)
        z = jax.random.normal(z_key,(batch,n,dim),dtype=observations.dtype)
        obs = jnp.broadcast_to(observations[:,None],(batch,n,obs_dim)).reshape(batch*n,obs_dim)
        def loss_fn(params):
            mu, log_std = actor.apply_fn({'params':params},obs,z.reshape(batch*n,dim))
            mu, log_std = mu.reshape(batch,n,dim), log_std.reshape(batch,n,dim)
            policy_samples = sample_box(eps_key,mu,log_std)
            means, scales = jax.lax.stop_gradient(mu), jax.lax.stop_gradient(log_std)
            # Preserve the reference floor (an identity at the allowed bounds).
            scales = jnp.maximum(scales, jnp.log(jnp.asarray(0.006737946999085467)))
            component_key, noise_key = jax.random.split(proposal_key)
            m = n*repeats
            indices = jax.random.randint(component_key,(batch,m),0,n)
            selected_mu = jnp.take_along_axis(means,indices[:,:,None],axis=1)
            selected_ls = jnp.take_along_axis(scales,indices[:,:,None],axis=1)
            candidates = jax.lax.stop_gradient(sample_box(noise_key,selected_mu,selected_ls))
            candidate_obs = jnp.broadcast_to(observations[:,None],(batch,m,obs_dim)).reshape(batch*m,obs_dim)
            values = critic.apply_fn({'params':critic.params,'batch_stats':critic.batch_stats},
                candidate_obs,candidates.reshape(batch*m,dim),rngs={'dropout':dropout_key},train=False)
            q = jax.lax.stop_gradient(values.reshape(2,batch,m,-1)[...,0].mean(axis=0))
            density = mixture_log_prob(candidates,means,scales)
            weights = jax.lax.stop_gradient(jax.nn.softmax(q/temperature-beta*density,axis=-1))
            loss, component = direct_gmm_nll(mu,log_std,candidates,weights)
            responsibility = jax.nn.softmax(component,axis=1)
            usage = (responsibility*weights[:,None,:]).sum(-1)
            ess = 1/(weights**2).sum(-1)
            std = jnp.exp(log_std)
            between, within = jnp.var(mu,axis=1).sum(-1), (std**2).mean(axis=1).sum(-1)
            metrics = dict(actor_loss=loss,actor_std_mean=std.mean(),actor_std_min=std.min(),
                actor_std_max=std.max(),actor_log_std_mean=log_std.mean(),actor_log_std_min=log_std.min(),
                actor_log_std_max=log_std.max(),actor_mu_abs_max=jnp.abs(mu).max(),
                actor_std_at_min_fraction=(log_std<=min_log_std+1e-6).mean(),
                actor_std_at_max_fraction=(log_std>=max_log_std-1e-6).mean(),
                actor_latent_mean_variance_fraction=(between/jnp.maximum(between+within,1e-20)).mean(),
                actor_between_mean_variance=between.mean(),actor_within_variance=within.mean(),
                source_ess_absolute=ess.mean(),source_ess_fraction=(ess/m).mean(),source_ess_min=ess.min(),
                max_source_weight=weights.max(-1).mean(),source_q_mean=q.mean(),source_q_std=q.std(-1).mean(),
                teacher_log_density_mean=density.mean(),teacher_log_density_std=density.std(-1).mean(),
                teacher_log_density_max=density.max(),teacher_action_saturation_fraction=(jnp.abs(candidates)>.99).mean(),
                student_action_saturation_fraction=(jnp.abs(policy_samples)>.99).mean(),
                policy_spread_l2=jnp.linalg.norm(policy_samples.std(1),axis=-1).mean(),
                gmm_component_ess_fraction=(1/(usage**2).sum(-1)/n).mean(),
                gmm_component_usage_min=usage.min(),gmm_underused_fraction=(usage<.1/n).mean())
            return loss, metrics
        (_,metrics), grads = jax.value_and_grad(loss_fn,has_aux=True)(actor.params)
        return actor.apply_gradients(grads=grads), metrics, key
