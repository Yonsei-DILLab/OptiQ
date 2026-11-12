"""JAX execution of the pinned SB3 SAC equations; shared SB3 rollout/HER API.

PyTorch constructs the original seeded networks once. Training and action
inference use JAX. Inference zip files and resumable JAX state are both saved.
This is private rebuttal code, never an anonymous-branch export.
"""
from functools import partial
from pathlib import Path
import hashlib
import json

import flax.serialization
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from stable_baselines3 import SAC

ADAM = optax.adam(3e-4, b1=.9, b2=.999, eps=1e-8, eps_root=0.)


@struct.dataclass
class SACState:
    actor: object
    critic: object
    target: object
    actor_opt: object
    critic_opt: object
    log_alpha: object
    alpha_opt: object
    updates: object
    key: object
    # Static mode: fixed alpha=0 needs no log(0) or entropy optimizer.
    # Restored from the saved model's ent_coef when constructing the state template.
    zero_alpha: bool = struct.field(pytree_node=False, default=False)


def _layer(module):
    # Copy before importing: CPU JAX may otherwise alias mutable Torch storage.
    return {'weight': jnp.asarray(module.weight.detach().cpu().numpy().copy()),
            'bias': jnp.asarray(module.bias.detach().cpu().numpy().copy())}


def export_actor(actor):
    return dict(hidden=tuple(_layer(m) for m in actor.latent_pi if isinstance(m, torch.nn.Linear)),
                mu=_layer(actor.mu), log_std=_layer(actor.log_std))


def critic_parameter_tree(critic, tensor_tree):
    """Keep each optional LayerNorm beside its preceding hidden Linear."""
    heads = []
    for network in critic.q_networks:
        layers = []
        for module in network:
            if isinstance(module, torch.nn.Linear):
                layers.append(tensor_tree(module))
            elif isinstance(module, torch.nn.LayerNorm):
                assert module.eps == 1e-6 and module.elementwise_affine
                assert tuple(module.normalized_shape) == tuple(layers[-1]['bias'].shape)
                assert 'norm' not in layers[-1]
                layers[-1]['norm'] = tensor_tree(module)
            elif not isinstance(module, torch.nn.ReLU):
                raise ValueError(f'Unsupported SAC critic module: {type(module).__name__}')
        assert isinstance(network[-1], torch.nn.Linear)
        heads.append(tuple(layers))
    return tuple(heads)


def export_critic(critic):
    return critic_parameter_tree(critic, _layer)


def initial_state(model):
    actor, critic, target = export_actor(model.actor), export_critic(model.critic), export_critic(model.critic_target)
    zero_alpha = model.ent_coef_optimizer is None and float(model.ent_coef_tensor) == 0.
    log_alpha_tensor = (torch.zeros_like(model.ent_coef_tensor) if zero_alpha else
                        model.log_ent_coef if model.ent_coef_optimizer else model.ent_coef_tensor.log())
    log_alpha = jnp.asarray(log_alpha_tensor.detach().cpu().numpy().copy())
    for optimizer in (model.actor.optimizer, model.critic.optimizer, model.ent_coef_optimizer):
        if optimizer is None:
            continue
        d=optimizer.defaults
        assert d['lr']==3e-4 and d['betas']==(.9,.999) and d['eps']==1e-8
        assert d['weight_decay']==0 and not d['amsgrad'] and not d.get('maximize',False)
    return SACState(actor,critic,target,ADAM.init(actor),ADAM.init(critic),log_alpha,
                    ADAM.init(log_alpha) if model.ent_coef_optimizer else None,
                    jnp.int32(0),jax.random.PRNGKey(int(model.seed or 0)),zero_alpha=zero_alpha)


def entropy_coefficient(state):
    return jnp.zeros_like(state.log_alpha) if state.zero_alpha else jnp.exp(state.log_alpha)


def linear(x, p):
    return jnp.matmul(x,p['weight'].T,precision=jax.lax.Precision.HIGHEST)+p['bias']


@jax.custom_jvp
def clamp_log_std(x):
    return jnp.clip(x,-20.,2.)


@clamp_log_std.defjvp
def _clamp_jvp(primals,tangents):
    x,=primals; dx,=tangents
    return clamp_log_std(x), jnp.where((x>=-20.)&(x<=2.),dx,0.)


def actor_distribution(params, obs):
    x=obs
    for p in params['hidden']:x=jax.nn.relu(linear(x,p))
    return linear(x,params['mu']),clamp_log_std(linear(x,params['log_std']))


def sample_actor(params,obs,epsilon):
    mean,log_std=actor_distribution(params,obs);std=jnp.exp(log_std)
    gaussian=mean+std*epsilon;actions=jnp.tanh(gaussian)
    # Exact SB3 SquashedDiagGaussianDistribution convention, including epsilon.
    normal=-((gaussian-mean)**2)/(2.*std**2)-log_std-.5*jnp.log(2.*jnp.pi)
    log_prob=normal.sum(-1)-jnp.log(1.-actions**2+1e-6).sum(-1)
    return actions,log_prob[:,None]


def critic_values(params,obs,actions):
    values=[]
    for layers in params:
        x=jnp.concatenate([obs,actions],axis=-1)
        for p in layers[:-1]:
            x = linear(x, p)
            if 'norm' in p:
                centered = x - x.mean(axis=-1, keepdims=True)
                variance = jnp.square(centered).mean(axis=-1, keepdims=True)
                x = centered * jax.lax.rsqrt(variance + 1e-6)
                x = x * p['norm']['weight'] + p['norm']['bias']
            x = jax.nn.relu(x)
        values.append(linear(x,layers[-1]))
    return jnp.stack(values)


def minimum_critic(values):
    # torch.min(dim=...) routes equal-value gradients to the first critic.
    return jnp.where(values[0]<=values[1],values[0],values[1])


@partial(jax.jit,static_argnames=['diagnostics'])
def sac_step(state,batch,epsilon_current,epsilon_next,diagnostics=False):
    obs,actions,next_obs,dones,rewards=batch
    sampled,log_prob=sample_actor(state.actor,obs,epsilon_current)
    alpha=entropy_coefficient(state)
    alpha_opt, log_alpha = state.alpha_opt, state.log_alpha
    info = dict(ent_coef=alpha.squeeze())
    if alpha_opt is not None:
        entropy_residual = jax.lax.stop_gradient(log_prob - actions.shape[-1])
        alpha_loss, alpha_grad = jax.value_and_grad(lambda p: -(p*entropy_residual).mean())(log_alpha)
        delta, alpha_opt = ADAM.update(alpha_grad, alpha_opt, log_alpha)
        log_alpha = optax.apply_updates(log_alpha, delta)
        info['ent_coef_loss'] = alpha_loss
        if diagnostics:
            info['alpha_grad'] = alpha_grad
    next_actions,next_log_prob=sample_actor(state.actor,next_obs,epsilon_next)
    next_q=minimum_critic(critic_values(state.target,next_obs,next_actions))-alpha*next_log_prob
    target=jax.lax.stop_gradient(rewards+(1.-dones)*.99*next_q)

    def critic_loss_fn(params):
        return .5*((critic_values(params,obs,actions)-target[None])**2).mean(axis=(1,2)).sum()
    critic_loss,critic_grad=jax.value_and_grad(critic_loss_fn)(state.critic)
    delta,critic_opt=ADAM.update(critic_grad,state.critic_opt,state.critic)
    critic=optax.apply_updates(state.critic,delta)

    def actor_loss_fn(params):
        a,lp=sample_actor(params,obs,epsilon_current)
        q=minimum_critic(critic_values(critic,obs,a))
        return (alpha*lp-q).mean()
    actor_loss,actor_grad=jax.value_and_grad(actor_loss_fn)(state.actor)
    delta,actor_opt=ADAM.update(actor_grad,state.actor_opt,state.actor)
    actor=optax.apply_updates(state.actor,delta)
    target_params=jax.tree_util.tree_map(lambda old,new:old*(1.-.005)+new*.005,state.target,critic)
    updated=state.replace(actor=actor,critic=critic,target=target_params,actor_opt=actor_opt,
        critic_opt=critic_opt,log_alpha=log_alpha,alpha_opt=alpha_opt,updates=state.updates+1)
    info.update(actor_loss=actor_loss, critic_loss=critic_loss)
    if diagnostics:
        info.update(td_target=target,actions=sampled,log_prob=log_prob,actor_grad=actor_grad,
                    critic_grad=critic_grad)
    return updated,info


@jax.jit
def train_fixed_noise(state,batches,current_noise,next_noise):
    def body(s,inputs):
        b,ec,en=inputs
        return sac_step(s,b,ec,en)
    return jax.lax.scan(body,state,(batches,current_noise,next_noise))


@jax.jit
def train_many(state,batches):
    def body(s,b):
        key,kc,kn=jax.random.split(s.key,3)
        shape=(b[0].shape[0],s.actor['mu']['bias'].shape[0])
        return sac_step(s.replace(key=key),b,jax.random.normal(kc,shape),jax.random.normal(kn,shape))
    return jax.lax.scan(body,state,batches)


@partial(jax.jit,static_argnames=['deterministic'])
def action_from_state(state,obs,deterministic):
    if deterministic:
        mean,_=actor_distribution(state.actor,obs)
        return state.key,jnp.tanh(mean)
    key,noise_key=jax.random.split(state.key)
    actions,_=sample_actor(state.actor,obs,jax.random.normal(noise_key,(obs.shape[0],state.actor['mu']['bias'].shape[0])))
    return key,actions


class JaxSAC(SAC):
    """Keep SB3 collection, timeouts, normalization and schedules unchanged."""
    def __init__(self,*args,device='cpu',**kwargs):
        super().__init__(*args,device='cpu',**kwargs)
        if not hasattr(self,'actor'):return
        assert self.gamma==.99 and self.tau==.005 and self.target_update_interval==1
        assert self.target_entropy == -float(np.prod(self.action_space.shape)) and not self.use_sde
        assert self.ent_coef == 'auto' or (not isinstance(self.ent_coef, str) and np.isfinite(self.ent_coef) and self.ent_coef >= 0)
        assert self.batch_size==256 and self.policy.net_arch in ([256,256],[256,256,256])
        assert len(self.critic.q_networks)==2
        assert not any(isinstance(m, torch.nn.LayerNorm) for m in self.actor.modules())
        self.jax_state=initial_state(self)
        self.jax_initial_digest=self.actor_digest()
        self.jax_last_audit={}

    def actor_digest(self):
        return hashlib.sha256(flax.serialization.to_bytes(self.jax_state.actor)).hexdigest()

    def predict(self,observation,state=None,episode_start=None,deterministic=False):
        obs=np.asarray(observation,dtype=np.float32);single=obs.ndim==1
        if single:obs=obs[None]
        key,actions=action_from_state(self.jax_state,obs,deterministic)
        if not deterministic:self.jax_state=self.jax_state.replace(key=key)
        action=np.asarray(actions)
        # Match SB3 BasePolicy.predict: map normalized tanh output to env bounds.
        action=self.policy.unscale_action(action)
        return (action[0] if single else action),state

    def train(self,gradient_steps,batch_size=256):
        assert batch_size == 256 and gradient_steps > 0
        data=self.replay_buffer.sample(batch_size*gradient_steps,env=self._vec_normalize_env)
        arrays=[x.detach().cpu().numpy() for x in (data.observations,data.actions,data.next_observations,data.dones,data.rewards)]
        batches=tuple(x.reshape((gradient_steps,batch_size)+x.shape[1:]) for x in arrays)
        self.jax_state,metrics=train_many(self.jax_state,batches)
        # One synchronization per group, rather than repeated scalar copies per update.
        values=jax.device_get({k:v.mean() for k,v in metrics.items()})
        assert all(np.isfinite(v).all() for v in values.values())
        self._n_updates+=gradient_steps
        self.logger.record('train/n_updates',self._n_updates,exclude='tensorboard')
        self.logger.record('train/learning_rate',3e-4)
        for k,v in values.items():self.logger.record('train/'+k,float(v))

    def sync_torch_policy(self):
        def copy(module,params):
            with torch.no_grad():
                module.weight.copy_(torch.from_numpy(np.asarray(params['weight']).copy()))
                module.bias.copy_(torch.from_numpy(np.asarray(params['bias']).copy()))
        p=jax.device_get(self.jax_state)
        layers=[m for m in self.actor.latent_pi if isinstance(m,torch.nn.Linear)]
        for m,w in zip(layers,p.actor['hidden'],strict=True):copy(m,w)
        copy(self.actor.mu,p.actor['mu']);copy(self.actor.log_std,p.actor['log_std'])
        for model,weights in [(self.critic,p.critic),(self.critic_target,p.target)]:
            for q,params in zip(model.q_networks,weights,strict=True):
                parameter_layers = []
                for layer in params:
                    parameter_layers.append(layer)
                    if 'norm' in layer:
                        parameter_layers.append(layer['norm'])
                modules = (m for m in q if isinstance(m, (torch.nn.Linear, torch.nn.LayerNorm)))
                for m,w in zip(modules,parameter_layers,strict=True):copy(m,w)
        if self.log_ent_coef is not None:
            with torch.no_grad():self.log_ent_coef.copy_(torch.from_numpy(np.asarray(p.log_alpha).copy()))

    def _excluded_save_params(self):
        return super()._excluded_save_params()+['jax_state','jax_initial_digest','jax_last_audit']

    def save(self,path,*args,**kwargs):
        self.sync_torch_policy()
        super().save(path,*args,**kwargs)
        path=Path(path)
        blob=path.with_suffix('.jax.msgpack')
        temp=blob.with_suffix('.tmp');temp.write_bytes(flax.serialization.to_bytes(self.jax_state));temp.replace(blob)
        meta=dict(steps=self.num_timesteps,updates=self._n_updates,backend=jax.default_backend(),
            networks=self.policy.net_arch,format='SB3 inference zip + complete JAX optimizer/RNG state',
            zip_training_resume='Use JAX state sidecar; Torch optimizer state is intentionally unused.',
            sha256=hashlib.sha256(blob.read_bytes()).hexdigest())
        path.with_suffix('.jax.json').write_text(json.dumps(meta,indent=2)+'\n')

    def restore_jax_state(self,path):
        path=Path(path);blob=path.with_suffix('.jax.msgpack');meta=json.loads(path.with_suffix('.jax.json').read_text())
        assert hashlib.sha256(blob.read_bytes()).hexdigest()==meta['sha256']
        assert list(meta['networks'])==list(self.policy.net_arch)
        if not hasattr(self, 'jax_state'):
            self.jax_state = initial_state(self)
        self.jax_state=flax.serialization.from_bytes(self.jax_state,blob.read_bytes())
        self._n_updates=int(self.jax_state.updates);self.num_timesteps=meta['steps']
        assert self._n_updates==meta['updates']
        self.sync_torch_policy()

    @classmethod
    def load(cls, path, *args, **kwargs):
        model = super().load(path, *args, **kwargs)
        model.restore_jax_state(path)
        return model
