"""Production v5 actors/critic; original deterministic implicit baseline stays distinct."""
import functools
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from common.type_aliases import RLTrainState
from optiq_dime.policy import SemiImplicitActor,OptiQPolicy
from legacy_optiq.policy import ImplicitActor
from optiq_dime.algorithm import OptiQDIME
from models.critic import VectorCritic
from models.utils import activation_fn
from .config import METHODS

@functools.lru_cache(None)
def actor_model(dim,legacy=False):
 return ImplicitActor(dim,(256,256,256)) if legacy else SemiImplicitActor(dim,(256,256),-5.,1.,float(np.log(.5)))
@functools.lru_cache(None)
def actor_apply(method,dim):
 solver,fixed,_=METHODS[method];model=actor_model(dim,solver=='legacy')
 if solver=='legacy':return model.apply
 def apply(variables,observations,z):
  mu,ls=model.apply(variables,observations,z)
  return mu,ls if fixed is None else jnp.full_like(ls,np.log(fixed))
 return apply

def actor_state(method,seed,dim):
 p=actor_model(dim,METHODS[method][0]=='legacy').init(jax.random.PRNGKey(seed),jnp.zeros((1,dim)),jnp.zeros((1,dim)))['params']
 return TrainState.create(apply_fn=actor_apply(method,dim),params=p,tx=optax.adam(3e-4))
@functools.lru_cache(None)
def critic_model():return VectorCritic(net_arch=(256,256),activation_fn=activation_fn['gelu'],batch_norm_momentum=.99,use_batch_norm=False,batch_norm_mode='brn_actor',use_layer_norm=False,dropout_rate=None,n_critics=2,n_atoms=1)

def critic_state(seed,dim):
 k=jax.random.PRNGKey(310000+seed);v=critic_model().init({'params':k,'dropout':k,'batch_stats':k},jnp.zeros((1,dim)),jnp.zeros((1,dim)),train=False)
 return RLTrainState.create(apply_fn=critic_model().apply,params=v['params'],batch_stats=v.get('batch_stats',{}),target_params=v['params'],target_batch_stats=v.get('batch_stats',{}),tx=optax.adam(3e-4))
@jax.jit
def q_mean(params,obs,a):
 d=a.shape[-1];ob=jnp.broadcast_to(obs[:,None,:],a.shape)
 q=critic_model().apply({'params':params},ob.reshape(-1,d),a.reshape(-1,d),train=False)
 return q[...,0].mean(0).reshape(a.shape[:-1])
@jax.jit
def q_reference_mean(params,obs,a):
 with jax.default_matmul_precision('highest'):
  d=a.shape[-1];ob=jnp.broadcast_to(obs[:,None,:],a.shape)
  q=critic_model().apply({'params':params},ob.reshape(-1,d),a.reshape(-1,d),train=False)
  return q[...,0].mean(0).reshape(a.shape[:-1])
@jax.jit
def td_update(actor,critic,batch,key):
 critic,metrics,key=OptiQDIME.update_critic(False,False,.99,actor,critic,batch['s'],batch['a'],batch['sp'],batch['r'],jnp.zeros_like(batch['r']),1,jnp.array([-3600.]),-3600.,3600.,0.,0.,0.,key,'mu' in actor.params,0,.25,'td')
 return OptiQDIME.soft_update(.005,critic),metrics,key
sample_action=OptiQPolicy.sample_action
