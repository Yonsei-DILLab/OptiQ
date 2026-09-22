"""MFPO actor/divergence updates in native normalized actions with an oracle Q.

Uses upstream MeanFlow, time embeddings, sampler and time distribution directly.
The velocity/JVP/divergence equations follow MeanFlowLearner.update_actor/logp.
"""
import math
import sys
from functools import partial

import flax.linen as nn
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from .target import ROOT, SCALE

sys.path.insert(0,str(ROOT/"gmm40-baseline/MFPO"))
from jaxrl5.networks.mlp import MLP
from jaxrl5.networks.mean_flow import MeanFlow,TimestepEmbedder,action_sampler_with_logp,action_sampler,tr_sampler


class MFPO:
    def __init__(self,target,seed,batch):
        self.target,self.batch,self.updates=target,batch,0
        self.key,ak,dk=jax.random.split(jax.random.PRNGKey(seed),3)
        embedding=partial(TimestepEmbedder,frequency_embedding_size=128,fourier_feature_learnable=True,hidden_size=64,activations=nn.gelu)
        self.actor=MeanFlow(embedding,embedding,partial(MLP,hidden_dims=(256,256,256,2),activations=nn.gelu,use_layer_norm=True))
        self.div=MeanFlow(embedding,embedding,partial(MLP,hidden_dims=(256,256,256,1),activations=nn.gelu,use_layer_norm=True))
        dummy=(jnp.zeros((1,1)),jnp.zeros((1,2)),jnp.ones((1,1)),jnp.ones((1,1)))
        self.state=TrainState.create(apply_fn=self.actor.apply,params=self.actor.init(ak,*dummy)["params"],tx=optax.adam(3e-4))
        self.divstate=TrainState.create(apply_fn=self.div.apply,params=self.div.init(dk,*dummy)["params"],tx=optax.adam(3e-4))
        self.advance_fn=jax.jit(self._advance,static_argnums=3)
        self.sample_fn=jax.jit(self._sample,static_argnums=2)

    def q_u(self,u):
        return self.target.jax_log_prob(SCALE*u)

    def _update(self,carry,_):
        state,divstate,key=carry
        keys=jax.random.split(key,9); key=keys[0]
        b=self.batch; k1,k2=16,32
        obs=jnp.zeros((b,1))
        t,r=tr_sampler(keys[1],b,"logit_normal",.75)
        policy,logp=action_sampler_with_logp(state.apply_fn,state.params,divstate.apply_fn,divstate.params,2,
                      jax.random.normal(keys[2],(b*k1,2)),jnp.zeros((b*k1,1)),True)
        policy=policy.reshape(b,k1,2); logp=logp.reshape(b,k1)
        noisy=(1-t)*policy[:,0]+t*jax.random.normal(keys[3],(b,2))
        lower=(noisy[:,None,:]-(1-t[:,:,None]))/t[:,:,None]
        upper=(noisy[:,None,:]+(1-t[:,:,None]))/t[:,:,None]
        clean_key,fallback_key=jax.random.split(keys[4])
        noise=jax.random.truncated_normal(clean_key,lower=lower,upper=upper,shape=(b,k2,2))
        fallback=jnp.clip(jax.random.normal(fallback_key,(b,k2,2)),lower,upper)
        noise=jnp.where(jnp.isnan(noise),fallback,noise)
        clean=(noisy[:,None,:]-t[:,:,None]*noise)/(1-t[:,:,None])
        critic=self.q_u(jnp.concatenate([policy,clean],axis=1))
        lp=critic[:,:k1]-jnp.sum((noisy[:,None,:]-(1-t[:,:,None])*policy)**2,axis=-1)/(2*t**2)-logp
        wp=jax.nn.softmax(lp,axis=1); wc=jax.nn.softmax(critic[:,k1:],axis=1)
        vp=jnp.sum(wp[:,:,None]*(noisy[:,None,:]-policy)/t[:,:,None],axis=1)
        vc=jnp.sum(wc[:,:,None]*(noisy[:,None,:]-clean)/t[:,:,None],axis=1)
        ep=1/jnp.sum(wp**2,axis=1); ec=1/jnp.sum(wc**2,axis=1)
        blend=(ep/(ep+ec+1e-10))[:,None]
        velocity=jax.lax.stop_gradient(blend*vp+(1-blend)*vc)
        def actor_loss(params):
            def u_fn(x,t,r): return state.apply_fn({"params":params},obs,x,t,t-r)
            pred,derivative=jax.jvp(u_fn,(noisy,t,r),(velocity,jnp.ones_like(t),jnp.zeros_like(r)))
            target=jax.lax.stop_gradient(velocity-(t-r)*derivative)
            return jnp.mean((target-pred)**2)
        loss,grad=jax.value_and_grad(actor_loss)(state.params)
        state=state.apply_gradients(grads=grad)

        # Average divergence learner from upstream update_logp.
        dt,dr=tr_sampler(keys[5],b,"logit_normal",.75)
        noisy_div=(1-dt)*policy[:,0]+dt*jax.random.normal(keys[6],(b,2))
        probes=jax.random.normal(keys[6],(b,2,2))
        repeated=jnp.repeat(noisy_div[:,None,:],2,axis=1)
        repeated_obs=jnp.zeros((b,2,1)); repeated_t=jnp.repeat(dt[:,None,:],2,axis=1)
        def velocity_fn(x,t,r): return state.apply_fn({"params":state.params},repeated_obs,x,t,t-r)
        vel,jvp=jax.jvp(velocity_fn,(repeated,repeated_t,repeated_t),(probes,jnp.zeros_like(repeated_t),jnp.zeros_like(repeated_t)))
        divergence=jnp.mean(jnp.sum(jvp*probes,axis=-1),axis=1,keepdims=True)
        def div_loss(params):
            def fn(x,t,r): return divstate.apply_fn({"params":params},obs,x,t,t-r)
            pred,derivative=jax.jvp(fn,(noisy_div,dt,dr),(vel[:,0],jnp.ones_like(dt),jnp.zeros_like(dr)))
            target=jax.lax.stop_gradient(divergence-(dt-dr)*derivative)
            return jnp.mean((target-pred)**2)
        dloss,dgrad=jax.value_and_grad(div_loss)(divstate.params)
        divstate=divstate.apply_gradients(grads=dgrad)
        info=dict(loss=loss,divergence_loss=dloss,policy_ess=ep.mean(),gaussian_ess=ec.mean(),
                  proposal_logp=logp.mean(),policy_u_abs_max=jnp.abs(policy).max())
        return (state,divstate,key),info

    def _advance(self,state,divstate,key,count):
        (state,divstate,key),info=jax.lax.scan(self._update,(state,divstate,key),None,length=count)
        return state,divstate,key,jax.tree_util.tree_map(lambda x:x.mean(),info)

    def advance(self,count):
        self.state,self.divstate,self.key,info=self.advance_fn(self.state,self.divstate,self.key,count)
        self.updates+=count
        return {**{k:float(v) for k,v in info.items()},"Q_evaluations":self.updates*self.batch*48}

    def _sample(self,params,key,n):
        u=jax.random.normal(key,(n,2)); path=[SCALE*u[:128]]
        for t in (1.,.5):
            velocity=self.actor.apply({"params":params},jnp.zeros((n,1)),u,jnp.full((n,1),t),jnp.full((n,1),.5))
            u=u-.5*velocity
            path.append(SCALE*u[:128])
        x=SCALE*jnp.clip(u,-1,1)
        path[-1]=x[:128]
        return x,jnp.stack(path)

    def evaluate_samples(self,n,seed):
        x,path=self.sample_fn(self.state.params,jax.random.PRNGKey(seed),n)
        return np.asarray(x),np.asarray(path),{}

    def save(self,path):
        path.write_bytes(flax.serialization.to_bytes(dict(state=self.state,divstate=self.divstate,key=self.key,updates=self.updates)))

    def restore(self,path):
        saved=flax.serialization.from_bytes(dict(state=self.state,divstate=self.divstate,key=self.key,updates=0),path.read_bytes())
        self.state,self.divstate,self.key,self.updates=saved["state"],saved["divstate"],saved["key"],int(saved["updates"])
