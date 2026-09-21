"""Frozen teacher actor updates; epsilon only changes the registered Sinkhorn solve."""
import functools
import numpy as np
import jax
import jax.numpy as jnp
import optax
from common.type_aliases import RLTrainState
from optiq_dime.semi_implicit import ConditionalGaussianProposal
from optiq_dime.transport import sinkhorn
from optiq_dime.distillation import conditional_ot_nll,direct_gmm_nll
from legacy_optiq.algorithm import OptiQDIME as LegacyOptiQ
from .models import actor_apply,q_mean
from .problems import analytic_q_jax
from .config import METHODS
from .exact_nd import exact_plan

@functools.lru_cache(None)
def engine(method,n,m,dim,qkind):
 solver,fixed,epsilon=METHODS[method];apply=actor_apply(method,dim);legacy=solver=='legacy'
 def heads(params,obs,z):
  b=obs.shape[0];ob=jnp.broadcast_to(obs[:,None,:],(b,z.shape[1],dim))
  result=apply({'params':params},ob.reshape(-1,dim),z.reshape(-1,dim))
  mu,ls=(result,jnp.zeros_like(result)) if legacy else result
  return mu.reshape(b,-1,dim),ls.reshape(b,-1,dim)
 def qvalue(obs,a,qarg):return analytic_q_jax(a,qarg) if qkind=='analytic' else q_mean(qarg,obs,a)
 def teacher_impl(params,obs,key,qarg):
  key,lk,pk,_=jax.random.split(key,4);zk,ek=jax.random.split(lk)
  z=jax.random.normal(zk,(obs.shape[0],n,dim));mu,ls=heads(params,obs,z);eps=jax.random.normal(ek,mu.shape)
  proposal=ConditionalGaussianProposal(jax.lax.stop_gradient(mu),jax.lax.stop_gradient(ls),.05)
  b,u,indices=proposal.sample(pk,m//n,'exact');lq=proposal.log_prob(u);q=qvalue(obs,b,qarg)
  logits=q/.25-lq;w=jax.nn.softmax(logits,axis=-1)
  t=dict(obs=obs,z=z,mu=mu,log_sigma=ls,positions=jnp.tanh(mu),b=b,u=u,component=indices,Q=q,log_q=lq,logits=logits,w=w,student_eps=eps)
  if solver=='sinkhorn':
   cost=jnp.square(t['positions'][:,:,None]-b[:,None]).sum(-1)
   P=sinkhorn(cost,w,epsilon,100);t.update(P=P,R=P/jnp.maximum(P.sum(-1,keepdims=True),1e-20))
  return jax.tree_util.tree_map(jax.lax.stop_gradient,t),key
 teachjit=jax.jit(teacher_impl)
 def teacher(params,obs,key,qarg):
  if legacy:
   from .models import actor_state
   dummy=actor_state(method,0,dim).replace(params=params)
   _,key,t,_,_=legacy_step(dummy,obs,key,qarg)
   return t,key
  t,key=teachjit(params,obs,key,qarg)
  if solver=='exact':
   pos,b,w=jax.device_get((t['positions'],t['b'],t['w']))
   P=jnp.asarray(np.stack([exact_plan(x,y,v) for x,y,v in zip(pos,b,w)]))
   t=dict(t,P=P,R=P/jnp.maximum(P.sum(-1,keepdims=True),1e-20))
  return t,key
 def loss(params,t):
  mu,ls=heads(params,t['obs'],t['z'])
  if legacy:return jnp.square(mu-t['selected']).sum(-1).mean()
  if solver=='gmm':return direct_gmm_nll(mu,ls,t['u'],t['w'])[0]
  return conditional_ot_nll(mu,ls,t['u'],t['R'])
 @jax.jit
 def update(state,t):
  value,g=jax.value_and_grad(loss)(state.params,t)
  return state.apply_gradients(grads=g),value,jnp.sqrt(sum(jnp.square(v).sum() for v in jax.tree_util.tree_leaves(g)))
 @jax.jit
 def fused(state,obs,key,qarg):
  t,key=teacher_impl(state.params,obs,key,qarg);new,value,gn=update(state,t)
  return new,key,t,value,gn
 def oracle_apply(variables,observations,actions,**kwargs):
  q=qvalue(observations,actions[:,None,:],variables['params'])[:,0]
  return jnp.stack([q,q],axis=0)[...,None]
 @jax.jit
 def legacy_step(state,obs,key,qarg):
  oracle=RLTrainState.create(apply_fn=oracle_apply,params=qarg,target_params=qarg,batch_stats={},target_batch_stats={},tx=optax.identity())
  new,value,nk,cloud=LegacyOptiQ.update_actor(actor_state=state,qf_state=oracle,observations=obs,key=key,z_atoms=jnp.ones(1),
   num_policy_samples=n,proposals_per_policy_sample=m//n,proposal_sampling_mode='stratified',proposal_std=.2,proposal_clip=.5,include_anchor=False,
   density_correction=True,density_beta=1.,adaptive_density_beta=False,minimum_source_ess=float(n),density_beta_grid_size=257,
   temperature=.25,sinkhorn_epsilon=.05,sinkhorn_iterations=30,source_q_eval='mean',transport_target_mode='argmax',return_clouds=True)
  z=jax.random.normal(jax.random.split(key,4)[1],(obs.shape[0],n,dim),dtype=obs.dtype)
  mu,ls=heads(state.params,obs,z);P=cloud['cloud_transport'];b=cloud['cloud_proposals'];w=cloud['cloud_weights'];q=qvalue(obs,b,qarg);lq=cloud['cloud_log_density']
  t=dict(obs=obs,z=z,mu=mu,log_sigma=ls,positions=cloud['cloud_transport_sources'],b=b,w=w,Q=q,log_q=lq,logits=q/.25-lq,
   P=P,R=P/jnp.maximum(P.sum(-1,keepdims=True),1e-20),selected=cloud['cloud_selected'],u=jnp.arctanh(jnp.clip(b,-1+1e-6,1-1e-6)))
  g=jax.grad(loss)(state.params,t);gn=jnp.sqrt(sum(jnp.square(v).sum() for v in jax.tree_util.tree_leaves(g)))
  return new,nk,t,value,gn
 def step(state,obs,key,qarg):
  if legacy:return legacy_step(state,obs,key,qarg)
  if solver!='exact':return fused(state,obs,key,qarg)
  t,key=teacher(state.params,obs,key,qarg);new,v,gn=update(state,t);return new,key,t,v,gn
 @jax.jit
 def common_nll(params,t):
  if legacy:return jnp.array(float('nan')),jnp.zeros((t['obs'].shape[0],n,m))
  mu,ls=heads(params,t['obs'],t['z']);base,ell=direct_gmm_nll(mu,ls,t['u'],t['w'])
  jac=2*(jnp.log(2.)-t['u']-jax.nn.softplus(-2*t['u']))
  return base+(t['w']*jac.sum(-1)).sum(-1).mean(),ell
 @jax.jit
 def scalars(params,t,value,gn):
  mu,ls=heads(params,t['obs'],t['z']);sigma=jnp.zeros_like(ls) if legacy else jnp.exp(ls)
  r=dict(loss=value,gradient_norm=gn,ess=(1/jnp.square(t['w']).sum(-1)).mean(),wmax=t['w'].max(),sigma_mean=sigma.mean(),sigma_min=sigma.min(),sigma_max=sigma.max(),
   between_mu_variance=jnp.var(mu,axis=1).mean(),within_variance=jnp.square(sigma).mean(),mean_step_rms=jnp.sqrt(jnp.square(mu-t['mu']).mean()))
  if solver!='gmm':
   cost=jnp.square(t['positions'][:,:,None]-t['b'][:,None]).sum(-1)
   r.update(cost_mean=cost.mean(),cost_max=cost.max())
   r.update(ot_row_l1=jnp.abs(t['P'].sum(-1)-1/n).sum(-1).mean(),ot_col_l1=jnp.abs(t['P'].sum(-2)-t['w']).sum(-1).mean(),effective_col_l1=jnp.abs(t['R'].sum(-2)/n-t['w']).sum(-1).mean(),zero_rows=(t['P'].sum(-1)<1e-20).mean())
  return r
 @jax.jit
 def assignment(params,t):
  common,ell=common_nll(params,t)
  A=t['w'][:,None]*jax.nn.softmax(ell,axis=1) if solver=='gmm' else t['R']/n
  alpha=A.sum(-1);R=A/jnp.maximum(alpha[...,None],1e-30)
  mean=R@t['u'];var=jnp.maximum(R@jnp.square(t['u'])-jnp.square(mean),0)
  return dict(A=A,alpha=alpha,row_variance=var,row_mean=mean,common_nll=common,usage_ess=(1/jnp.square(alpha).sum(-1)).mean(),underused_fraction=(alpha<.1/n).mean(),row_entropy=-(R*jnp.log(jnp.maximum(R,1e-30))).sum(-1).mean())
 return dict(step=step,teacher=teacher,update=update,loss=jax.jit(loss),heads=jax.jit(heads),qvalue=jax.jit(qvalue),scalars=scalars,assignment=assignment,common_nll=common_nll)
