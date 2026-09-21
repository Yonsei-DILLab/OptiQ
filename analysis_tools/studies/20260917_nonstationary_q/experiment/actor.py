"""Same proposal and NLL as v5; only assignment, fixed sigma, and Q provider vary."""
import functools
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from optiq_dime.semi_implicit import ConditionalGaussianProposal
from optiq_dime.transport import sinkhorn
from optiq_dime.distillation import conditional_ot_nll,direct_gmm_nll
from .models import actor_apply,q_mean
from .problems import analytic_q_jax,GRID
from .config import METHODS
from .exact1d import monotone_plan

@functools.lru_cache(None)
def engine(method,n,m,qkind):
    solver,fixed=METHODS[method];apply=actor_apply(method)
    assert m%n==0 and qkind in ('analytic','replay','critic')
    def heads(params,obs,z):
        b=obs.shape[0]
        ob=jnp.broadcast_to(obs[:,None,:],(b,z.shape[1],1))
        mu,ls=apply({'params':params},ob.reshape(-1,1),z.reshape(-1,1))
        return mu.reshape(b,-1,1),ls.reshape(b,-1,1)
    def qvalue(obs,a,qarg):
        if qkind=='analytic':return analytic_q_jax(a,qarg)
        if qkind=='replay':return jnp.interp(a[...,0],jnp.asarray(GRID,jnp.float32),qarg)
        return q_mean(qarg,obs,a)
    def teacher_impl(params,obs,key,qarg):
        key,lk,pk,_=jax.random.split(key,4);zk,ek=jax.random.split(lk)
        z=jax.random.normal(zk,(obs.shape[0],n,1));mu,ls=heads(params,obs,z)
        eps=jax.random.normal(ek,mu.shape)
        proposal=ConditionalGaussianProposal(jax.lax.stop_gradient(mu),jax.lax.stop_gradient(ls),.05)
        b,u,indices=proposal.sample(pk,m//n,'exact');lq=proposal.log_prob(u)
        q=qvalue(obs,b,qarg);logits=q/.25-lq;w=jax.nn.softmax(logits,axis=-1)
        t=dict(obs=obs,z=z,mu=mu,log_sigma=ls,positions=jnp.tanh(mu),b=b,u=u,component=indices,
               Q=q,log_q=lq,logits=logits,w=w,student_eps=eps)
        if solver=='sinkhorn':
            c=jnp.square(t['positions'][:,:,None]-b[:,None]).sum(-1)
            P=sinkhorn(c,w,.1,100)
            t.update(P=P,R=P/jnp.maximum(P.sum(-1,keepdims=True),1e-20))
        return jax.tree_util.tree_map(jax.lax.stop_gradient,t),key
    teachjit=jax.jit(teacher_impl)
    def teacher(params,obs,key,qarg):
        t,key=teachjit(params,obs,key,qarg)
        if solver=='exact':
            pos,b,w=jax.device_get((t['positions'],t['b'],t['w']))
            P=np.stack([monotone_plan(x[:,0],y[:,0],v) for x,y,v in zip(pos,b,w)])
            P=jnp.asarray(P);t=dict(t,P=P,R=P/jnp.maximum(P.sum(-1,keepdims=True),1e-20))
        return t,key
    def loss(params,t):
        mu,ls=heads(params,t['obs'],t['z'])
        if solver=='gmm':return direct_gmm_nll(mu,ls,t['u'],t['w'])[0]
        return conditional_ot_nll(mu,ls,t['u'],t['R'])
    @jax.jit
    def update(state,t):
        value,g=jax.value_and_grad(loss)(state.params,t)
        return state.apply_gradients(grads=g),value,jnp.sqrt(sum(jnp.square(v).sum() for v in jax.tree_util.tree_leaves(g)))
    @jax.jit
    def fused(state,obs,key,qarg):
        t,key=teacher_impl(state.params,obs,key,qarg);new,v,gn=update(state,t)
        return new,key,t,v,gn
    def step(state,obs,key,qarg):
        if solver!='exact':return fused(state,obs,key,qarg)
        t,key=teacher(state.params,obs,key,qarg);new,v,gn=update(state,t)
        return new,key,t,v,gn
    @jax.jit
    def common_nll(params,t):
        mu,ls=heads(params,t['obs'],t['z'])
        # Include teacher Jacobian for the common action-space likelihood.
        base,ell=direct_gmm_nll(mu,ls,t['u'],t['w'])
        jac=2*(jnp.log(2.)-t['u']-jax.nn.softplus(-2*t['u']))
        return base+(t['w']*jac.sum(-1)).sum(-1).mean(),ell
    @jax.jit
    def scalars(newparams,t,loss_value,gn):
        mu,ls=heads(newparams,t['obs'],t['z']);sigma=jnp.exp(ls)
        r=dict(loss=loss_value,gradient_norm=gn,ess=(1/jnp.square(t['w']).sum(-1)).mean(),
            wmax=t['w'].max(),sigma_mean=sigma.mean(),sigma_min=sigma.min(),sigma_max=sigma.max(),
            between_mu_variance=jnp.var(mu,axis=1).mean(),within_variance=jnp.square(sigma).mean(),
            mean_step_rms=jnp.sqrt(jnp.square(mu-t['mu']).mean()),teacher_q_std=t['Q'].std(-1).mean())
        if solver!='gmm':
            r.update(ot_row_l1=jnp.abs(t['P'].sum(-1)-1/n).sum(-1).mean(),
                ot_col_l1=jnp.abs(t['P'].sum(-2)-t['w']).sum(-1).mean(),
                effective_col_l1=jnp.abs(t['R'].sum(-2)/n-t['w']).sum(-1).mean())
        return r
    @jax.jit
    def assignment(params,t):
        common,ell=common_nll(params,t)
        post=t['w'][:,None]*jax.nn.softmax(ell,axis=1)
        A=post if solver=='gmm' else t['R']/n
        alpha=A.sum(-1);R=A/jnp.maximum(alpha[...,None],1e-30)
        mean=R@t['u'];var=jnp.maximum(R@jnp.square(t['u'])-jnp.square(mean),0)
        return dict(A=A,alpha=alpha,row_variance=var,row_mean=mean,common_nll=common,
            usage_ess=(1/jnp.square(alpha).sum(-1)).mean(),
            underused_fraction=(alpha<.1/n).mean(),row_entropy=-(R*jnp.log(jnp.maximum(R,1e-30))).sum(-1).mean())
    return dict(step=step,teacher=teacher,update=update,loss=jax.jit(loss),heads=jax.jit(heads),
        qvalue=jax.jit(qvalue),scalars=scalars,assignment=assignment,common_nll=common_nll)
