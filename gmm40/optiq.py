"""v5 actor update using the original actor, proposal, Sinkhorn and NLL code."""
import math
from functools import partial

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from optiq_dime.policy import SemiImplicitActor
from optiq_dime.semi_implicit import ConditionalGaussianProposal
from optiq_dime.transport import sinkhorn
from optiq_dime.distillation import conditional_ot_nll, direct_gmm_nll
from gmm40.row_balance import sigma_row_balanced_ot_nll


def truncate_plan_rows(plan, threshold):
    """Select raw P entries strictly above threshold, then normalize each row.

    Preserve the original row if truncation removes every entry; a zero row
    would otherwise turn moment-based NLL into supervision at the origin.
    """
    full_row=plan/jnp.maximum(plan.sum(-1,keepdims=True),1e-20)
    raw_keep=plan>threshold
    empty=~jnp.any(raw_keep,axis=-1,keepdims=True)
    keep=jnp.where(empty,jnp.ones_like(raw_keep),raw_keep).astype(plan.dtype)
    retained=(full_row*keep).sum(-1,keepdims=True)
    row=jax.lax.stop_gradient(full_row*keep/jnp.maximum(retained,1e-20))
    return row,keep,retained,empty,raw_keep.sum(-1)


class OptiQ:
    def __init__(self,target,seed=0,n=16,m=64,batch=256,epsilon=.1,hidden_dims=(256,256),sinkhorn_iterations=100,
                 mean_output_init_scale=1e-4,temperature=1.0,nll_top_k=None,nll_plan_threshold=None,
                 sigma_row_balance=False,distillation_loss="conditional_ot_nll"):
        assert m%n==0
        assert sinkhorn_iterations>0
        if not math.isfinite(mean_output_init_scale) or mean_output_init_scale <= 0:
            raise ValueError('mean_output_init_scale must be positive and finite')
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('temperature must be positive and finite')
        if nll_top_k is not None and (not isinstance(nll_top_k,int) or not 1 <= nll_top_k <= m):
            raise ValueError('nll_top_k must be an integer between 1 and teacher count m')
        if nll_plan_threshold is not None and (not math.isfinite(nll_plan_threshold) or nll_plan_threshold<=0):
            raise ValueError('nll_plan_threshold must be positive and finite')
        if nll_top_k is not None and nll_plan_threshold is not None:
            raise ValueError('Choose either top-K or raw-plan threshold NLL selection')
        if sigma_row_balance and (nll_top_k is not None or nll_plan_threshold is not None):
            raise ValueError('Sigma row balancing ablation requires the original full-row NLL')
        self.target,self.n,self.m,self.batch,self.epsilon=target,n,m,batch,epsilon
        self.temperature=float(temperature)
        self.nll_top_k=nll_top_k
        self.nll_plan_threshold=nll_plan_threshold
        self.sigma_row_balance=bool(sigma_row_balance)
        if distillation_loss not in {"conditional_ot_nll", "direct_gmm_nll"}:
            raise ValueError("Unknown distillation loss")
        if distillation_loss == "direct_gmm_nll" and (nll_top_k is not None or nll_plan_threshold is not None or sigma_row_balance):
            raise ValueError("OT row ablations cannot be used with Direct GMM")
        self.distillation_loss=distillation_loss
        self.gradient_axis=None
        self.sinkhorn_iterations=sinkhorn_iterations
        self.actor=SemiImplicitActor(2,tuple(hidden_dims),-5.,1.,math.log(.5),
                                     mean_output_init_scale=mean_output_init_scale)
        self.key,init=jax.random.split(jax.random.PRNGKey(seed))
        params=self.actor.init(init,jnp.zeros((1,1)),jnp.zeros((1,2)))["params"]
        self.state=TrainState.create(apply_fn=self.actor.apply,params=params,tx=optax.adam(3e-4))
        self.updates=0
        self.advance_fn=jax.jit(self._advance,static_argnums=2)
        self.sample_fn=jax.jit(self._sample,static_argnums=2)

    def _update(self,carry,_):
        state,key=carry
        key,zk,ek,pk=jax.random.split(key,4)
        obs=jnp.zeros((self.batch*self.n,1))
        z=jax.random.normal(zk,(self.batch*self.n,2))
        def loss(params):
            mu,ls=state.apply_fn({"params":params},obs,z)
            mu,ls=mu.reshape(self.batch,self.n,2),ls.reshape(self.batch,self.n,2)
            # Retain the sample-action noise draw used in v5, although OT uses means.
            sampled_u=mu+jnp.exp(ls)*jax.random.normal(ek,mu.shape)
            proposal=ConditionalGaussianProposal(jax.lax.stop_gradient(mu),jax.lax.stop_gradient(ls),.05)
            a,u,_=proposal.sample(pk,self.m//self.n,"exact")
            logq=proposal.log_prob(u)-2*math.log(40)
            q=self.target.jax_log_prob(40*a)
            q_score=q if self.temperature==1.0 else q/self.temperature
            w=jax.lax.stop_gradient(jax.nn.softmax(q_score-logq,axis=-1))
            balance_diagnostics={}
            if self.distillation_loss == "direct_gmm_nll":
                value,ell=direct_gmm_nll(mu,ls,u,w)
                usage=(jax.nn.softmax(ell,axis=1)*w[:,None,:]).sum(-1)
                balance_diagnostics.update(
                    gmm_component_ess_fraction=(1/jnp.square(usage).sum(-1)/self.n).mean(),
                    gmm_component_usage_min=usage.min(),
                    gmm_underused_fraction=(usage<.1/self.n).mean())
            else:
                cost=jnp.sum((jnp.tanh(mu)[:,:,None,:]-a[:,None,:,:])**2,axis=-1)
                plan=jax.lax.stop_gradient(sinkhorn(cost,w,self.epsilon,self.sinkhorn_iterations))
                row=plan/jnp.maximum(plan.sum(-1,keepdims=True),1e-20)
                if self.nll_top_k is not None:
                    full_row=row
                    indices=jax.lax.top_k(full_row,self.nll_top_k)[1]
                    keep=jnp.zeros_like(full_row).at[
                        jnp.arange(self.batch)[:,None,None],jnp.arange(self.n)[None,:,None],indices
                    ].set(1.0)
                    retained_mass=(full_row*keep).sum(-1,keepdims=True)
                    row=jax.lax.stop_gradient(full_row*keep/jnp.maximum(retained_mass,1e-20))
                elif self.nll_plan_threshold is not None:
                    full_row=row
                    row,keep,retained_mass,empty_rows,selected_counts=truncate_plan_rows(plan,self.nll_plan_threshold)
                balance_diagnostics={}
                if self.sigma_row_balance:
                    value,balance_diagnostics=sigma_row_balanced_ot_nll(mu,ls,u,row)
                else:
                    value=conditional_ot_nll(mu,ls,u,row)
            diagnostics=dict(loss=value,teacher_ess=(1/jnp.sum(w*w,axis=-1)).mean(),
                             sigma_mean=jnp.exp(ls).mean(),sigma_min=jnp.exp(ls).min(),sigma_max=jnp.exp(ls).max(),
                             mean_spread=jnp.var(jnp.tanh(mu),axis=1).mean(),
                             teacher_Q=q.mean(),weighted_teacher_Q=(w*q).sum(-1).mean(),
                             sampled_action_std=jnp.std(jnp.tanh(sampled_u)))
            if self.distillation_loss != "direct_gmm_nll":
                diagnostics.update(
                             ot_row_error=jnp.abs(plan.sum(-1)-1/self.n).max(),
                             ot_column_error=jnp.abs(plan.sum(-2)-w).max(),
                             row_entropy=-(row*jnp.log(jnp.maximum(row,1e-30))).sum(-1).mean(),
                )
            diagnostics.update(balance_diagnostics)
            if self.sigma_row_balance:
                diagnostics['teacher_floor_active_fraction']=(ls<math.log(.05)).astype(mu.dtype).mean()
            if self.gradient_axis is not None:
                action=jnp.tanh(sampled_u)
                first=jax.lax.pmean(action.mean(),self.gradient_axis)
                second=jax.lax.pmean((action*action).mean(),self.gradient_axis)
                diagnostics['sampled_action_std']=jnp.sqrt(jnp.maximum(second-first*first,0))
            if self.nll_top_k is not None or self.nll_plan_threshold is not None:
                # Aux-only attribution: small transport mass can still matter
                # through large squared residuals in the log-sigma gradient.
                error2=(mu[:,:,None,:]-u[:,None,:,:])**2
                standardized2=error2*jnp.exp(-2*ls)[:,:,None,:]
                tail=full_row*(1-keep)
                full_error=(full_row[...,None]*error2).sum(-2)
                kept_error=(row[...,None]*error2).sum(-2)
                expansion=jnp.maximum(standardized2-1,0)
                full_expansion=(full_row[...,None]*expansion).sum(-2).mean()
                diagnostics.update(
                    nll_retained_mass_mean=retained_mass.mean(),
                    nll_retained_mass_min=retained_mass.min(),
                    full_row_sigma2_target_mean=full_error.mean(),
                    nll_sigma2_target_mean=kept_error.mean(),
                    tail_squared_error_fraction=(tail[...,None]*error2).sum(-2).mean()/jnp.maximum(full_error.mean(),1e-20),
                    tail_sigma_expansion_pressure_fraction=(tail[...,None]*expansion).sum(-2).mean()/jnp.maximum(full_expansion,1e-20),
                    tail_log_sigma_gradient_mean=(tail[...,None]*(1-standardized2)).sum(-2).mean(),
                    full_log_sigma_gradient_mean=(1-full_error*jnp.exp(-2*ls)).mean(),
                    selected_log_sigma_gradient_mean=(1-kept_error*jnp.exp(-2*ls)).mean())
                if self.nll_top_k is not None:
                    diagnostics['nll_top_k']=jnp.asarray(self.nll_top_k,dtype=mu.dtype)
                else:
                    diagnostics.update(
                        nll_plan_threshold=jnp.asarray(self.nll_plan_threshold,dtype=mu.dtype),
                        nll_empty_row_fallback_fraction=empty_rows.astype(mu.dtype).mean(),
                        nll_selected_candidates_mean=selected_counts.astype(mu.dtype).mean(),
                        nll_effective_candidates_mean=keep.sum(-1).mean(),
                        teacher_floor_active_fraction=(ls<math.log(.05)).astype(mu.dtype).mean())
            return value,diagnostics
        (_,info),grad=jax.value_and_grad(loss,has_aux=True)(state.params)
        if self.gradient_axis is not None:
            grad=jax.lax.pmean(grad,self.gradient_axis)
            info={k:(jax.lax.pmax(v,self.gradient_axis) if k in ('sigma_max','ot_row_error','ot_column_error','sigma_row_weight_max')
                     else jax.lax.pmin(v,self.gradient_axis) if k in ('sigma_min','nll_retained_mass_min','sigma_row_weight_min')
                     else jax.lax.pmean(v,self.gradient_axis)) for k,v in info.items()}
        state=state.apply_gradients(grads=grad)
        return (state,key),info

    def _advance(self,state,key,count):
        (state,key),info=jax.lax.scan(self._update,(state,key),None,length=count)
        return state,key,jax.tree_util.tree_map(lambda x:x.mean(),info)

    def advance(self,count):
        self.state,self.key,info=self.advance_fn(self.state,self.key,count)
        info={k:float(v) for k,v in info.items()}
        self.updates+=count
        info["Q_evaluations"]=self.updates*self.batch*self.m
        return info

    def _sample(self,params,key,n):
        zkey,ekey=jax.random.split(key)
        z=jax.random.normal(zkey,(n,2)); eps=jax.random.normal(ekey,(n,2))
        mu,ls=self.actor.apply({"params":params},jnp.zeros((n,1)),z)
        x=40*jnp.tanh(mu+jnp.exp(ls)*eps)
        return x,40*jnp.tanh(mu),40*jnp.tanh(eps),jnp.exp(ls)

    def evaluate_samples(self,n,seed):
        x,means,base,sigma=self.sample_fn(self.state.params,jax.random.PRNGKey(seed),n)
        return np.asarray(x),np.stack([np.asarray(base[:128]),np.asarray(x[:128])]),{"mu_only":np.asarray(means)}

    def save(self,path):
        path.write_bytes(flax.serialization.to_bytes(dict(state=self.state,key=self.key,updates=self.updates)))

    def restore(self,path):
        saved=flax.serialization.from_bytes(dict(state=self.state,key=self.key,updates=0),path.read_bytes())
        self.state,self.key,self.updates=saved["state"],saved["key"],int(saved["updates"])
