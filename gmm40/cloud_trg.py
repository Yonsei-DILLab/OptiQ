"""Full-batch TRG objective with sequential cloud gradient accumulation.

Preserves full-shape latent/proposal random draws, beta, target and one Adam
update per batch. Only floating-point reduction order differs from the dense
implementation. Used for N=M>=1024; no truncation or candidate subsampling.
"""
import jax
import jax.numpy as jnp
from .optiq_trg import OptiQTRG, Proposal, direct_gmm_nll


class CloudTRG(OptiQTRG):
    def _update(self, carry, _):
        state,key=carry
        key,zk,pk=jax.random.split(key,3)
        z=jax.random.normal(zk,(self.batch*self.n,2))
        obs=jnp.zeros((self.batch*self.n,1))
        mu,ls=state.apply_fn({'params':state.params},obs,z)
        mu,ls=mu.reshape(self.batch,self.n,2),ls.reshape(self.batch,self.n,2)
        proposal=Proposal(jax.lax.stop_gradient(mu),jax.lax.stop_gradient(ls),self.teacher_std_floor)
        actions,_,_=proposal.sample(pk,self.m//self.n,'exact')
        actions=jax.lax.stop_gradient(actions)

        def cloud_loss(params, zc, ac, mc, lc):
            cm,cl=state.apply_fn({'params':params},jnp.zeros((self.n,1)),zc)
            cm,cl=cm[None],cl[None]
            p=Proposal(mc[None],lc[None],self.teacher_std_floor)
            q=self.target.jax_log_prob(40*ac[None])
            w=jax.lax.stop_gradient(jax.nn.softmax(q/self.temperature-p.log_prob(ac[None]),axis=-1))
            value,ell=direct_gmm_nll(cm,cl,ac[None],w)
            usage=(jax.nn.softmax(ell,axis=1)*w[:,None,:]).sum(-1)
            info=jnp.stack([value,(1/(w*w).sum(-1)).mean(),jnp.exp(cl).mean(),
                            q.mean(),(w*q).sum(-1).mean(),jnp.var(cm,axis=1).mean(),
                            (1/(usage*usage).sum(-1)).mean()])
            return value,info
        def accumulate(carry,inputs):
            gs,infos=carry
            (_,info),grad=jax.value_and_grad(cloud_loss,has_aux=True)(state.params,*inputs)
            return (jax.tree_util.tree_map(jnp.add,gs,grad),infos+info),None
        (grads,infos),_=jax.lax.scan(accumulate,
            (jax.tree_util.tree_map(jnp.zeros_like,state.params),jnp.zeros(7)),
            (z.reshape(self.batch,self.n,2),actions,jax.lax.stop_gradient(mu),jax.lax.stop_gradient(ls)))
        grads=jax.tree_util.tree_map(lambda x:x/self.batch,grads)
        names=['loss','teacher_ess','sigma_mean','teacher_Q','weighted_teacher_Q','mean_spread','component_usage_ess']
        info=dict(zip(names,infos/self.batch))
        info.update(sigma_min=jnp.exp(ls).min(),sigma_max=jnp.exp(ls).max())
        return (state.apply_gradients(grads=grads),key),info
