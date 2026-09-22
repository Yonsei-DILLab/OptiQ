"""Learners receive an energy-only facade, never evaluation target samples."""
import math
import numpy as np

class EnergyOnly:
    __slots__=('jax_log_prob','torch_log_prob')
    def __init__(self,target):
        self.jax_log_prob=target.jax_log_prob
        self.torch_log_prob=target.torch_log_prob


def make_agent(cfg,oracle,seed):
    method=cfg['method']; b=cfg['batch']
    if method=='optiq_trg':
        from gmm40.optiq_trg import OptiQTRG
        return OptiQTRG(oracle,seed,cfg['n'],cfg['m'],b)
    if method in ('optiq','v5_gmm'):
        from gmm40.optiq import OptiQ
        if method=='optiq':return OptiQ(oracle,seed,cfg['n'],cfg['m'],b)
        return SquashedGMM(oracle,seed,cfg['n'],cfg['m'],b)
    if method=='sql':
        from gmm40.sql import SQL
        return SQL(oracle,seed,b)
    if method=='mfpo':
        from gmm40.mfpo import MFPO
        return MFPO(oracle,seed,b)
    if method in ('legacy','monge'):return Legacy(oracle,seed,cfg['n'],cfg['m'],method)
    from gmm40.torch_agents import make_agent as make_torch
    return make_torch(method,oracle,seed,b)

# Lazy numerical imports: evaluation-only commands do not initialize a GPU.
def SquashedGMM(oracle,seed,n,m,batch):
    import jax
    import jax.numpy as jnp
    import jax.scipy as jsp
    from gmm40.optiq import OptiQ,ConditionalGaussianProposal
    from gmm40.target import SCALE
    class Direct(OptiQ):
        def _update(self,carry,_):
            state,key=carry
            key,zk,ek,pk=jax.random.split(key,4)  # same v5 random streams
            obs=jnp.zeros((self.batch*self.n,1)); z=jax.random.normal(zk,(self.batch*self.n,2))
            def loss(params):
                mu,ls=state.apply_fn({'params':params},obs,z)
                mu=mu.reshape(self.batch,self.n,2);ls=ls.reshape(self.batch,self.n,2)
                proposal=ConditionalGaussianProposal(jax.lax.stop_gradient(mu),jax.lax.stop_gradient(ls),.05)
                a,u,_=proposal.sample(pk,self.m//self.n,'exact')
                a,u=jax.lax.stop_gradient(a),jax.lax.stop_gradient(u)
                q=self.target.jax_log_prob(SCALE*a)
                w=jax.lax.stop_gradient(jax.nn.softmax(q-proposal.log_prob(u),axis=-1))
                ell=(-.5*((u[:,None]-mu[:,:,None])*jnp.exp(-ls[:,:,None]))**2-ls[:,:,None]-.5*jnp.log(2*jnp.pi)).sum(-1)
                # Jacobian independent of actor parameters at stopped teacher u.
                jac=(2*(math.log(2)-u-jax.nn.softplus(-2*u))+math.log(SCALE)).sum(-1)
                value=-(w*(jsp.special.logsumexp(ell,axis=1)-math.log(self.n)-jac)).sum(-1).mean()
                return value,dict(loss=value,teacher_ess=(1/(w*w).sum(-1)).mean(),sigma_mean=jnp.exp(ls).mean())
            (_,info),grads=jax.value_and_grad(loss,has_aux=True)(state.params)
            return (state.apply_gradients(grads=grads),key),info
    return Direct(oracle,seed,n,m,batch)


def Legacy(oracle,seed,n,m,method):
    import jax
    import jax.numpy as jnp
    import flax.serialization
    import optax
    from flax.training.train_state import TrainState
    from .legacy_actor import ImplicitActor
    from .legacy_transport import GaussianKDE,sinkhorn
    from benchmarks.gmm40.latent_sampling import gaussian_grid
    from .monge import monge_indices
    from gmm40.optiq import OptiQ
    class Native(OptiQ):
        def __init__(self):
            self.n,self.m,self.batch,self.updates=n,m,1,0
            self.actor=ImplicitActor(2,(512,)*5)
            self.key,ik=jax.random.split(jax.random.PRNGKey(seed))
            params=self.actor.init(ik,jnp.zeros((1,0)),jnp.zeros((1,2)))['params']
            self.state=TrainState.create(apply_fn=self.actor.apply,params=params,
                tx=optax.adam(lambda k:jnp.where(k<50000,3e-4,1e-4)))
            self.advance_fn=jax.jit(self._advance,static_argnums=2)
            self.sample_fn=jax.jit(self._sample,static_argnums=2)
        def _update(self,carry,_):
            state,key=carry;key,zk,pk,rk=jax.random.split(key,4)
            z=gaussian_grid(zk,n);obs=jnp.zeros((n,0))
            pos=state.apply_fn({'params':state.params},obs,z)
            frac=jnp.minimum(state.step/15000.,1.)
            proposal=GaussianKDE(pos[None],8*jnp.power(1/8,frac))
            b=proposal.sample_stratified(pk,m//n,False).reshape(1,m,2)
            logq=proposal.log_prob(b)[0];b=b[0]
            q=oracle.jax_log_prob(b);w=jax.nn.softmax(q-logq)
            if method=='legacy':
                cost=((pos[:,None]-b[None])**2).sum(-1);cost/=cost.mean()+1e-8
                epsilon=.01*jnp.power(.0001/.01,frac)
                plan=sinkhorn(cost[None],w[None],epsilon,300)[0]
                selected=b[jnp.argmax(plan,axis=-1)]
            else:
                ok,sk=jax.random.split(rk)
                indices=jax.pure_callback(monge_indices,jax.ShapeDtypeStruct((n,),jnp.int32),pos,b,w,
                    jax.random.uniform(ok,()),jax.random.permutation(sk,m))
                selected=b[indices]
            selected=jax.lax.stop_gradient(selected)
            loss=lambda p:((state.apply_fn({'params':p},obs,z)-selected)**2).sum(-1).mean()
            value,grads=jax.value_and_grad(loss)(state.params)
            return (state.apply_gradients(grads=grads),key),dict(loss=value,teacher_ess=1/(w*w).sum(),teacher_wmax=w.max())
        def _sample(self,params,key,count):
            z=jax.random.normal(key,(count,2))
            x=self.actor.apply({'params':params},jnp.zeros((count,0)),z)
            return x,x,z,jnp.zeros_like(x)
    return Native()
