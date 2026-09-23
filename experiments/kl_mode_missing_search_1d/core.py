"""Parameter-search adapter: only target centers/width and mean init differ."""
import math
from functools import partial
import flax.serialization
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import optax
from flax.training.train_state import TrainState
from ..kl_forward_wide_1d.actor import SemiImplicitActor
from ..kl_forward_wide_1d.box_gaussian import sample_box, component_log_prob, mixture_log_prob
from ..kl_forward_wide_1d.distillation import direct_gmm_nll

def dense_score(a,mu,ls):
    ell=component_log_prob(a[None],mu[None],ls[None])[0]
    weights=jax.nn.softmax(ell,axis=0)
    score=(weights[...,None]*(-(a[None]-mu[:,None])*jnp.exp(-2*ls[:,None]))).sum(0)
    return score,jsp.special.logsumexp(ell,axis=0)-jnp.log(mu.shape[0])

class Experiment:
    def __init__(self,cfg,method,L,seed):
        assert method in ('forward','reverse')
        assert cfg['action_bound']==10
        assert (method=='forward' and L==0) or (method=='reverse' and L in (1024,1048576))
        self.centers=jnp.asarray(cfg['target_centers'])
        self.width=float(cfg['target_width'])
        self.cfg,self.method,self.L=cfg,method,int(L)
        self.n,self.m,self.batch=cfg['n'],cfg['m'],cfg['batch']
        self.actor=SemiImplicitActor(1,tuple(cfg['hidden_dims']),cfg['log_std_min'],
            cfg['log_std_max'],cfg['initial_log_std'],cfg['mean_output_init_scale'])
        self.key,init=jax.random.split(jax.random.PRNGKey(seed))
        params=self.actor.init(init,jnp.zeros((1,1)),jnp.zeros((1,1)))['params']
        self.state=TrainState.create(apply_fn=self.actor.apply,params=params,tx=optax.adam(cfg['learning_rate']))
        self.advance_fn=jax.jit(self._advance,static_argnums=2)
        self.sample_fn=jax.jit(self._sample,static_argnums=2)
        self.score_fn=jax.jit(self.density_score,static_argnums=3)

    def components(self,params,z):
        return self.actor.apply({'params':params},jnp.zeros_like(z),z)

    def log_f(self,a):
        x=a[...,0,None]
        terms=-.5*((x-self.centers)/self.width)**2-math.log(self.width*math.sqrt(2*math.pi))
        return jsp.special.logsumexp(terms,axis=-1)-math.log(3)

    def q_value(self,a):
        return self.cfg['temperature']*self.log_f(a)

    def target_score(self,a):
        r=jax.nn.softmax(-.5*((a-self.centers)/self.width)**2,axis=-1)
        return (r*(self.centers-a)/self.width**2).sum(-1)[...,None]

    def density_components(self, params, z):
        # Preserve default-precision source actions exactly as in forward; only
        # density-bank MLP evaluation needs consistent full-float32 dot products.
        with jax.default_matmul_precision('highest'):
            return self.components(params, z)

    def density_score(self, params, a, key, L):
        """Stream the SAME IID bank with max-scaled sums, avoiding log cancellation.

        Directly accumulating score numerators and denominators avoids losing
        convex weight normalization when log densities are very negative.
        This is the identical finite-bank ratio estimator, not another objective.
        """
        chunk=min(self.cfg['density_chunk'],L)
        assert L % chunk == 0
        params=jax.tree_util.tree_map(jax.lax.stop_gradient,params)
        a=jax.lax.stop_gradient(a)
        def block(carry,index):
            peak,total,numerator,squared=carry
            z=jax.random.normal(jax.random.fold_in(key,index),(chunk,1))
            mu,ls=self.density_components(params,z)
            ell=component_log_prob(a[None],mu[None],ls[None])[0]
            local_peak=jnp.max(ell,axis=0)
            mass=jnp.exp(ell-local_peak[None])
            local_score=(-(a[None]-mu[:,None])*jnp.exp(-2*ls[:,None]))
            new_peak=jnp.maximum(peak,local_peak)
            old_scale=jnp.exp(peak-new_peak)
            new_scale=jnp.exp(local_peak-new_peak)
            total=old_scale*total+new_scale*mass.sum(0)
            numerator=old_scale[:,None]*numerator+new_scale[:,None]*(mass[...,None]*local_score).sum(0)
            squared=old_scale**2*squared+new_scale**2*(mass**2).sum(0)
            return (new_peak,total,numerator,squared),None
        shape=(a.shape[0],)
        init=(jnp.full(shape,-jnp.inf),jnp.zeros(shape),jnp.zeros_like(a),jnp.zeros(shape))
        (peak,total,numerator,squared),_=jax.lax.scan(block,init,jnp.arange(L//chunk))
        return numerator/total[:,None],peak+jnp.log(total)-jnp.log(L),total**2/squared

    def group_loss(self,params,key):
        zk,ik,ek,dk=jax.random.split(key,4)
        z=jax.random.normal(zk,(self.n,1))
        mu,ls=self.components(params,z)
        indices=jax.random.randint(ik,(self.m,),0,self.n)
        common={'sigma_mean':jnp.exp(ls).mean(),'sigma_min':jnp.exp(ls).min(),
                'sigma_max':jnp.exp(ls).max(),'mean_variance':jnp.var(mu)}
        if self.method=='forward':
            pmu=jax.lax.stop_gradient(mu)
            pls=jnp.maximum(jax.lax.stop_gradient(ls),jnp.log(self.cfg['teacher_std_floor']))
            a=jax.lax.stop_gradient(sample_box(ek,pmu[indices],pls[indices]))
            logq=mixture_log_prob(a[None],pmu[None],pls[None])[0]
            q=self.q_value(a)
            w=jax.lax.stop_gradient(jax.nn.softmax(q/self.cfg['temperature']-logq))
            loss,ell=direct_gmm_nll(mu[None],ls[None],a[None],w[None])
            common.update(objective_estimate=loss,teacher_ess=1/(w*w).sum(),
                          teacher_wmax=w.max(),mean_Q=q.mean(),weighted_Q=(w*q).sum())
        else:
            a=sample_box(ek,mu[indices],ls[indices])
            score,logp,ess=self.density_score(params,a,dk,self.L)
            a_fixed=jax.lax.stop_gradient(a)
            # Exact oracle Q is differentiable; equivalent analytic action score.
            qscore=self.cfg['temperature']*self.target_score(a_fixed)
            direction=jax.lax.stop_gradient(score-qscore/self.cfg['temperature'])
            loss=(a*direction).sum(-1).mean()
            common.update(objective_estimate=(logp-self.q_value(a_fixed)/self.cfg['temperature']).mean(),
                          density_ess=ess.mean(),score_abs_mean=jnp.abs(score).mean(),mean_Q=self.q_value(a_fixed).mean())
        return loss,common

    def loss(self,params,keys):
        values,metrics=jax.vmap(self.group_loss,in_axes=(None,0))(params,keys)
        return values.mean(),jax.tree_util.tree_map(lambda x:x.mean(),metrics)

    def step(self,carry,_):
        state,key=carry;key,draw=jax.random.split(key)
        keys=jax.random.split(draw,self.batch)
        (_,info),grads=jax.value_and_grad(self.loss,has_aux=True)(state.params,keys)
        info['gradient_norm']=optax.global_norm(grads)
        return (state.apply_gradients(grads=grads),key),info

    def _advance(self,state,key,count):
        (state,key),info=jax.lax.scan(self.step,(state,key),None,length=count)
        return state,key,jax.tree_util.tree_map(lambda x:x.mean(),info)

    def advance(self,count):
        self.state,self.key,info=self.advance_fn(self.state,self.key,count)
        # Host conversion synchronizes every training block, including parameters.
        jax.block_until_ready(self.state)
        return {k:float(v) for k,v in info.items()}

    def _sample(self,params,key,count):
        zk,ek=jax.random.split(key);z=jax.random.normal(zk,(count,1))
        mu,ls=self.components(params,z)
        return sample_box(ek,mu,ls),mu,jnp.exp(ls)

    def samples(self,count=32768,seed=197):
        return tuple(np.asarray(x) for x in self.sample_fn(self.state.params,jax.random.PRNGKey(seed),count))

    def save(self,path):
        path.write_bytes(flax.serialization.to_bytes({'state':self.state,'key':self.key}))

    def restore(self,path):
        restored=flax.serialization.from_bytes({'state':self.state,'key':self.key},path.read_bytes())
        self.state,self.key=restored['state'],restored['key']
