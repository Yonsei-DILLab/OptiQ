"""Forward NLL: centers(-5,0,5),target width1;unchanged TRG actor/proposal."""
import math
from functools import partial
import flax.serialization
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import optax
from flax.training.train_state import TrainState
from .actor import SemiImplicitActor
from .box_gaussian import sample_box, component_log_prob, mixture_log_prob
from .distillation import direct_gmm_nll

CENTERS = (-5., 0., 5.)
WIDTH = 1.
ENERGY_TEMPERATURE = .25

def log_f(a):
    x=a[..., 0, None]
    terms=-.5*((x-jnp.asarray(CENTERS))/WIDTH)**2-math.log(WIDTH*math.sqrt(2*math.pi))
    return jsp.special.logsumexp(terms,axis=-1)-math.log(3)

def q_value(a):
    return ENERGY_TEMPERATURE*log_f(a)

def target_score(a):
    terms=-.5*((a-jnp.asarray(CENTERS))/WIDTH)**2
    r=jax.nn.softmax(terms,axis=-1)
    return ((r*(jnp.asarray(CENTERS)-a)/WIDTH**2).sum(-1))[...,None]

def dense_score(a,mu,ls):
    ell=component_log_prob(a[None],mu[None],ls[None])[0]
    weights=jax.nn.softmax(ell,axis=0)
    score=(weights[...,None]*(-(a[None]-mu[:,None])*jnp.exp(-2*ls[:,None]))).sum(0)
    return score,jsp.special.logsumexp(ell,axis=0)-jnp.log(mu.shape[0])

class Experiment:
    def __init__(self,cfg,method,L,seed):
        assert method=='forward' and int(L)==0
        assert cfg['action_bound']==10 and cfg['target_centers']==list(CENTERS) and cfg['target_width']==WIDTH
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

    def density_score(self,params,a,key,L):
        """Exact chunk aggregation of a finite MC bank; no auxiliary theta path.

        Every chunk sees the SAME pre-update theta. Banks are prefix-coupled
        across L for diagnostics, but independent of source/action random draws.
        Truncation normalizers depend on mu/sigma, not on interior action a.
        """
        chunk=min(self.cfg['density_chunk'],L)
        assert L%chunk==0
        p=jax.tree_util.tree_map(jax.lax.stop_gradient,params)
        a=jax.lax.stop_gradient(a)
        def block(carry,index):
            logS,score,logS2=carry
            z=jax.random.normal(jax.random.fold_in(key,index),(chunk,1))
            mu,ls=self.components(p,z)
            ell=component_log_prob(a[None],mu[None],ls[None])[0]
            c=jsp.special.logsumexp(ell,axis=0)
            v=(jax.nn.softmax(ell,axis=0)[...,None]*(-(a[None]-mu[:,None])*jnp.exp(-2*ls[:,None]))).sum(0)
            combined=jnp.logaddexp(logS,c)
            score=jnp.exp(logS-combined)[:,None]*score+jnp.exp(c-combined)[:,None]*v
            return (combined,score,jnp.logaddexp(logS2,jsp.special.logsumexp(2*ell,axis=0))),None
        init=(jnp.full((a.shape[0],),-jnp.inf),jnp.zeros_like(a),jnp.full((a.shape[0],),-jnp.inf))
        (logS,score,logS2),_=jax.lax.scan(block,init,jnp.arange(L//chunk))
        return score,logS-jnp.log(L),jnp.exp(2*logS-logS2)

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
            q=q_value(a)
            w=jax.lax.stop_gradient(jax.nn.softmax(q/self.cfg['temperature']-logq))
            loss,ell=direct_gmm_nll(mu[None],ls[None],a[None],w[None])
            common.update(objective_estimate=loss,teacher_ess=1/(w*w).sum(),
                          teacher_wmax=w.max(),mean_Q=q.mean(),weighted_Q=(w*q).sum())
        else:
            a=sample_box(ek,mu[indices],ls[indices])
            score,logp,ess=self.density_score(params,a,dk,self.L)
            a_fixed=jax.lax.stop_gradient(a)
            # Exact oracle Q is differentiable; equivalent analytic action score.
            qscore=ENERGY_TEMPERATURE*target_score(a_fixed)
            direction=jax.lax.stop_gradient(score-qscore/self.cfg['temperature'])
            loss=(a*direction).sum(-1).mean()
            common.update(objective_estimate=(logp-q_value(a_fixed)/self.cfg['temperature']).mean(),
                          density_ess=ess.mean(),score_abs_mean=jnp.abs(score).mean(),mean_Q=q_value(a_fixed).mean())
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
