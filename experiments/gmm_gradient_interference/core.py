from . import bootstrap
import functools
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
import importlib
PREFIX='analysis_tools.studies.20260917_nonstationary_q.experiment'
actor=importlib.import_module(PREFIX+'.actor')
models=importlib.import_module(PREFIX+'.models')
problems=importlib.import_module(PREFIX+'.problems')
from optiq_dime.distillation import direct_gmm_nll
OBS=jnp.zeros((1,1))
QARG=jnp.asarray(problems.schedule('prefix',1))
BOUNDARIES=jnp.array([-jnp.inf,jnp.arctanh(-.3),jnp.arctanh(.3),jnp.inf])

def initialize(seed):
    return models.actor_state('gmm_learned',seed),jax.random.PRNGKey(1000+seed)

def base(n,m):return actor.engine('gmm_learned',n,m,'analytic')

@functools.lru_cache(None)
def block(n,m):
    f=base(n,m)
    @jax.jit
    def run(state,key,count):
        def body(_,carry):
            s,k,_=carry
            new,k,_,loss,_=f['step'](s,OBS,k,QARG)
            return new,k,loss
        return jax.lax.fori_loop(0,count,body,(state,key,jnp.array(0.,jnp.float32)))
    return run

@jax.jit
def heads(params,z):
    return models.actor_apply('gmm_learned')({'params':params},jnp.zeros_like(z),z)

@jax.jit
def basin_prob(mu,ls):
    cdf=jax.scipy.special.ndtr((BOUNDARIES[None,:]-mu)*jnp.exp(-ls))
    return jnp.diff(cdf,axis=1)

@jax.jit
def mode_terms(params,t):
    t=jax.tree_util.tree_map(jax.lax.stop_gradient,t)
    mu,ls=heads(params,t['z'][0])
    _,ell=direct_gmm_nll(mu[None],ls[None],t['u'],t['w'])
    logmix=jax.scipy.special.logsumexp(ell[0],axis=0)-jnp.log(mu.shape[0])
    # Action NLL includes the parameter-independent tanh Jacobian.
    u=t['u'][0,:,0]
    logjac=2*(jnp.log(2.)-u-jax.nn.softplus(-2*u))
    which=jnp.digitize(t['b'][0,:,0],jnp.array([-.3,.3]))
    return jnp.sum(jax.nn.one_hot(which,3)*(-t['w'][0]*(logmix-logjac))[:,None],axis=0)

@jax.jit
def mode_grads(params,t):return jax.jacrev(mode_terms)(params,t)

def reference_teacher(z,count=1024):
    edges=np.linspace(-1,1,count+1);mid=(edges[:-1]+edges[1:])/2
    w=np.diff(problems.analytic_cdf(edges,problems.schedule('prefix',1)))
    return dict(z=z[None],u=jnp.asarray(np.arctanh(mid)[None,:,None],jnp.float32),
                b=jnp.asarray(mid[None,:,None],jnp.float32),w=jnp.asarray(w[None],jnp.float32))

@functools.partial(jax.jit,static_argnames=['count'])
def draw(params,key,count=32768):
    zk,ek=jax.random.split(key);z=jax.random.normal(zk,(count,1));mu,ls=heads(params,z)
    return jnp.tanh(mu+jnp.exp(ls)*jax.random.normal(ek,mu.shape))[:,0]

@jax.jit
def gram_tree(grads):
    leaves=jax.tree_util.tree_leaves(grads)
    return sum(x.reshape(3,-1)@x.reshape(3,-1).T for x in leaves)

@jax.jit
def responsibility(params,t):
    mu,ls=heads(params,t['z'][0]);_,ell=direct_gmm_nll(mu[None],ls[None],t['u'],t['w'])
    assignment=jax.nn.softmax(ell[0],axis=0)*t['w'][0][None]
    labels=jnp.digitize(t['b'][0,:,0],jnp.array([-.3,.3]))
    mass=assignment@jax.nn.one_hot(labels,3)
    alpha=mass.sum(-1)
    return mass,alpha,mass/jnp.maximum(alpha[:,None],1e-30)

