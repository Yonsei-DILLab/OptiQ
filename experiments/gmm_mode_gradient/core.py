"""Detached posterior gradient routing, applied at actor output level before VJP."""
from experiments.gmm_gradient_interference.core import *
METHODS=('baseline','mode_only','mode_confidence')

@jax.jit
def output_components(mu,ls,t):
    _,ell=direct_gmm_nll(mu[None],ls[None],t['u'],t['w'])
    ell=ell[0]  # N x M, pre-tanh component log density
    gamma=jax.nn.softmax(ell,axis=0)
    joint=gamma*t['w'][0][None,:]
    labels=jnp.digitize(t['b'][0,:,0],jnp.array([-.3,.3]))
    mode_mass=joint@jax.nn.one_hot(labels,3)
    alpha=mode_mass.sum(1)
    H=mode_mass/jnp.maximum(alpha[:,None],jnp.finfo(mu.dtype).tiny)
    mode=jnp.argmax(H,axis=1);confidence=H.max(1)
    return dict(ell=ell,joint=joint,H=H,alpha=alpha,mode=mode,confidence=confidence,labels=labels)

@jax.jit
def assignment(params,t):
    mu,ls=heads(params,t['z'][0]);d=output_components(mu,ls,t)
    return jax.tree_util.tree_map(jax.lax.stop_gradient,d)

@functools.partial(jax.jit,static_argnames=['method'])
def surrogate(params,t,coefficient,method):
    mu,ls=heads(params,t['z'][0])
    _,ell=direct_gmm_nll(mu[None],ls[None],t['u'],t['w'])
    # coefficient already contains teacher weight, posterior, gate and confidence.
    # No division by N, no mode-wise normalization, no gradient through coefficient.
    return -jnp.sum(jax.lax.stop_gradient(coefficient)*ell[0])

@functools.partial(jax.jit,static_argnames=['method'])
def gradients(params,t,method):
    d=assignment(params,t)
    gate=(d['mode'][:,None]==d['labels'][None,:])*(d['alpha'][:,None]>0)
    scale=d['confidence'] if method=='mode_confidence' else jnp.ones_like(d['confidence'])
    coeff=d['joint'] if method=='baseline' else d['joint']*gate*scale[:,None]
    g=jax.grad(surrogate)(params,t,coeff,method)
    return g,d,coeff

@functools.lru_cache(None)
def engine(n,m,method):
    assert method in METHODS
    f=base(n,m)
    @jax.jit
    def update(state,t):
        if method=='baseline':
            new,value,gn=f['update'](state,t)
            d=assignment(state.params,t)
            retained=jnp.array(1.)
        else:
            g,d,coeff=gradients(state.params,t,method)
            new=state.apply_gradients(grads=g)
            value=f['loss'](state.params,t)  # common marginal NLL, NOT surrogate value
            gn=jnp.sqrt(sum(jnp.square(x).sum() for x in jax.tree_util.tree_leaves(g)))
            retained=coeff.sum()
        metric=jnp.concatenate((jnp.array([value,gn,d['confidence'].mean(),retained,
            1/jnp.square(t['w']).sum(),t['w'].max()]),
            jnp.mean(jax.nn.one_hot(d['mode'],3),axis=0)))
        return new,metric
    @jax.jit
    def step(state,key):
        t,key=f['teacher'](state.params,OBS,key,QARG)
        new,metric=update(state,t)
        return new,key,metric
    @jax.jit
    def block(state,key,count):
        def body(_,carry):
            s,k,tot,last=carry
            ns,nk,metric=step(s,k)
            return ns,nk,tot+metric,metric
        ns,nk,total,last=jax.lax.fori_loop(0,count,body,(state,key,jnp.zeros(9),jnp.zeros(9)))
        return ns,nk,total/count,last
    return dict(update=update,step=step,block=block)
