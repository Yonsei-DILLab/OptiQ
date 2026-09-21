"""Original 1D frozen landscapes, Direct GMM vs unscaled winning-mode gradients."""
from experiments.gmm_gradient_interference.core import *
from . import legacy_problems as landscape

METHODS=('baseline','mode_only')
BATCH_SIZE=128
MAX_PARALLEL_PAIRS=8388608


def microbatch_size(n,m,batch=BATCH_SIZE):
    return max(k for k in (1,2,4,8,16,32,64,128)
        if k<=batch and batch%k==0 and (k==1 or k*n*m<=MAX_PARALLEL_PAIRS))


@functools.partial(jax.jit,static_argnames=['batch'])
def batch_keys(key,batch=BATCH_SIZE):
    next_key,sample_key=jax.random.split(key)
    return next_key,jax.random.split(sample_key,batch)


@functools.lru_cache(None)
def functions(name,n,m):
    f=base(n,m)
    ref=landscape.reference(name)
    edges=jnp.asarray(ref['boundaries'][1:-1])
    k=len(ref['mode_mass'])
    @jax.jit
    def teacher(params,key):
        # Preserve the exact old actor proposal, candidate RNG, density correction.
        # Replace only the frozen Q and consequent teacher weights.
        t,next_key=f['teacher'](params,OBS,key,QARG)
        q=landscape.q_jax(t['b'],name);logits=q/.25-t['log_q']
        t=dict(t,Q=q,logits=logits,w=jax.nn.softmax(logits,axis=-1))
        return jax.tree_util.tree_map(jax.lax.stop_gradient,t),next_key
    @jax.jit
    def assignment(params,t):
        mu,ls=heads(params,t['z'][0]);_,ell=direct_gmm_nll(mu[None],ls[None],t['u'],t['w'])
        joint=jax.nn.softmax(ell[0],axis=0)*t['w'][0][None,:]
        labels=jnp.digitize(t['b'][0,:,0],edges)
        mass=joint@jax.nn.one_hot(labels,k);alpha=mass.sum(1)
        H=mass/jnp.maximum(alpha[:,None],jnp.finfo(mu.dtype).tiny)
        return jax.tree_util.tree_map(jax.lax.stop_gradient,dict(joint=joint,H=H,alpha=alpha,
            mode=jnp.argmax(H,axis=1),confidence=H.max(1),labels=labels))
    @jax.jit
    def surrogate(params,t,coeff):
        mu,ls=heads(params,t['z'][0]);_,ell=direct_gmm_nll(mu[None],ls[None],t['u'],t['w'])
        return -jnp.sum(jax.lax.stop_gradient(coeff)*ell[0])
    @functools.partial(jax.jit,static_argnames=['method'])
    def gradient(params,t,method):
        d=assignment(params,t)
        if method=='baseline':
            value,g=jax.value_and_grad(f['loss'])(params,t);coeff=d['joint']
        else:
            assert method=='mode_only'
            # No multiplication by confidence, no renormalization of retained mass.
            gate=(d['mode'][:,None]==d['labels'][None,:])*(d['alpha'][:,None]>0)
            coeff=d['joint']*gate
            g=jax.grad(surrogate)(params,t,coeff);value=f['loss'](params,t)
        return g,value,d,coeff
    @jax.jit
    def terms(params,t):
        mu,ls=heads(params,t['z'][0]);_,ell=direct_gmm_nll(mu[None],ls[None],t['u'],t['w'])
        logmix=jax.scipy.special.logsumexp(ell[0],axis=0)-jnp.log(n)
        label=jnp.digitize(t['b'][0,:,0],edges)
        return jax.nn.one_hot(label,k).T@(-t['w'][0]*logmix)
    return dict(teacher=teacher,assignment=assignment,gradient=gradient,terms=terms,
        surrogate=surrogate,loss=f['loss'],ref=ref,k=k)


@functools.lru_cache(None)
def group_fn(name,n,m,method):
    f=functions(name,n,m)
    @jax.jit
    def one(params,key):
        t,_=f['teacher'](params,key);g,value,d,coeff=f['gradient'](params,t,method)
        metric=jnp.concatenate((jnp.array([value,d['confidence'].mean(),coeff.sum(),
            1/jnp.square(t['w']).sum(),t['w'].max()]),
            jnp.mean(jax.nn.one_hot(d['mode'],f['k']),axis=0)))
        return g,metric
    return one


@functools.lru_cache(None)
def aggregate_fn(name,n,m,method,batch=BATCH_SIZE,micro=None):
    micro=micro or microbatch_size(n,m,batch);assert batch%micro==0
    one=group_fn(name,n,m,method);width=5+functions(name,n,m)['k']
    @jax.jit
    def aggregate(params,keys):
        assert keys.shape==(batch,2)
        zero=jax.tree_util.tree_map(jnp.zeros_like,params)
        def body(carry,ks):
            acc,met=carry;gs,ms=jax.vmap(one,in_axes=(None,0))(params,ks)
            acc=jax.tree_util.tree_map(lambda a,x:a+x.sum(0)/batch,acc,gs)
            return (acc,met+ms.sum(0)/batch),None
        return jax.lax.scan(body,(zero,jnp.zeros(width)),keys.reshape(batch//micro,micro,2))[0]
    return aggregate


@functools.lru_cache(None)
def engine(name,n,m,method):
    assert method in METHODS
    aggregate=aggregate_fn(name,n,m,method);width=6+functions(name,n,m)['k']
    @jax.jit
    def step(state,key):
        next_key,keys=batch_keys(key);g,metric=aggregate(state.params,keys)
        new=state.apply_gradients(grads=g)
        gn=jnp.sqrt(sum(jnp.square(x).sum() for x in jax.tree_util.tree_leaves(g)))
        return new,next_key,jnp.concatenate((metric[:1],gn[None],metric[1:]))
    @jax.jit
    def block(state,key,count):
        def body(_,carry):
            s,k,total,_=carry;ns,nk,metric=step(s,k)
            return ns,nk,total+metric,metric
        ns,nk,total,last=jax.lax.fori_loop(0,count,body,(state,key,jnp.zeros(width),jnp.zeros(width)))
        return ns,nk,total/count,last
    return dict(step=step,block=block)


@functools.partial(jax.jit,static_argnames=['name'])
def conditional_basin_prob(mu,ls,name):
    bounds=np.asarray(landscape.reference(name)['boundaries'])
    pre=np.r_[-np.inf,np.arctanh(bounds[1:-1]),np.inf]
    cdf=jax.scipy.special.ndtr((jnp.asarray(pre)[None,:]-mu)*jnp.exp(-ls))
    return jnp.diff(cdf,axis=1)
