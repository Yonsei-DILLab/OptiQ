"""Faithful legacy GMM recipe and v5 conditional-mixture comparison adapters."""
import functools
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import optax
from flax.training.train_state import TrainState
from scipy.optimize import linear_sum_assignment
from benchmarks.gmm40.sampler import initialize as legacy_initialize, log_prob
from benchmarks.gmm40.latent_sampling import gaussian_grid
from optiq_dime.transport import GaussianKDE, sinkhorn
from .v5_actor import SemiImplicitActor
from .v5_distribution import ConditionalGaussianProposal, conditional_ot_nll


def cfg_for(condition,seed):
    legacy=condition['method'] in ('legacy','monge')
    return dict(condition,seed=seed,updates=75000,temperature=1.,density_beta=1.,
        hidden_dims=[512]*5 if legacy else [256,256],learning_rate=.0003,
        learning_rate_after_50k=.0001,latent_sampling='grid' if legacy else 'iid',
        matmul_precision='highest',batch_size=1,output_scale=1. if legacy else 50.,
        proposal='unbounded Gaussian KDE' if legacy else 'v5 conditional squashed Gaussian mixture',
        proposal_std_start=8.,proposal_std_final=1.,anneal_updates=15000,
        sinkhorn_epsilon_start=.01 if legacy else .1,sinkhorn_epsilon_final=.0001 if legacy else .1,
        sinkhorn_iterations=300 if legacy else 100,teacher_std_floor=.05,
        eval_samples=32768,eval_interval=5000)


def initialize(cfg):
    if cfg['method'] in ('legacy','monge'):
        actor,oracle,key=legacy_initialize(cfg['seed'],cfg['hidden_dims'],coordinate_scale=1.)
        target=oracle.params
    else:
        _,oracle,_=legacy_initialize(0,hidden_dims=(2,),coordinate_scale=1.)
        target=oracle.params
        model=SemiImplicitActor(2,tuple(cfg['hidden_dims']),-5.,1.,np.log(.5))
        key,ik=jax.random.split(jax.random.PRNGKey(cfg['seed']))
        params=model.init(ik,jnp.zeros((1,0)),jnp.zeros((1,2)))['params']
        actor=TrainState.create(apply_fn=model.apply,params=params,tx=optax.adam(.0003))
    # Same Adam history across LR boundary, coefficient is a deterministic update schedule.
    schedule=lambda k:jnp.where(k<50000,.0003,.0001)
    tx=optax.adam(schedule)
    actor=actor.replace(tx=tx,opt_state=tx.init(actor.params))
    return actor,key,target


def monge_indices(x,b,w,offset,order):
    """Exact 2D linear assignment to N systematic teacher representatives.

    This is Monge for the equal-weight empirical approximation, not the arbitrary
    original weighted M-atom measure. Randomize candidate order before systematic
    resampling so there is no artificial coordinate/mode order preference.
    """
    x,b,w=np.asarray(x,np.float64),np.asarray(b,np.float64),np.asarray(w,np.float64)
    order=np.asarray(order,np.int64);cdf=np.cumsum(w[order]/w.sum());cdf[-1]=1.
    selected=order[np.minimum(np.searchsorted(cdf,(np.arange(len(x))+float(offset))/len(x),side='right'),len(b)-1)]
    cost=((x[:,None,:]-b[selected][None,:,:])**2).sum(-1)
    rows,cols=linear_sum_assignment(cost)
    result=np.empty(len(x),np.int32);result[rows]=selected[cols]
    return result


@functools.lru_cache(None)
def engine(method,n,m):
    legacy=method in ('legacy','monge')
    def teacher(actor,key,target):
        key,lk,pk,_=jax.random.split(key,4)
        z=gaussian_grid(lk,n) if legacy else jax.random.normal(jax.random.split(lk)[0],(n,2))
        output=actor.apply_fn({'params':actor.params},jnp.zeros((n,0)),z)
        frac=jnp.minimum(actor.step/15000.,1.)
        if legacy:
            pos=output
            std=8.*jnp.power(1./8.,frac)
            proposal=GaussianKDE(pos[None],std)
            b=proposal.sample_stratified(pk,m//n,False).reshape(1,m,2)
            lq=proposal.log_prob(b)[0];b=b[0];u=b
            ep=.01*jnp.power(.0001/.01,frac)
        else:
            mu,ls=output;pos=jnp.tanh(mu)
            proposal=ConditionalGaussianProposal(mu[None],ls[None],.05)
            action,u,_=proposal.sample(pk,m//n,'exact');u=u[0];b=action[0]*50.
            lq=proposal.log_prob(u[None])[0]-2*jnp.log(50.);ep=jnp.array(.1)
        q=log_prob(b,target['locs'],target['scales'])
        w=jax.nn.softmax(q-lq)
        t=dict(z=z,b=b,u=u,w=w,q=q,log_q=lq,pos=pos,epsilon=ep)
        if method in ('legacy','v5_ot'):
            y=b if legacy else b/50.
            cost=jnp.square(pos[:,None,:]-y[None,:,:]).sum(-1)
            if legacy:cost=cost/(cost.mean()+1e-8)
            P=sinkhorn(cost[None],w[None],ep,300 if legacy else 100)[0]
            R=P/jnp.maximum(P.sum(-1,keepdims=True),1e-20)
            t.update(P=P,R=R)
            if legacy:t['selected']=b[jnp.argmax(R,axis=-1)]
        elif method=='monge':
            rk,ok=jax.random.split(jax.random.fold_in(pk,927))
            offset=jax.random.uniform(rk,());order=jax.random.permutation(ok,m)
            indices=jax.pure_callback(monge_indices,jax.ShapeDtypeStruct((n,),jnp.int32),
                jax.lax.stop_gradient(pos),jax.lax.stop_gradient(b),jax.lax.stop_gradient(w),offset,order)
            t['indices']=indices;t['selected']=b[indices]
        return jax.tree_util.tree_map(jax.lax.stop_gradient,t),key

    def loss(params,actor,t):
        out=actor.apply_fn({'params':params},jnp.zeros((n,0)),t['z'])
        if legacy:return jnp.square(out-t['selected']).sum(-1).mean()
        mu,ls=out
        if method=='v5_ot':return conditional_ot_nll(mu[None],ls[None],t['u'][None],t['R'][None])
        ell=component_logp(t['u'],mu,ls)
        return -(t['w']*(jsp.special.logsumexp(ell,axis=0)-jnp.log(n))).sum()

    def train_step(actor,key,target):
        t,key=teacher(actor,key,target)
        v,g=jax.value_and_grad(loss)(actor.params,actor,t)
        new=actor.apply_gradients(grads=g)
        return new,key,v
    @functools.partial(jax.jit,static_argnums=(3,))
    def block(actor,key,target,count):
        def body(_,carry):
            a,k,_=carry;return train_step(a,k,target)
        return jax.lax.fori_loop(0,count,body,(actor,key,jnp.array(0.,jnp.float32)))
    @jax.jit
    def diagnostics(actor,key,target):
        t,_=teacher(actor,key,target)
        out=actor.apply_fn({'params':actor.params},jnp.zeros((n,0)),t['z'])
        stats=dict(ess=1/jnp.square(t['w']).sum(),wmax=t['w'].max(),epsilon=t['epsilon'])
        if method in ('legacy','v5_ot'):
            A=t['R']/n
            stats.update(row_l1=jnp.abs(t['P'].sum(1)-1/n).sum(),column_l1=jnp.abs(t['P'].sum(0)-t['w']).sum(),
                         effective_column_l1=jnp.abs(A.sum(0)-t['w']).sum())
        elif method=='monge':
            A=jax.nn.one_hot(t['indices'],m)/n
            stats['quantization_atom_tv']=.5*jnp.abs(A.sum(0)-t['w']).sum()
        else:
            ell=component_logp(t['u'],*out)
            A=jax.nn.softmax(ell,axis=0)*t['w'][None]
        stats['usage_ess']=1/jnp.square(A.sum(-1)).sum()
        if not legacy:
            mu,ls=out;stats.update(sigma_mean=jnp.exp(ls).mean(),sigma_min=jnp.exp(ls).min(),sigma_max=jnp.exp(ls).max(),between_mu_variance=jnp.var(mu,axis=0).mean())
            t.update(mu=mu,log_sigma=ls)
        return dict(t,A=A),stats
    return dict(teacher=jax.jit(teacher),loss=loss,step=jax.jit(train_step),block=block,diagnostics=diagnostics)


def component_logp(u,mu,ls):
    return (-.5*jnp.square((u[None]-mu[:,None])*jnp.exp(-ls[:,None]))-ls[:,None]-.5*jnp.log(2*jnp.pi)).sum(-1)


@functools.partial(jax.jit,static_argnames=('count','method'))
def draw(actor,key,count,method):
    zk,ek=jax.random.split(key);z=jax.random.normal(zk,(count,2))
    out=actor.apply_fn({'params':actor.params},jnp.zeros((count,0)),z)
    if method in ('legacy','monge'):return out
    mu,ls=out;return 50*jnp.tanh(mu+jnp.exp(ls)*jax.random.normal(ek,mu.shape))
