"""Every policy density is a 32768-latent/32768-noise histogram, never a KDE."""
import functools
import numpy as np
import jax
import jax.numpy as jnp
from .models import actor_apply,q_mean,q_reference_mean,sample_action
from .problems import GRID,EDGES,analytic_q,reference,reward
from .config import raw_step
from .io import save

@functools.lru_cache(None)
def sampler(method):
    apply=actor_apply(method)
    @jax.jit
    def sample(params,obs,key):
        zk,ek=jax.random.split(key)
        z=jax.random.normal(zk,(32768,1));eps=jax.random.normal(ek,(32768,1))
        mu,ls=apply({'params':params},jnp.full_like(z,obs),z)
        return dict(actions=jnp.tanh(mu+jnp.exp(ls)*eps)[:,0],z=z[:,0],eps=eps[:,0],mu=mu[:,0],log_sigma=ls[:,0])
    return sample

def q_reference(f,qarg,qkind,obs=0.):
    if qkind=='analytic':q=analytic_q(GRID,np.asarray(qarg));params=qarg
    elif qkind=='replay':q=np.asarray(qarg);params=None
    else:q=np.asarray(q_reference_mean(qarg,jnp.array([[obs]],jnp.float32),jnp.asarray(GRID,jnp.float32)[None,:,None]))[0];params=None
    return reference(q,params)

def interpolation_check(params):
    fine=np.linspace(-1,1,16385,dtype=np.float32)
    q=np.asarray(q_reference_mean(params,jnp.zeros((1,1)),jnp.asarray(fine)[None,:,None]))[0].astype(float)
    default=np.asarray(q_mean(params,jnp.zeros((1,1)),jnp.asarray(fine)[None,:,None]))[0].astype(float)
    err=float(np.max(abs(q-np.interp(fine,fine[::2],q[::2]))))
    ref=reference(q[::2]);scaled=np.exp((q-q.max())/.25)
    areas=(scaled[1:]+scaled[:-1])/2*np.diff(fine);cdf=np.r_[0,np.cumsum(areas)];cdf/=cdf[-1]
    mass=np.diff(np.interp(EDGES,fine,cdf));tv=float(.5*np.abs(mass-ref['target_bin_mass']).sum())
    z=np.trapz(scaled,fine);backup=float(np.trapz(scaled*q,fine)/z)
    gap=abs(backup-ref['target_backup'])
    assert err<1e-4 and tv<1e-4 and gap<1e-4,(err,tv,gap)
    return dict(grid_interpolation_linf=err,grid_bin_tv=tv,grid_backup_gap=gap,
                reference_vs_training_precision_linf=float(abs(q-default).max()))

def evaluate(out,task,step,state,f,qarg,qkind,final=False,obs=0.):
    ref=q_reference(f,qarg,qkind,obs)
    # Fixed evaluation random numbers across time, disjoint from training.
    evalkey=jax.random.PRNGKey((970001 if final else 710001)+task['seed'])
    d=jax.device_get(sampler(task['method'])(state.params,float(obs),evalkey))
    a=d['actions'];counts,_=np.histogram(a,EDGES);mass=counts/32768
    bc,_=np.histogram(a,ref['boundaries']);bm=bc/32768
    assert counts.sum()==bc.sum()==32768
    reference_backup=ref['target_backup']
    sampled_q=np.asarray(f['qvalue'](jnp.array([[obs]],jnp.float32),jnp.asarray(a)[None,:,None],qarg))[0]
    backup=float(sampled_q.mean())
    metrics=dict(step=step,binned_tv=float(.5*abs(mass-ref['target_bin_mass']).sum()),
        basin_mass_tv=float(.5*abs(bm-ref['target_basin_mass']).sum()),backup=backup,
        target_backup=reference_backup,backup_bias=backup-reference_backup,
        sigma_mean=float(np.exp(d['log_sigma']).mean()),between_mu_variance=float(np.var(d['mu'])),
        within_variance=float(np.mean(np.exp(2*d['log_sigma']))),target_peaks=len(ref['peaks']))
    # A fresh independent proposal is evaluated at exactly the same probe state.
    t,_=f['teacher'](state.params,jnp.array([[obs]],jnp.float32),jax.random.PRNGKey(850001+task['seed']+step),qarg)
    b,w=np.asarray(t['b'])[0,:,0],np.asarray(t['w'])[0]
    pcount,_=np.histogram(b,EDGES);wm,_=np.histogram(b,EDGES,weights=w)
    tbm,_=np.histogram(b,ref['boundaries'],weights=w)
    assign=f['assignment'](state.params,t)
    metrics.update(teacher_binned_tv=float(.5*abs(wm-ref['target_bin_mass']).sum()),
        teacher_basin_mass_tv=float(.5*abs(tbm-ref['target_basin_mass']).sum()),
        actor_teacher_binned_tv=float(.5*abs(mass-wm).sum()),
        common_marginal_nll=float(assign['common_nll']),usage_ess=float(assign['usage_ess']),
        underused_fraction=float(assign['underused_fraction']),row_entropy=float(assign['row_entropy']),
        probe_ess=float(1/np.square(w).sum()),probe_wmax=float(w.max()))
    assert all(np.isfinite(v) for v in metrics.values()),metrics
    d.update(**ref,counts=counts,bin_mass=mass,density=mass/np.diff(EDGES),basin_mass=bm,
        proposal_actions=b,proposal_counts=pcount,teacher_weights=w,teacher_bin_mass=wm,
        teacher_basin_mass=tbm,step=np.array(step),probe_state=np.array(obs),evaluation_key=np.asarray(evalkey),
        sample_count=32768,**{k:np.asarray(v) for k,v in metrics.items() if k not in ('step','target_backup')})
    suffix='final_independent' if final else f'{step:06d}'
    save(out/'density'/f'{suffix}.npz',d)
    if raw_step(step) or final:
        raw={k:np.asarray(v) for k,v in t.items()}
        raw.update({k:np.asarray(v) for k,v in assign.items()})
        raw.update(step=step,probe_state=obs,source_order=np.argsort(np.asarray(t['positions'])[0,:,0],kind='stable'),
            candidate_order=np.argsort(b,kind='stable'),target_boundaries=ref['boundaries'])
        save(out/'assignments'/f'{suffix}.npz',raw)
    return metrics

@functools.lru_cache(None)
def return_engine():
    @jax.jit
    def run(actor,key):
        def step(carry,_):
            obs,key,total=carry;key,ak=jax.random.split(key)
            a=sample_action(actor,obs,ak,deterministic=False)
            g=jnp.exp(-.5*((a[...,0,None]-jnp.array([-.6,0,.6]))/.1)**2).sum(-1)
            r=g-.25*jnp.square(a[:,0]-obs[:,0])
            return (a,key,total+r),None
        (_,_,returns),_=jax.lax.scan(step,(jnp.zeros((10,1)),key,jnp.zeros(10)),None,length=200)
        return returns
    return run

def policy_returns(state,seed):
    r=np.asarray(return_engine()(state,jax.random.PRNGKey(112300+seed)))
    return dict(stochastic_return_mean=float(r.mean()),stochastic_return_sd=float(r.std(ddof=1)),returns=r)
