"""Actual-action evaluation of three narrow modes in [-20,20]."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr, logsumexp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .core import CENTERS, WIDTH, ENERGY_TEMPERATURE, q_value
from .box_gaussian import sample_box, mixture_log_prob

BOUND=20.
CS=np.array(CENTERS)
Z=float(np.mean(ndtr((BOUND-CS)/WIDTH)-ndtr((-BOUND-CS)/WIDTH)))

def target_logf(x):
    return logsumexp(-.5*((np.asarray(x)[...,None]-CS)/WIDTH)**2,axis=-1)-np.log(3*WIDTH*np.sqrt(2*np.pi))

def cdf(x):
    return np.mean(ndtr((np.asarray(x)[...,None]-CS)/WIDTH)-ndtr((-BOUND-CS)/WIDTH),axis=-1)/Z

# Integration breakpoints bracket every narrow peak to avoid missing them.
POINTS=sorted([float(c+d*WIDTH) for c in CS for d in [-8,-4,0,4,8]])
REFERENCE_BACKUP=quad(lambda x: np.exp(target_logf(x))/Z*ENERGY_TEMPERATURE*target_logf(x),-BOUND,BOUND,epsabs=1e-11,points=POINTS,limit=300)[0]

def separation(a):
    x=np.asarray(a).ravel()
    core=np.array([np.mean((x>=c-WIDTH)&(x<c+WIDTH)) for c in CS])
    valley=np.array([np.mean((x>=c-WIDTH)&(x<c+WIDTH)) for c in [-5.,5.]])
    true=cdf(CS+WIDTH)-cdf(CS-WIDTH)
    ratios=valley/np.maximum(np.minimum(core[:-1],core[1:]),1/len(x))
    return dict(three_peak_pass=bool(np.all(core>=.5*true) and np.all(ratios<=.5)),core_mass=core.tolist(),target_core_mass=true.tolist(),valley_core_ratio=ratios.tolist())

def teacher_probe(exp,folder,step):
    """One independent diagnostic group; never feeds optimization or training RNG."""
    zk,ik,ek=jax.random.split(jax.random.PRNGKey(17431),3)
    z=jax.random.normal(zk,(exp.n,1));mu,ls=exp.components(exp.state.params,z)
    idx=jax.random.randint(ik,(exp.m,),0,exp.n)
    ls=jnp.maximum(ls,jnp.log(exp.cfg['teacher_std_floor']))
    a=sample_box(ek,mu[idx],ls[idx]);logq=mixture_log_prob(a[None],mu[None],ls[None])[0]
    q=q_value(a);w=jax.nn.softmax(q/exp.cfg['temperature']-logq)
    aa,ww=np.asarray(a).ravel(),np.asarray(w)
    np.savez_compressed(folder/f'teacher_{step:05d}.npz',actions=aa,weights=ww,Q=np.asarray(q),logq=np.asarray(logq),mu=np.asarray(mu),sigma=np.exp(np.asarray(ls)))
    return dict(teacher_probe_ess=float(1/np.sum(ww**2)),teacher_probe_wmax=float(ww.max()),teacher_probe_basin_mass=np.histogram(aa,[-20,-5,5,20],weights=ww)[0].tolist())

def evaluate(exp,folder,step):
    folder=Path(folder);a,mu,sigma=exp.samples(exp.cfg['eval_samples'])
    edges=np.linspace(-BOUND,BOUND,exp.cfg['histogram_bins']+1)
    mass=np.histogram(a,edges)[0]/a.size;target=np.diff(cdf(edges))
    bounds=np.array([-20.,-5.,5.,20.]);masses=np.histogram(a,bounds)[0]/a.size;tm=np.diff(cdf(bounds))
    q=ENERGY_TEMPERATURE*target_logf(a.ravel());grid=np.linspace(-BOUND,BOUND,262145)
    quantile=np.interp((np.arange(a.size)+.5)/a.size,cdf(grid),grid)
    metric=dict(step=step,histogram_TV=float(.5*np.abs(mass-target).sum()),basin_TV=float(.5*np.abs(masses-tm).sum()),mode_mass=masses.tolist(),target_mode_mass=tm.tolist(),covered_basins=int((masses>=.25*tm).sum()),wasserstein1=float(np.abs(np.sort(a.ravel())-quantile).mean()),backup=float(q.mean()),reference_backup=REFERENCE_BACKUP,backup_error=float(q.mean()-REFERENCE_BACKUP),backup_mc_se=float(q.std(ddof=1)/np.sqrt(q.size)),sigma_mean=float(sigma.mean()),sigma_min=float(sigma.min()),sigma_max=float(sigma.max()),between_mean_variance=float(mu.var()),gaussian_scale_squared_mean=float((sigma**2).mean()),**separation(a),**teacher_probe(exp,folder,step))
    np.savez_compressed(folder/f'samples_{step:05d}.npz',actions=a,mu=mu,sigma=sigma,edges=edges,histogram_mass=mass,target_mass=target)
    fig,axs=plt.subplots(1,4,figsize=(16,3.4))
    for ax,limits in zip(axs,[(-20,20),(-10.6,-9.4),(-.6,.6),(9.4,10.6)]):
        x=np.linspace(*limits,4001);ax.plot(x,np.exp(target_logf(x))/Z,'k--',label='Exact target')
        ax.stairs(mass/np.diff(edges),edges,lw=.8,label='32,768 action histogram');ax.set(xlim=limits,xlabel='Action',ylabel='Density');ax.grid(alpha=.15)
    axs[0].legend(fontsize=7);fig.suptitle(f'N=M={exp.n} | update {step:,} | TV={metric["histogram_TV"]:.3f} | separated peaks: {metric["three_peak_pass"]}')
    fig.tight_layout();fig.savefig(folder/f'histogram_{step:05d}.png',dpi=130);plt.close(fig)
    return metric
