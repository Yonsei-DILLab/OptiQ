"""Independent evaluation, actual sampled histograms, and finite-L score checks."""
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
from .core import CENTERS, WIDTH, ENERGY_TEMPERATURE

def target_logf(x):
    x=np.asarray(x)[...,None]
    return logsumexp(-.5*((x-np.array(CENTERS))/WIDTH)**2,axis=-1)-np.log(3*WIDTH*np.sqrt(2*np.pi))

Z=float(np.mean(ndtr((1-np.array(CENTERS))/WIDTH)-ndtr((-1-np.array(CENTERS))/WIDTH)))

def cdf(x):
    x=np.asarray(x)[...,None]
    return np.mean(ndtr((x-np.array(CENTERS))/WIDTH)-ndtr((-1-np.array(CENTERS))/WIDTH),axis=-1)/Z

REFERENCE_BACKUP=quad(lambda x: np.exp(target_logf(x))/Z*ENERGY_TEMPERATURE*target_logf(x),-1,1,epsabs=1e-11,points=list(CENTERS))[0]

def evaluate(experiment,folder,step):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    a,mu,sigma=experiment.samples(experiment.cfg['eval_samples'])
    edges=np.linspace(-1,1,experiment.cfg['histogram_bins']+1)
    mass=np.histogram(a,bins=edges)[0]/a.size;target=np.diff(cdf(edges))
    bounds=np.array([-1,-.3,.3,1.]);mode_mass=np.histogram(a,bins=bounds)[0]/a.size
    target_mode=np.diff(cdf(bounds));q=ENERGY_TEMPERATURE*target_logf(a.ravel())
    qhat=float(q.mean())
    # Target reference quantiles, no generated target samples supplied to training.
    grid=np.linspace(-1,1,32769);quantile=np.interp((np.arange(a.size)+.5)/a.size,cdf(grid),grid)
    metrics=dict(step=step,histogram_TV=float(.5*np.abs(mass-target).sum()),
        basin_TV=float(.5*np.abs(mode_mass-target_mode).sum()),mode_mass=mode_mass.tolist(),
        target_mode_mass=target_mode.tolist(),covered_modes=int((mode_mass>=.25*target_mode).sum()),
        wasserstein1=float(np.abs(np.sort(a.ravel())-quantile).mean()),
        backup=qhat,reference_backup=REFERENCE_BACKUP,backup_error=qhat-REFERENCE_BACKUP,
        backup_mc_se=float(q.std(ddof=1)/np.sqrt(q.size)),
        sigma_mean=float(sigma.mean()),sigma_min=float(sigma.min()),sigma_max=float(sigma.max()),
        between_mean_variance=float(mu.var()),gaussian_scale_squared_mean=float((sigma**2).mean()))
    np.savez_compressed(folder/f'samples_{step:05d}.npz',actions=a,mu=mu,sigma=sigma,
        edges=edges,histogram_mass=mass,target_mass=target)
    (folder/f'metrics_{step:05d}.json').write_text(json.dumps(metrics,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(8,3.8));x=np.linspace(-1,1,2001)
    ax.plot(x,np.exp(target_logf(x))/Z,'k--',label='Exact target')
    ax.stairs(mass/np.diff(edges),edges,label=f'{experiment.method} L={experiment.L}: 32,768 samples')
    ax.set(xlabel='Action',ylabel='Density',title=f'Update {step:,} | TV {metrics["histogram_TV"]:.3f}')
    ax.legend();fig.tight_layout();fig.savefig(folder/f'histogram_{step:05d}.png',dpi=150);plt.close(fig)
    return metrics

def score_diagnostic(experiment,folder,step):
    """Same checkpoint/actions, independent repeated banks; reference is also MC."""
    folder=Path(folder);count=experiment.cfg['diagnostic_actions']
    a=experiment.samples(count,seed=72011+step)[0]
    scores={};ess={};logs={}
    def calc(L,seed):
        values=experiment.score_fn(experiment.state.params,jnp.asarray(a),jax.random.PRNGKey(seed),L)
        return tuple(np.asarray(v) for v in values)
    refL=experiment.cfg['diagnostic_L_ref']
    ref,ref_log,_=calc(refL,81107+step)
    ref2,_,_=calc(2*refL,81107+step)
    metrics=dict(step=step,reference_L=refL,reference_doubling_RMSE=float(np.sqrt(np.mean((ref-ref2)**2))),L_results=[])
    for L in (128,256,1024,4096):
        values=[calc(L,100000+step+r) for r in range(experiment.cfg['diagnostic_repetitions'])]
        s=np.stack([v[0] for v in values]);e=np.stack([v[2] for v in values]);lp=np.stack([v[1] for v in values])
        scores[f'score_L{L}']=s;ess[f'ess_L{L}']=e;logs[f'logp_L{L}']=lp
        metrics['L_results'].append(dict(L=L,score_RMSE=float(np.sqrt(np.mean((s-ref)**2))),
            score_relative_RMSE=float(np.sqrt(np.mean((s-ref)**2))/(np.sqrt(np.mean(ref**2))+1e-8)),
            score_repeat_SD=float(np.std(s,axis=0).mean()),density_ESS_mean=float(e.mean()),
            log_density_RMSE=float(np.sqrt(np.mean((lp-ref_log)**2)))))
    np.savez_compressed(folder/f'score_{step:05d}.npz',actions=a,reference=ref,reference_double=ref2,**scores,**ess,**logs)
    (folder/f'score_{step:05d}.json').write_text(json.dumps(metrics,indent=2)+'\n')
    return metrics
