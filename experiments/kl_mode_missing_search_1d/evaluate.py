"""Sample histograms (no smoothing); per-mode cores disambiguate broad coverage."""
import json
from pathlib import Path
import numpy as np
from scipy.special import ndtr,logsumexp


def reference(cfg,x):
    cs=np.asarray(cfg['target_centers']);h=cfg['target_width'];b=cfg['action_bound']
    Z=np.mean(ndtr((b-cs)/h)-ndtr((-b-cs)/h))
    lf=logsumexp(-.5*((np.asarray(x)[...,None]-cs)/h)**2,axis=-1)-np.log(3*h*np.sqrt(2*np.pi))
    cdf=np.mean(ndtr((np.asarray(x)[...,None]-cs)/h)-ndtr((-b-cs)/h),axis=-1)/Z
    return np.exp(lf)/Z,cdf


def evaluate(exp,folder,step):
    cfg=exp.cfg;cs=np.asarray(cfg['target_centers']);h=cfg['target_width'];b=cfg['action_bound']
    a,mu,sigma=exp.samples(cfg['eval_samples']);a=a.ravel()
    edges=np.linspace(-b,b,cfg['histogram_bins']+1);mass=np.histogram(a,edges)[0]/a.size
    target=np.diff(reference(cfg,edges)[1]);bounds=np.r_[-b,(cs[:-1]+cs[1:])/2,b]
    tm=np.diff(reference(cfg,bounds)[1]);mm=np.histogram(a,bounds)[0]/a.size
    tc=reference(cfg,cs+h)[1]-reference(cfg,cs-h)[1]
    core=np.array([np.mean((a>=c-h)&(a<c+h)) for c in cs])
    mids=(cs[:-1]+cs[1:])/2
    valley=np.array([np.mean((a>=c-h)&(a<c+h)) for c in mids])
    ratios=valley/np.maximum(np.minimum(core[:-1],core[1:]),1/a.size)
    # Missing means little probability both inside the peak and in its basin.
    missing=(core < .25*tc)&(mm < .25*tm)
    peak_ok=(core>=.5*tc).all() and (ratios<=.5).all()
    metric=dict(step=step,histogram_TV=float(.5*np.abs(mass-target).sum()),mode_mass=mm.tolist(),target_mode_mass=tm.tolist(),core_mass=core.tolist(),target_core_mass=tc.tolist(),valley_core_ratio=ratios.tolist(),missing_modes=int(missing.sum()),missing_mask=missing.tolist(),three_peak_pass=bool(peak_ok),sigma_mean=float(sigma.mean()),sigma_min=float(sigma.min()),sigma_max=float(sigma.max()),between_mean_variance=float(mu.var()))
    np.savez_compressed(Path(folder)/f'samples_{step:06d}.npz',actions=a,mu=mu,sigma=sigma,edges=edges,histogram_mass=mass,target_mass=target)
    return metric
