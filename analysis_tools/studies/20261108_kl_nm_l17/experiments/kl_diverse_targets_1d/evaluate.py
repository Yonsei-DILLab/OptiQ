import numpy as np
from .target import reference,geometry


def evaluate(exp,folder,step):
    cfg=exp.cfg;b=cfg['action_bound'];g=geometry(cfg)
    # Same fixed evaluation RNG for paired methods; independent of training RNG.
    a,mu,sigma=exp.samples(cfg['final_eval_samples'] if step==cfg['steps'] else cfg['eval_samples'])
    a=a.ravel();edges=np.linspace(-b,b,cfg['histogram_bins']+1)
    mass=np.histogram(a,edges)[0]/len(a);target=np.diff(reference(cfg,edges)[1])
    basin=np.histogram(a,g['basin_bounds'])[0]/len(a)
    window=lambda lo,hi:np.array([np.mean((a>=l)&(a<r)) for l,r in zip(lo,hi)])
    core=window(g['core_left'],g['core_right']);valley=window(g['valley_left'],g['valley_right'])
    cr=core/g['target_core'];br=basin/g['target_basin']
    vr=valley/np.maximum(np.minimum(core[:-1],core[1:]),1/len(a))
    missing=(cr<.25)&(br<.25)
    peak_ok=bool((cr>=.5).all() and (br>=.5).all() and (vr<=np.maximum(.5,2*g['target_valley_core_ratio'])).all())
    tv=float(.5*np.abs(mass-target).sum())
    metric=dict(step=step,sample_count=len(a),histogram_TV=tv,mode_mass=basin.tolist(),target_mode_mass=g['target_basin'].tolist(),
        core_mass=core.tolist(),target_core_mass=g['target_core'].tolist(),core_mass_ratio=cr.tolist(),basin_mass_ratio=br.tolist(),
        valley_core_ratio=vr.tolist(),target_valley_core_ratio=g['target_valley_core_ratio'].tolist(),
        missing_modes=int(missing.sum()),missing_mask=missing.tolist(),all_peak_pass=peak_ok,
        forward_fitting_pass=bool(peak_ok and tv<=.15),mode_count=len(core),
        sigma_mean=float(sigma.mean()),sigma_min=float(sigma.min()),sigma_max=float(sigma.max()))
    np.savez_compressed(folder/f'samples_{step:06d}.npz',actions=a,mu=mu,sigma=sigma,edges=edges,histogram_mass=mass,target_mass=target,**g)
    return metric
