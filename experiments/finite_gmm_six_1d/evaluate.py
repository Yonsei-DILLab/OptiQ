import json
import numpy as np
import jax
import jax.numpy as jnp
from scipy.special import ndtr, ndtri, softmax, logsumexp
from scipy.integrate import trapezoid
from .core import reference,quadrature,loss
from experiments.kl_diverse_targets_1d.target import geometry as gmm_geometry
from experiments.kl_nongmm_targets_1d.target import geometry as other_geometry


def components(p,cfg):
    mu=np.asarray(p[0]);sig=np.exp(p[1]) if cfg['variance']=='learned' else np.full_like(mu,cfg['fixed_sigma'])
    w=softmax(p[2]) if cfg['weights']=='learned' else np.ones_like(mu)/len(mu)
    return mu,sig,w


def density_cdf(p,cfg,x):
    mu,sig,w=components(p,cfg);z=(np.asarray(x)[...,None]-mu)/sig
    lower=0 if cfg.get('unbounded',False) else ndtr((-cfg['action_bound']-mu)/sig)
    norm=1 if cfg.get('unbounded',False) else ndtr((cfg['action_bound']-mu)/sig)-lower
    return np.sum(w*np.exp(-z*z/2)/(sig*np.sqrt(2*np.pi)*norm),axis=-1),np.sum(w*(ndtr(z)-lower)/norm,axis=-1)


def evaluate(p,cfg,folder,step,seed):
    mu,sig,w=components(p,cfg);n=cfg['final_eval_samples'] if step==cfg['steps'] else cfg['eval_samples']
    rng=np.random.default_rng(197+seed);ix=rng.choice(len(mu),size=n,p=w)
    if cfg.get('unbounded',False):a=mu[ix]+sig[ix]*rng.standard_normal(n)
    else:
        b=cfg['action_bound'];lo=ndtr((-b-mu[ix])/sig[ix]);hi=ndtr((b-mu[ix])/sig[ix]);u=lo+rng.random(n)*(hi-lo)
        a=mu[ix]+sig[ix]*ndtri(np.clip(u,np.finfo(float).eps,1-np.finfo(float).eps))
    b=cfg['action_bound'];edges=np.linspace(-b,b,cfg['histogram_bins']+1)
    mass=np.histogram(a,edges)[0]/n;target=np.diff(reference(cfg,edges)[1]);student_exact=np.diff(density_cdf(p,cfg,edges)[1])
    g=(other_geometry if cfg['target_kind']=='nongmm' else gmm_geometry)(cfg)
    basin=np.histogram(a,g['basin_bounds'])[0]/n
    window=lambda lo,hi:np.array([np.mean((a>=l)&(a<r)) for l,r in zip(lo,hi)])
    core=window(g['core_left'],g['core_right']);valley=window(g['valley_left'],g['valley_right'])
    cr=core/g['target_core'];br=basin/g['target_basin'];vr=valley/np.maximum(np.minimum(core[:-1],core[1:]),1/n)
    missing=(cr<.25)&(br<.25);peak_ok=bool((cr>=.5).all() and (br>=.5).all() and (vr<=np.maximum(.5,2*g['target_valley_core_ratio'])).all())
    x=np.linspace(-cfg.get('integration_bound',b),cfg.get('integration_bound',b),65537)
    pdf,cdf=density_cdf(p,cfg,x);tpdf,tcdf=reference(cfg,x)
    tv=float(.5*np.abs(mass-target).sum())
    m=dict(step=step,seed=seed,sample_count=n,histogram_TV=tv,analytic_binned_TV=float(.5*np.abs(student_exact-target).sum()),
           integrated_TV=float(.5*trapezoid(np.abs(pdf-tpdf),x)),wasserstein_1=float(trapezoid(np.abs(cdf-tcdf),x)),
           mode_mass=basin.tolist(),target_mode_mass=g['target_basin'].tolist(),core_mass_ratio=cr.tolist(),basin_mass_ratio=br.tolist(),
           missing_modes=int(missing.sum()),missing_mask=missing.tolist(),all_peak_pass=peak_ok,forward_fitting_pass=bool(peak_ok and tv<=.15),
           means=mu.tolist(),sigmas=sig.tolist(),weights=w.tolist(),sigma_bound_count=int(((sig<=cfg['sigma_min']*1.00001)|(sig>=cfg['sigma_max']*.99999)).sum()))
    if step==cfg['steps']:
        xq,mq=quadrature(cfg,cfg['validation_quad_points']);qx=jnp.asarray(xq);qm=jnp.asarray(mq)
        val,grad=jax.value_and_grad(lambda p:loss(p,qx,qm,cfg))(jnp.asarray(p))
        xl,ml=quadrature(cfg,cfg['quad_points']);vl,gl=jax.value_and_grad(lambda p:loss(p,jnp.asarray(xl),jnp.asarray(ml),cfg))(jnp.asarray(p))
        entropy=-np.sum(mq*np.log(np.maximum(reference(cfg,xq)[0],1e-300)))
        m.update(population_nll=float(val),forward_KL=float(val-entropy),gradient_norm=float(jnp.linalg.norm(grad)),
                 refined_loss_difference=float(abs(val-vl)),refined_gradient_max_difference=float(jnp.max(jnp.abs(grad-gl))))
        # Assignment diagnostic is evaluation only: integrate p*(a) responsibility inside true density basins.
        z=(xq[:,None]-mu)/sig;ell=-z*z/2-np.log(sig)+np.log(w)
        if not cfg.get('unbounded',False):ell-=np.log(ndtr((b-mu)/sig)-ndtr((-b-mu)/sig))
        resp=softmax(ell,axis=-1);ids=np.digitize(xq,g['basin_bounds'][1:-1])
        H=np.stack([np.sum(mq[ids==i,None]*resp[ids==i],axis=0) for i in range(len(g['peaks']))])
        np.savez_compressed(folder/'assignment_final.npz',mode_component_mass=H,row_normalized=H/np.maximum(H.sum(0,keepdims=True),1e-300),means=mu,sigmas=sig,weights=w)
        if cfg['stage']=='paper_control':
            fn=lambda means:loss(jnp.asarray(p).at[0].set(means),qx,qm,cfg)
            hess=np.asarray(jax.hessian(fn)(jnp.asarray(mu)));eig=np.linalg.eigvalsh(hess)
            rng2=np.random.default_rng(7000+seed);directions=rng2.normal(size=(64,len(mu)));directions/=np.linalg.norm(directions,axis=1,keepdims=True)
            bumps={str(radius):float(np.min([float(fn(jnp.asarray(mu+radius*d)))-float(val) for d in directions])) for radius in [.001,.01,.1]}
            m.update(hessian_eigenvalues=eig.tolist(),local_perturbation_min_loss_change=bumps,
                     stationarity_note='Finite numerical necessary conditions only; not a proof of a local minimum.')
    np.savez_compressed(folder/f'histogram_{step:06d}.npz',edges=edges,histogram_mass=mass,target_mass=target,analytic_student_mass=student_exact,means=mu,sigmas=sig,weights=w)
    if step==cfg['steps']:np.savez_compressed(folder/'final_actions.npz',actions=a.astype(np.float32))
    (folder/f'metrics_{step:06d}.json').write_text(json.dumps(m,indent=2)+'\n')
    return m
