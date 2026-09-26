"""Reference metrics; ground-truth samples are evaluation-only."""
import numpy as np
from scipy.spatial.distance import cdist, jensenshannon

def mmd2(x, y, limit=2048):
    x,y = x[:limit].astype(np.float64), y[:limit].astype(np.float64)
    xx, yy, xy = cdist(x,x,"sqeuclidean"), cdist(y,y,"sqeuclidean"), cdist(x,y,"sqeuclidean")
    scores={}
    for h in (1.,2.,5.,10.,20.):
        kxx,kyy,kxy=np.exp(-xx/(2*h*h)),np.exp(-yy/(2*h*h)),np.exp(-xy/(2*h*h))
        scores[str(h)] = float((kxx.sum()-len(x))/(len(x)*(len(x)-1))+(kyy.sum()-len(y))/(len(y)*(len(y)-1))-2*kxy.mean())
    return float(np.mean(list(scores.values()))),scores


def assignments(x,target):
    dist=cdist(x,target.means,"sqeuclidean")/(target.std[None,:]**2)
    labels=dist.argmin(axis=1)
    near=dist[np.arange(len(x)),labels] <= 9.0
    counts=np.bincount(labels[near],minlength=40)
    return labels,near,counts


def metrics(x,target,reference,full_reference):
    assert np.isfinite(x).all() and x.shape[1]==2
    labels,near,counts=assignments(x,target)
    _,_,refcounts=assignments(reference,target)
    threshold=np.maximum(10,0.1*refcounts*len(x)/len(reference))
    covered=counts>=threshold
    probs=counts/max(counts.sum(),1)
    refprobs=refcounts/refcounts.sum()
    rng=np.random.default_rng(451)
    directions=rng.normal(size=(128,2)); directions/=np.linalg.norm(directions,axis=1,keepdims=True)
    sw2=np.sqrt(np.mean((np.sort(x@directions.T,axis=0)-np.sort(reference@directions.T,axis=0))**2))
    mmd,by_bandwidth=mmd2(x,reference)
    full_mmd,_=mmd2(x,full_reference)
    edges=np.linspace(-40,40,81)
    hx=np.histogram2d(*x.T,bins=(edges,edges))[0].ravel()+1e-8
    hy=np.histogram2d(*reference.T,bins=(edges,edges))[0].ravel()+1e-8
    return dict(mode_coverage=int(covered.sum()),coverage_reference='40 GMM component centers; not an exact count of density local maxima',covered_modes=np.flatnonzero(covered).tolist(),
                mode_counts_3sigma=counts.tolist(), coverage_threshold=threshold.tolist(),
                high_density_fraction=float(near.mean()), mode_mass_tv=float(0.5*np.abs(probs-refprobs).sum()),
                mode_mass_js=float(jensenshannon(probs+1e-12,refprobs+1e-12)**2),
                effective_modes=float(np.exp(-np.sum(probs*np.log(probs+1e-30)))),
                mmd2=mmd,mmd2_by_bandwidth=by_bandwidth,mmd2_full_unbounded_target=full_mmd,
                sliced_wasserstein2=float(sw2), histogram_js=float(jensenshannon(hx,hy)**2),
                mean_log_target=float(target.log_prob(x).mean()),
                boundary_fraction=float((np.abs(x).max(axis=1)>39.9).mean()),
                outside_fraction=float(np.any(np.abs(x)>40,axis=1).mean()),
                n_samples=len(x),mean=x.mean(0).tolist(),std=x.std(0).tolist())
