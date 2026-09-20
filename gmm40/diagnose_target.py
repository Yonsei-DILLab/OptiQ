"""Read-only distinction between GMM components and density local maxima."""
import numpy as np
from scipy.special import softmax
from scipy.spatial.distance import cdist
import json
from .target import Target,RESULTS
from .evaluation import atomic_json


def main():
    target=Target();mu=np.asarray(target.means,dtype=np.float64);sigma=np.asarray(target.std,dtype=np.float64)
    assert np.allclose(sigma,sigma[0]);variance=sigma[0]**2
    seed=8143
    starts=np.concatenate([mu,np.random.default_rng(seed).uniform(-40,40,size=(400,2))]);x=starts.copy()
    for iteration in range(10000):
        weights=softmax(-cdist(x,mu,'sqeuclidean')/(2*variance),axis=1);new=weights@mu
        shift=float(np.max(np.linalg.norm(new-x,axis=1)));x=new
        if shift<1e-10:break
    assert shift<1e-10
    centers=[];labels=[]
    for pos in x:
        matches=[i for i,c in enumerate(centers) if np.linalg.norm(c-pos)<1e-5]
        label=matches[0] if matches else len(centers)
        if not matches:centers.append(pos)
        labels.append(label)
    records=[]
    for i,pos in enumerate(centers):
        w=softmax(-np.sum((mu-pos)**2,axis=1)/(2*variance));d=mu-pos
        covariance=(d.T*w)@d-np.outer(w@d,w@d)
        eigenvalues=np.linalg.eigvalsh(covariance/(variance**2)-np.eye(2)/variance)
        records.append(dict(position=pos.tolist(),log_density=float(target.log_prob(pos)),hessian_eigenvalues=eigenvalues.tolist(),
                            is_strict_local_max=bool(np.all(eigenvalues<0)),source_components=[j for j in range(40) if labels[j]==i],starts_converging=labels.count(i)))
    result=dict(component_count=40,probe_starts=len(starts),seed=seed,iterations=iteration+1,final_max_shift=shift,
                maxima_found=sum(r['is_strict_local_max'] for r in records),stationary_points=records,
                method='Equal-covariance Gaussian mean shift from all 40 centers and 400 uniform starts, then analytic Hessian classification.',
                caveat='Numerical search, not proof that all stationary points were found. Primary coverage remains the 40-component-center proxy.')
    atomic_json(RESULTS/'diagnostics/target_stationary_points.json',result)
    peaks=np.array([r['position'] for r in records if r['is_strict_local_max']])
    reference=target.sample(10000,20260917,bounded=True)
    def counts(samples):
        distance=cdist(samples,peaks,'sqeuclidean')/variance;labels=distance.argmin(1)
        close=distance[np.arange(len(samples)),labels]<=9
        return np.bincount(labels[close],minlength=len(peaks))
    ref_counts=counts(reference);rows=[]
    queue=json.loads((RESULTS/'queue.json').read_text())['jobs']
    for job in queue:
        if '--navigation' in job.get('args',[]) or job['steps']!=100000:continue
        folder=RESULTS/job['name']
        if not (folder/'latest.json').exists():continue
        metrics=json.loads((folder/'latest.json').read_text());step=metrics['step']
        samples=np.load(folder/'evaluations'/f'step_{step:07d}'/'samples.npy')
        observed=counts(samples);threshold=np.maximum(10,.1*ref_counts*len(samples)/len(reference))
        rows.append(dict(name=job['name'],updates=step,primary_component_coverage=metrics['mode_coverage'],
                         numerical_peak_coverage=int((observed>=threshold).sum()),numerical_peaks_total=len(peaks),
                         counts=observed.tolist(),threshold=threshold.tolist()))
    atomic_json(RESULTS/'diagnostics/stationary_mode_coverage.json',dict(auxiliary_only=True,primary_40_component_metric_unchanged=True,results=rows))
    print(json.dumps(dict(maxima_found=result['maxima_found'],comparisons=[{k:v for k,v in r.items() if k not in ('counts','threshold')} for r in rows]),indent=2))


if __name__=='__main__':main()
