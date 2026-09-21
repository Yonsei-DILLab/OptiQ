"""Float64 T=1 bounded quadrature; independent GMM Monte Carlo reference."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from analysis_boltzmann.problems import CASES, make_problem
from .core import write_json


def frozen_quadrature(problem, n, temperature=1.0):
    base = problem.base() if problem.separable else problem
    x = -1+(np.arange(n,dtype=np.float64)+.5)*2/n
    grid = x[:,None] if base.dim==1 else np.stack(np.meshgrid(x,x,indexing='ij'),-1).reshape(-1,2)
    q = base.q(grid)
    logits=q/temperature
    log_z=logsumexp(logits)+base.dim*np.log(2/n)
    mass=np.exp(logits-logsumexp(logits))
    factor=problem.dim if problem.separable else 1
    return grid,mass,float(mass@q)*factor,float(log_z)*factor


def reference_sample(problem, ref, rng, count):
    if problem.separable:
        return ref['grid'][rng.choice(len(ref['mass']),size=(count,problem.dim),p=ref['mass']),0]
    return ref['grid'][rng.choice(len(ref['mass']),size=count,p=ref['mass'])]


def gmm_logp(samples, centers, scales):
    output=[]
    for offset in range(0,len(samples),4096):
        delta=(samples[offset:offset+4096,None,:]-centers)/scales
        components=-.5*(delta**2).sum(-1)-np.log(scales).sum(-1)-np.log(2*np.pi)
        output.append(logsumexp(components,axis=-1)-np.log(len(centers)))
    return np.concatenate(output)


def gmm_labels(samples, centers):
    return np.concatenate([np.argmin(((samples[i:i+4096,None,:]-centers)**2).sum(-1),axis=-1)
                           for i in range(0,len(samples),4096)])


def prepare(campaign):
    root=Path(campaign)/'references'; root.mkdir(exist_ok=True)
    evidence=[]
    for case in CASES:
        problem=make_problem(case)
        old=frozen_quadrature(problem,256); consecutive=0
        trace=[]
        for n in (512,1024,2048,4096):
            current=frozen_quadrature(problem,n)
            delta=max(abs(current[2]-old[2]),abs(current[3]-old[3]))
            trace.append(dict(n=n,value=current[2],log_z=current[3],delta=delta))
            consecutive=consecutive+1 if delta<1e-5 else 0
            if consecutive>=2: break
            old=current
        if consecutive<2: raise RuntimeError('T=1 quadrature did not converge: '+case)
        grid,mass,value,log_z=current
        ref=dict(grid=grid,mass=mass,truth=np.array(value),log_z=np.array(log_z))
        rng=np.random.default_rng(20260911)
        samples=reference_sample(problem,ref,rng,100000)
        labels=problem.labels(samples)
        modes=len(problem.weights)**problem.dim if problem.separable else len(problem.weights)
        if problem.separable:
            p1=np.bincount(problem.base().labels(grid),weights=mass,minlength=len(problem.weights))
            ids=np.arange(modes); occupancy=np.ones(modes)
            for d in range(problem.dim): occupancy*=p1[(ids//(len(p1)**d))%len(p1)]
        else:
            occupancy=np.bincount(problem.labels(grid),weights=mass,minlength=modes)
        np.savez_compressed(root/(case+'.npz'),**ref,samples=samples,
                            mode_mass=occupancy,temperature=1.,resolution=n)
        row=dict(case=case,temperature=1.,truth=value,reference_kind='float64 midpoint quadrature',
                 domain='[-1,1]^d',trace=trace,convergence_tolerance=1e-5,consecutive_passes=consecutive)
        write_json(root/(case+'.json'),row); evidence.append(row)
        print('REFERENCE '+case+' '+str(value),flush=True)
    from benchmarks.gmm40.sampler import make_target
    target=make_target()
    centers=target.locs.numpy().astype(np.float64)
    scales=np.diagonal(target.scale_trils.numpy(),axis1=-2,axis2=-1).astype(np.float64)
    rng=np.random.default_rng(20260911)
    labels=rng.integers(0,40,size=1000000)
    samples=centers[labels]+scales[labels]*rng.normal(size=(len(labels),2))
    q=gmm_logp(samples,centers,scales)
    assignments=gmm_labels(samples,centers)
    ref=dict(centers=centers,scales=scales,samples=samples[:100000],truth=np.array(q.mean()),
        truth_se=np.array(q.std(ddof=1)/np.sqrt(len(q))),mode_mass=np.bincount(assignments,minlength=40)/len(q))
    np.savez_compressed(root/'gmm40.npz',**ref)
    row=dict(case='gmm40',temperature=1.,reference_kind='1M independent float64 GMM samples',
             reference_seed=20260911,truth=float(q.mean()),truth_se=float(ref['truth_se']),
             target_sha256=hashlib.sha256(centers.tobytes()+scales.tobytes()).hexdigest())
    write_json(root/'gmm40.json',row); evidence.append(row)
    write_json(root/'passed.json',dict(passed=True,cases=evidence))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    prepare(parser.parse_args().campaign)
