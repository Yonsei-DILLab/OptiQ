"""Same-input verification run; saves failures and never clips persistence."""
import argparse
import hashlib
import itertools
import json
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from legacy_diagnose import legacy
from verified_solver import Mixture,ascend,catalog,assign,filtration


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(a):
    t0=time.monotonic();events=[]
    def report(stage,**kw):
        row=dict(stage=stage,seconds=time.monotonic()-t0,**kw);events.append(row)
        print(json.dumps(row),flush=True)
        (a.out/'progress.json').write_text(json.dumps(events,indent=2)+'\n')
    with np.load(a.data) as f:mu,ls=f['mu'][0],f['log_std'][0]
    samples=np.load(a.samples)
    if a.limit:samples=samples[:a.limit]
    mix=Mixture(mu,ls,a.device)
    X=torch.tensor(samples,dtype=torch.float64,device=a.device)
    if a.endpoints:
        with np.load(a.endpoints) as cache:
            if str(cache['data_sha256'])!=digest(a.data) or str(cache['samples_sha256'])!=digest(a.samples):
                raise ValueError('Endpoint cache belongs to different density or samples')
            end=torch.tensor(cache['points'][:len(X)],dtype=torch.float64,device=a.device)
        if len(end)!=len(X):raise ValueError('Endpoint cache is too short')
        res=torch.cat([mix.em(xx)[1] for xx in end.split(mix.chunk)])
        decrease=float((mix.logp(X)-mix.logp(end)).max())
        if decrease>1e-7:raise ValueError('Cached ascent decreases density')
        info=dict(unconverged=int((res>1e-8).sum()),max_residual=float(res.max()),
                  density_decrease=decrease,reused_endpoint_sha256=digest(a.endpoints))
    else:
        end,res,info=ascend(mix,X,maxit=a.maxit,callback=lambda it,left:report('ascent_progress',iteration=it,remaining=left))
    np.savez(a.out/'endpoints.npz',points=end.cpu().numpy(),residual=res.cpu().numpy(),
             data_sha256=digest(a.data),samples_sha256=digest(a.samples))
    report('sample_ascent',**info)
    if info['unconverged']:raise RuntimeError('Unconverged samples; assignments not fabricated')
    M,verification=catalog(mix,end);lab,assignment=assign(mix,end,M)
    report('initial_modes',modes=len(M),**verification,**assignment)
    qb=np.bincount(lab.cpu().numpy(),minlength=len(M))/len(X)
    top=np.argsort(-qb)[:6]
    t=torch.linspace(0,1,10,device=a.device,dtype=torch.float64)[1:-1,None]
    seg=[M[i]*(1-t)+M[j]*t for i,j in itertools.combinations(top,2)]
    E=torch.cat([M]+seg) if seg else M
    ee,er,ei=ascend(mix,E,maxit=a.maxit)
    report('exploration_ascent',**ei)
    if ei['unconverged']:raise RuntimeError('Unconverged exploration points')
    # Exploration can discover additional modes. Include and verify them.
    M,verification=catalog(mix,torch.cat([end,ee]))
    all_end=torch.cat([end,ee,M]);labels,assignment=assign(mix,all_end,M)
    P=torch.cat([X,E,M]);fv=mix.logp(P);births=mix.logp(M)
    point_violation=fv-births[labels]
    if float(point_violation.max())>1e-7:raise RuntimeError('Point density exceeds assigned birth')
    L=legacy(a.upstream,a.device,0)
    edges=L['knn_edges'](P,a.k)
    labels_np=labels.cpu().numpy();li,lj=labels_np[edges[:,0]],labels_np[edges[:,1]]
    cross=li!=lj;ce=edges[cross];K=len(M)
    if len(ce):
        ii=torch.as_tensor(ce[:,0],device=a.device);jj=torch.as_tensor(ce[:,1],device=a.device)
        h=torch.minimum(torch.minimum(fv[ii],fv[jj]),mix.logp((P[ii]+P[jj])/2)).cpu().numpy()
        key=np.minimum(li,lj)[cross]*K+np.maximum(li,lj)[cross]
        order=np.lexsort((-h,key));key,h=key[order],h[order]
        first=np.r_[True,key[1:]!=key[:-1]]
        be=np.stack([key[first]//K,key[first]%K],1);bh=h[first]
    else:be=np.zeros((0,2),int);bh=np.zeros(0)
    b=births.cpu().numpy();pairs,dead,roots=filtration(b,be,bh)
    pers=pairs[:,0]-pairs[:,1]
    mass=np.bincount(labels_np[:len(X)],minlength=K)/len(X)
    counts={}
    for cutoff in (0.01,0.001,0.0001,0.):
        floor=b.max()+np.log(cutoff) if cutoff else -np.inf
        counts[str(cutoff)]={str(tau):int(((pers>=tau)&(pairs[:,0]>=floor)).sum()+(b[roots]>=floor).sum()) for tau in (.5,1.,2.)}
    np.savez(a.out/'verified_graph.npz',modes=M.cpu().numpy(),births=b,pairs=pairs,pers=pers,
             dead_modes=dead,roots=roots,basin_mass=mass,labels=labels_np[:len(X)],basin_edges=be,basin_edge_heights=bh)
    report('graph',modes=K,n_components=len(roots),negative_pairs=int((pers < -1e-7).sum()),
           minimum_persistence=float(pers.min()) if len(pers) else None,
           point_above_birth=float(point_violation.max()),**verification,**assignment,counts=counts)
    result=dict(source_commit=a.commit,data_sha256=digest(a.data),samples_sha256=digest(a.samples),
                n=len(mu),m=len(X),k=a.k,device=a.device,torch=torch.__version__,numpy=np.__version__,
                events=events,validation_passed=True,saddle_rule='legacy midpoint approximation')
    (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser()
    for name in ('data','samples','out','upstream'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--commit',required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--k',type=int,default=15);p.add_argument('--maxit',type=int,default=20000)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--endpoints',type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    try:execute(a)
    except Exception:
        (a.out/'failure.txt').write_text(traceback.format_exc())
        raise


if __name__=='__main__':main()
