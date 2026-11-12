"""Instrument the supplied equations, keeping original samples and result artifacts."""
import argparse
import ast
import hashlib
import json
import math
import pickle
import time
from pathlib import Path
import numpy as np
import torch


def legacy(path, device, seed):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef,ast.ClassDef))]
    namespace = dict(dev=device, SQ2=math.sqrt(2), EPS=1e-12,
                     gen=torch.Generator(device=device).manual_seed(seed))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'), namespace)
    namespace['ndtr'] = lambda x: .5*torch.erfc(-x/math.sqrt(2))
    return namespace


def details(mix, x):
    r = torch.softmax(mix.logits(x),-1)
    v = mix.lam[None]*(mix.mu[None]-x[:,None])
    g = torch.einsum('bn,bnd->bd',r,v)
    h = torch.einsum('bn,bnd,bne->bde',r,v,v)-torch.diag_embed(r@mix.lam)-g[:,:,None]*g[:,None,:]
    active = (x.abs()>1-1e-10)&(g*x.sign()>0)
    pg = g.masked_fill(active,0)
    scaled = pg/(r@mix.lam).sqrt()
    return g,h,scaled.abs().amax(-1)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--upstream',type=Path,required=True);p.add_argument('--commit',required=True)
    p.add_argument('--samples',type=int,default=32768);p.add_argument('--k',type=int,default=15)
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=0)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    L=legacy(a.upstream,a.device,a.seed);start=time.monotonic();dev=a.device
    with np.load(a.data) as f:mu,ls=f['mu'][0],f['log_std'][0];state=int(f['original_state'])
    m64=L['Mix'](torch.tensor(mu,device=dev,dtype=torch.float64),torch.tensor(ls,device=dev,dtype=torch.float64))
    m32=L['Mix'](m64.mu.float(),m64.ls.float())
    for _ in range(state):
        torch.randint(0,len(mu),(a.samples,),device=dev,generator=L['gen'])
        torch.rand((a.samples,mu.shape[1]),device=dev,generator=L['gen'],dtype=torch.float64)
    X=m64.sample(a.samples)
    np.save(a.out/'samples.npy',X.cpu().numpy())
    log=[]
    def progress(stage,**values):
        row=dict(stage=stage,seconds=time.monotonic()-start,**values);log.append(row)
        print(json.dumps(row),flush=True)
        (a.out/'progress.json').write_text(json.dumps(log,indent=2)+'\n')
    score_error=m32.logp(X.float()).double()-m64.logp(X.float().double())
    progress('sampled',score_error_max=float(score_error.abs().max()))
    xc=L['ascend'](m32,X.float(),1e-4,3000)
    np.save(a.out/'legacy_endpoints.npy',xc.cpu().numpy())
    cand=L['dedup'](xc,m32.logp(xc),1e-3)
    before=cand.shape[0]
    cand=L['ascend'](m64,cand.double(),1e-10,5000)
    cand=L['dedup'](cand,m64.logp(cand),1e-3)
    ok=L['local_max_mask'](m64,cand)
    residual=[]
    for chunk in cand.split(16):residual.append(details(m64,chunk)[2])
    residual=torch.cat(residual)
    fallback=not bool(ok.any())
    if fallback:ok[0]=True
    M=cand[ok];M=M[torch.argsort(m64.logp(M),descending=True)]
    np.savez(a.out/'candidates.npz',points=cand.cpu().numpy(),accepted=ok.cpu().numpy(),scaled_gradient=residual.cpu().numpy())
    lab=torch.empty(len(X),device=dev,dtype=torch.long);far=0;distances=[]
    for j in range(0,len(X),4096):
        dm,am=(xc[j:j+4096].double()[:,None]-M[None]).abs().amax(-1).min(1)
        lab[j:j+4096]=am;far+=int((dm>1e-2).sum());distances.append(dm)
    births=m64.logp(M);fx=m64.logp(X)
    endpoint_lp=m64.logp(xc.double());height_violation=endpoint_lp-births[lab]
    progress('modes',coarse_candidates=before,candidates=len(cand),modes=len(M),forced_fallback=fallback,
             far=far,accepted_scaled_gradient_max=float(residual[ok].max()),
             endpoint_above_assigned_mode_count=int((height_violation>1e-6).sum()),
             endpoint_above_assigned_mode_max=float(height_violation.max()))
    qb=np.bincount(lab.cpu().numpy(),minlength=len(M))/len(X)
    top=np.argsort(-qb)[:6]
    import itertools
    t=torch.linspace(0,1,10,device=dev,dtype=torch.float64)[1:-1,None]
    seg=[M[i]*(1-t)+M[j]*t for i,j in itertools.combinations(top,2)]
    E=torch.cat([M]+seg) if seg else M
    ec=L['ascend'](m32,E.float(),1e-4,3000)
    el=(ec.double()[:,None]-M[None]).abs().amax(-1).argmin(1)
    P=torch.cat([X,E]);labels=torch.cat([lab,el]).cpu().numpy()
    fv=m64.logp(P);edges=L['knn_edges'](P,a.k)
    ii=torch.tensor(edges[:,0],device=dev);jj=torch.tensor(edges[:,1],device=dev)
    heights=torch.minimum(torch.minimum(fv[ii],fv[jj]),m64.logp((P[ii]+P[jj])/2)).cpu().numpy()
    li,lj=labels[edges[:,0]],labels[edges[:,1]];cross=li!=lj;K=len(M)
    if cross.any():
        key=np.minimum(li,lj)[cross]*K+np.maximum(li,lj)[cross]
        order=np.lexsort((-heights[cross],key));key,hc=key[order],heights[cross][order]
        first=np.r_[True,key[1:]!=key[:-1]]
        be=np.stack([key[first]//K,key[first]%K],1);bh=hc[first]
    else:be=np.zeros((0,2),int);bh=np.zeros(0)
    b=births.cpu().numpy();pairs,root=L['basin_filtration'](b,be,bh)
    pers=pairs[:,0]-pairs[:,1]
    edge_bad=bh-np.minimum(b[be[:,0]],b[be[:,1]]) if len(be) else np.zeros(0)
    np.savez(a.out/'legacy_graph.npz',modes=M.cpu().numpy(),births=b,basin_mass=qb,
             labels=lab.cpu().numpy(),assignment_distance=torch.cat(distances).cpu().numpy(),
             endpoint_height_violation=height_violation.cpu().numpy(),
             basin_edges=be,basin_edge_heights=bh,pairs=pairs,pers=pers)
    progress('graph',negative_pairs=int((pers<-1e-6).sum()),minimum_persistence=float(pers.min()) if len(pers) else None,
             edges_above_incident_births=int((edge_bad>1e-6).sum()),
             maximum_edge_birth_violation=float(edge_bad.max()) if len(edge_bad) else None)
    result=dict(source_commit=a.commit,input_sha256=hashlib.sha256(a.data.read_bytes()).hexdigest(),
                torch=torch.__version__,numpy=np.__version__,device=torch.cuda.get_device_name() if dev=='cuda' else dev,
                original_state=state,samples=a.samples,n=len(mu),k=a.k,diagnostics=log)
    (a.out/'diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
