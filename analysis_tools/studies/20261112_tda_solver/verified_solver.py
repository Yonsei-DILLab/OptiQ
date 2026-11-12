"""Numerically verified basin ascent for diagonal box-truncated GMMs.

This finds sampled basins, not a proof that every continuous mode was found.
The graph uses the legacy midpoint rule for controlled comparison. Its saddles
are approximations, not certified minima of continuous edges.
"""
import math
import numpy as np
import torch


class Mixture:
    def __init__(self,mu,ls,device='cpu',max_logits=32_000_000):
        self.mu=torch.as_tensor(mu,dtype=torch.float64,device=device)
        self.ls=torch.as_tensor(ls,dtype=torch.float64,device=device)
        self.device=self.mu.device
        self.sd=self.ls.exp();self.lam=(-2*self.ls).exp();self.lm=self.lam*self.mu
        z=torch.special.ndtr((1-self.mu)/self.sd)-torch.special.ndtr((-1-self.mu)/self.sd)
        if not bool((z>0).all()):raise ValueError('Nonpositive component normalizer')
        self.log_norm=-math.log(len(self.mu))-self.ls.sum(-1)-z.log().sum(-1)-.5*self.mu.shape[1]*math.log(2*math.pi)
        self.c=self.log_norm-.5*(self.lam*self.mu**2).sum(-1)
        self.chunk=max(1,max_logits//len(self.mu))

    def logits(self,x):
        return self.c+x@self.lm.T-.5*(x*x)@self.lam.T

    def logp(self,x):
        return torch.cat([torch.logsumexp(self.logits(v),-1) for v in x.split(self.chunk)])

    def em(self,x):
        r=torch.softmax(self.logits(x),-1)
        precision=r@self.lam
        nxt=((r@self.lm)/precision).clamp(-1,1)
        # For an interior point, (nxt-x)*sqrt(precision) is the gradient
        # in local precision units. Clipping gives the projected analogue.
        residual=((nxt-x)*precision.sqrt()).abs().amax(-1)
        return nxt,residual

    def derivatives(self,x):
        r=torch.softmax(self.logits(x),-1)
        v=self.lam[None]*(self.mu[None]-x[:,None])
        g=torch.einsum('bn,bnd->bd',r,v)
        precision=r@self.lam
        h=torch.einsum('bn,bnd,bne->bde',r,v,v)-torch.diag_embed(precision)-g[:,:,None]*g[:,None,:]
        return g,h,precision


def ascend(mix,x,tol=1e-8,maxit=20000,callback=None):
    x=x.clone().to(dtype=torch.float64,device=mix.device)
    initial=mix.logp(x);done=torch.zeros(len(x),dtype=torch.bool,device=x.device)
    iterations=torch.zeros(len(x),dtype=torch.int32,device=x.device)
    residual=torch.full((len(x),),float('inf'),dtype=x.dtype,device=x.device)
    for it in range(maxit):
        active=(~done).nonzero().flatten()
        if not len(active):break
        for j in active.split(mix.chunk):
            nxt,r=mix.em(x[j]);x[j]=nxt;residual[j]=r
            done[j[r<=tol]]=True;iterations[j]+=1
        if callback and (it+1)%100==0:callback(it+1,int((~done).sum()))
    # Verify residual at the returned point, rather than the previous iterate.
    for start in range(0,len(x),mix.chunk):
        sl=slice(start,start+mix.chunk);_,residual[sl]=mix.em(x[sl])
    final=mix.logp(x)
    info=dict(unconverged=int((residual>tol).sum()),max_residual=float(residual.max()),
              max_iterations=int(iterations.max()),density_decrease=float((initial-final).max()))
    if info['density_decrease']>1e-7:raise RuntimeError(('EM density decreased',info))
    return x,residual,info


def verify(mix,x,grad_tol=1e-7,curvature_tol=1e-7):
    accepted=[];residuals=[];curvatures=[]
    for xx in x.split(max(1,min(16,2_000_000//len(mix.mu)))):
        g,h,prec=mix.derivatives(xx);sg=g/prec.sqrt()
        # Strictly outward active coordinates have no critical direction.
        active=(xx.abs()>=1-1e-12)&(sg*xx.sign()>grad_tol)
        pg=sg.masked_fill(active,0)
        hh=h/(prec.sqrt()[:,:,None]*prec.sqrt()[:,None,:])
        free=(~active).double()
        hh=hh*free[:,:,None]*free[:,None,:]-torch.diag_embed(1-free)
        ev=torch.linalg.eigvalsh(hh)[:,-1];res=pg.abs().amax(-1)
        accepted.append((res<=grad_tol)&(ev<-curvature_tol))
        residuals.append(res);curvatures.append(ev)
    return torch.cat(accepted),torch.cat(residuals),torch.cat(curvatures)


def catalog(mix,endpoints,dedup_tol=1e-7):
    # Greedy numerical equality after convergence; no coarse pre-ascent merge.
    order=torch.argsort(mix.logp(endpoints),descending=True)
    representatives=[]
    for indices in order.split(2048):
        x=endpoints[indices]
        if representatives:
            old=torch.cat(representatives)
            x=x[(torch.cdist(x,old,p=float('inf'))>=dedup_tol).all(1)]
        if not len(x):continue
        adj=(torch.cdist(x,x,p=float('inf'))<dedup_tol).cpu().numpy()
        suppressed=np.zeros(len(x),bool);keep=[]
        for i in range(len(x)):
            if not suppressed[i]:keep.append(i);suppressed|=adj[i]
        representatives.append(x[keep])
    modes=torch.cat(representatives)
    ok,res,curv=verify(mix,modes)
    if not bool(ok.all()):
        raise RuntimeError(f'{int((~ok).sum())}/{len(ok)} candidates fail gradient/curvature verification; no forced fallback')
    return modes,dict(max_mode_residual=float(res.max()),maximum_scaled_eigenvalue=float(curv.max()))


def assign(mix,endpoints,modes,tol=1e-6):
    labels=[];distances=[]
    for x in endpoints.split(2048):
        d,l=torch.cdist(x,modes,p=float('inf')).min(1)
        labels.append(l);distances.append(d)
    labels=torch.cat(labels);distances=torch.cat(distances)
    far=int((distances>tol).sum())
    violation=mix.logp(endpoints)-mix.logp(modes)[labels]
    if far or float(violation.max())>1e-7:
        raise RuntimeError(('Unresolved basin assignment',far,float(distances.max()),float(violation.max())))
    return labels,dict(far=far,max_assignment_distance=float(distances.max()),
                       endpoint_above_birth=float(violation.max()))


def filtration(births,edges,heights):
    if len(edges) and np.max(heights-np.minimum(births[edges[:,0]],births[edges[:,1]]))>1e-7:
        raise ValueError('Edge exceeds incident birth; invalid basin filtration')
    parent=np.arange(len(births));pairs=[];dead=[]
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    for i in np.argsort(-heights,kind='stable'):
        x,y=find(edges[i,0]),find(edges[i,1])
        if x==y:continue
        if births[x]<births[y]:x,y=y,x
        pairs.append((births[y],heights[i]));dead.append(y);parent[y]=x
    pairs=np.asarray(pairs).reshape(-1,2)
    roots=np.unique([find(i) for i in range(len(parent))])
    if len(pairs) and np.min(pairs[:,0]-pairs[:,1]) < -1e-7:
        raise ValueError('Negative persistence')
    return pairs,np.asarray(dead,dtype=int),roots
