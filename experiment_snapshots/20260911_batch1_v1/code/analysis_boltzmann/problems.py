"""Bounded energy landscapes with independently converged quadrature truth."""
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp, ndtr, ndtri

@dataclass
class Problem:
    name: str
    dim: int
    centers: np.ndarray
    scales: np.ndarray
    weights: np.ndarray
    separable: bool = False
    constant: bool = False

    def q(self, a, xp=np, lse=logsumexp):
        if self.constant:
            return xp.zeros(a.shape[:-1]) + 0.7
        c, s, w = map(xp.asarray, (self.centers, self.scales, self.weights))
        if self.separable:
            # Same per-coordinate energy and temperature across dimensionality.
            v = -.5*((a[..., :, None]-c[:, 0])/s[:, 0])**2
            v = v - xp.log(s[:, 0]) + xp.log(w)
            return .25*xp.sum(lse(v, axis=-1), axis=-1)
        v = -.5*xp.sum(((a[..., None, :]-c)/s)**2, axis=-1)
        return .25*lse(v-xp.sum(xp.log(s),axis=-1)+xp.log(w),axis=-1)

    def labels(self, a):
        if self.separable:
            bits = np.argmin(abs(a[..., :, None]-self.centers[:, 0]), axis=-1)
            return (bits * (len(self.centers)**np.arange(self.dim))).sum(-1)
        return np.argmin(np.sum(((a[...,None,:]-self.centers)/self.scales)**2,axis=-1),axis=-1)

    def base(self):
        return Problem(self.name,1,self.centers,self.scales,self.weights)

    def reference(self, n=512):
        if self.separable:
            grid, mass, value = self.base().reference(n)
            return grid, mass, self.dim*value
        # Midpoint quadrature: avoids endpoint point masses and scales to 2D.
        x = -1+(np.arange(n)+.5)*2/n
        if self.dim==1: grid=x[:,None]
        else: grid=np.stack(np.meshgrid(x,x,indexing='ij'),-1).reshape(-1,2)
        q=self.q(grid); mass=np.exp(q/.25-logsumexp(q/.25))
        return grid,mass,float(mass@q)

    def truth(self, tol=1e-4):
        _,_,old=self.reference(256)
        for n in (512,1024,2048):
            grid,mass,value=self.reference(n)
            if abs(value-old)<tol: return grid,mass,value,n,abs(value-old)
            old=value
        raise RuntimeError(f'Quadrature did not converge: {self.name}')

    def reference_sample(self,rng,size,grid,mass):
        if self.separable:
            return grid[rng.choice(len(grid),size=(*size,self.dim),p=mass),0]
        return grid[rng.choice(len(grid),size=size,p=mass)]


def make_problem(name):
    if name=='constant': return Problem(name,1,np.array([[0.]]),np.array([[.2]]),np.ones(1),constant=True)
    if name=='unimodal': return Problem(name,1,np.array([[.2]]),np.array([[.18]]),np.ones(1))
    if name.startswith('separable'):
        return Problem(name,int(name.split('_')[1]),np.array([[-.65],[.65]]),np.array([[.12],[.2]]),np.array([.3,.7]),separable=True)
    if name.startswith('modes2d'):
        k=int(name.split('_')[1]); angle=np.arange(k)*2*np.pi/k
        centers=.65*np.stack((np.cos(angle),np.sin(angle)),-1)
        scales=np.stack((np.linspace(.07,.16,k),np.linspace(.16,.07,k)),-1)
        w=np.arange(1,k+1,dtype=float); w/=w.sum()
        return Problem(name,2,centers,scales,w)
    # Fixed grid varies separation independently from width/relative mass.
    _, sep, width = name.split('_'); sep=float(sep); width=float(width)
    symmetric=name.startswith('symmetric')
    return Problem(name,1,np.array([[-sep/2],[sep/2]]),np.array([[width],[width if symmetric else width*1.8]]),np.array([.5,.5] if symmetric else [.2,.8]))

CASES=['constant','unimodal','symmetric_1.3_0.12']+[f'asymmetric_{s}_{w}' for s in (.4,.8,1.3) for w in (.08,.16)]+['modes2d_4','modes2d_8','separable_4','separable_8']


def local_estimates(problem, center, k, repetitions, rng, mode):
    noise=rng.normal(0,.2,(repetitions,k,problem.dim))
    if mode=='truncated_is':
        lo=np.maximum(-1-center,-.5); hi=np.minimum(1-center,.5)
        fl,fh=ndtr(lo/.2),ndtr(hi/.2)
        noise=.2*ndtri(np.clip(fl+(fh-fl)*rng.random(noise.shape),1e-12,1-1e-12))
        logp=np.sum(-.5*(noise/.2)**2-np.log(.2*np.sqrt(2*np.pi))-np.log(fh-fl),axis=-1)
        a=center+noise
    else:
        # Official SD3: density evaluated BEFORE clipping the sampled noise.
        logp=np.sum(-.5*(noise/.2)**2-np.log(.2*np.sqrt(2*np.pi)),axis=-1)
        a=np.clip(center+np.clip(noise,-.5,.5),-1,1)
    q=problem.q(a)
    logits=q/.25-(logp if mode!='no_is' else 0)
    w=np.exp(logits-logsumexp(logits,axis=1,keepdims=True))
    return np.sum(w*q,axis=1),a,w
