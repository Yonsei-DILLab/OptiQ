"""Exactly 2 or 3/6 JOINT modes, with independent transverse Gaussian coordinates."""
import numpy as np
from scipy.special import logsumexp,ndtr,ndtri
from scipy.signal import find_peaks
from .config import TAU,GRID_N,EDGES_N
GRID=np.linspace(-1,1,GRID_N)
EDGES=np.linspace(-1,1,EDGES_N)
ORTHO_STD=.35

def schedule(stage,t,family='tri'):
 if family=='double':
  c=np.array([-.65,.65]);w=np.array([.5,.5]);h=.12
  if stage=='mass':
   if 20000<t<=25000:w=np.array([.8,.2])
   elif 25000<t<=30000:w=np.array([.2,.8])
  elif stage!='prefix':raise ValueError((stage,family))
 else:
  c=np.array([-.6,0,.6]);w=np.full(3,1/3);h=.1;d=0.
  if stage=='mass':
   if 20000<t<=25000:w=np.array([.6,.2,.2])
   elif 25000<t<=30000:w=np.array([.2,.2,.6])
  elif stage=='split':
   if 20000<t<=22000:d=.15*(t-20000)/2000
   elif 22000<t<=25000:d=.15
   elif 25000<t<=27000:d=.15*(27000-t)/2000
  elif stage!='prefix':raise ValueError((stage,family))
  c=(c[:,None]+[-d,d]).ravel();w=np.repeat(w/2,2)
 return np.stack([c,w,np.full(len(c),h)]).astype(np.float32)

def analytic_q(a,p):
 a=np.asarray(a,float);c,w,h=np.asarray(p,float)
 first=logsumexp(np.log(w)-np.log(h*np.sqrt(2*np.pi))-.5*((a[...,0,None]-c)/h)**2,axis=-1)
 transverse=(-np.log(ORTHO_STD*np.sqrt(2*np.pi))-.5*(a[...,1:]/ORTHO_STD)**2).sum(-1)
 return TAU*(first+transverse)

def analytic_q_jax(a,p):
 import jax.numpy as jnp
 import jax.scipy as jsp
 c,w,h=p
 first=jsp.special.logsumexp(jnp.log(w)-jnp.log(h*jnp.sqrt(2*jnp.pi))-.5*((a[...,0,None]-c)/h)**2,axis=-1)
 transverse=(-jnp.log(ORTHO_STD*jnp.sqrt(2*jnp.pi))-.5*(a[...,1:]/ORTHO_STD)**2).sum(-1)
 return TAU*(first+transverse)

def cdf(x,p):
 c,w,h=np.asarray(p,float);lo=ndtr((-1-c)/h);z=(w*(ndtr((1-c)/h)-lo)).sum()
 return ((ndtr((np.asarray(x)[...,None]-c)/h)-lo)*w).sum(-1)/z

def transverse_cdf(x):
 lo=ndtr(-1/ORTHO_STD);return (ndtr(np.asarray(x)/ORTHO_STD)-lo)/(ndtr(1/ORTHO_STD)-lo)

def target_samples(p,dim,count,seed):
 rng=np.random.default_rng(seed);c,w,h=np.asarray(p,float)
 lo=ndtr((-1-c)/h);hi=ndtr((1-c)/h);mass=w*(hi-lo);mass/=mass.sum()
 ix=rng.choice(len(c),count,p=mass)
 a=np.empty((count,dim));a[:,0]=c[ix]+h[ix]*ndtri(lo[ix]+rng.random(count)*(hi[ix]-lo[ix]))
 if dim>1:
  lo=ndtr(-1/ORTHO_STD);hi=ndtr(1/ORTHO_STD)
  a[:,1:]=ORTHO_STD*ndtri(lo+rng.random((count,dim-1))*(hi-lo))
 return a.astype(np.float32)

def analytic_reference(p,dim):
 a=np.zeros((len(GRID),dim));a[:,0]=GRID;q=analytic_q(a,p)
 f=np.exp((q-q.max())/TAU);pdf=f/np.trapz(f,GRID)
 peaks=find_peaks(pdf)[0];bound=np.unique(np.r_[-1,[GRID[i+np.argmin(pdf[i:j+1])] for i,j in zip(peaks[:-1],peaks[1:])],1])
 marg=np.stack([np.diff(cdf(EDGES,p))]+[np.diff(transverse_cdf(EDGES))]*(dim-1))
 # Factorized expectation: integrate each additive log density in float64.
 first=np.zeros((len(GRID),1));first[:,0]=GRID
 eq=float(np.trapz(pdf*analytic_q(first,p),GRID))
 if dim>1:
  dens=np.exp(-.5*(GRID/ORTHO_STD)**2)/(ORTHO_STD*np.sqrt(2*np.pi));dens/=np.trapz(dens,GRID)
  eq+=(dim-1)*TAU*np.trapz(dens*(-np.log(ORTHO_STD*np.sqrt(2*np.pi))-.5*(GRID/ORTHO_STD)**2),GRID)
 return dict(grid=GRID,q_slice=q,pdf_first=pdf,edges=EDGES,target_marginals=marg,
  boundaries=bound,target_basin_mass=np.diff(cdf(bound,p)),peaks=GRID[peaks],target_backup=eq,
  reference_kind='analytic_factorized_integral',reference_ess=0.,reference_reliable=True)

def reward(s,a):
 a=np.asarray(a);s=np.asarray(s)
 g=np.exp(-.5*((a[...,0,None]-np.array([-.6,0,.6]))/.1)**2-.5*np.square(a[...,1:]/ORTHO_STD).sum(-1)[...,None]).sum(-1)
 return g-.25*np.square(a-s).mean(-1)

class PositionChoice:
 def __init__(self,dim):self.dim=dim;self.state=np.zeros(dim,np.float32);self.episode_step=0
 def step(self,a):
  a=np.asarray(a,np.float32);assert a.shape==(self.dim,) and np.isfinite(a).all() and (np.abs(a)<=1).all()
  r=float(reward(self.state,a));self.episode_step+=1;timeout=self.episode_step==200
  self.state=np.zeros(self.dim,np.float32) if timeout else a.copy()
  if timeout:self.episode_step=0
  return a.copy(),r,False,timeout
