"""Only the registered time schedule changes analytic Q; the MDP is stationary."""
import numpy as np
from scipy.special import logsumexp,ndtr
from scipy.signal import find_peaks
from scipy.optimize import minimize_scalar
from .config import TAU,GRID_N,EDGES_N

GRID=np.linspace(-1,1,GRID_N)
EDGES=np.linspace(-1,1,EDGES_N)
CENTERS=np.array([-.6,0,.6])

def schedule(stage,t):
    # Updates 1..20000 use A. A change at 20K first affects update 20001.
    weights=np.full(3,1/3);d=0.
    if stage=='mass':
        if 20000<t<=25000:weights=np.array([.6,.2,.2])
        elif 25000<t<=30000:weights=np.array([.2,.2,.6])
    elif stage=='split':
        if 20000<t<=22000:d=.15*(t-20000)/2000
        elif 22000<t<=25000:d=.15
        elif 25000<t<=27000:d=.15*(27000-t)/2000
    elif stage!='prefix':raise ValueError(stage)
    centers=(CENTERS[:,None]+np.array([-d,d])).ravel()
    weights=np.repeat(weights/2,2)
    return np.stack([centers,weights]).astype(np.float32)

def analytic_q(a,parameters):
    c,w=np.asarray(parameters,dtype=float)
    return TAU*logsumexp(np.log(w)-np.log(.1*np.sqrt(2*np.pi))-.5*((np.asarray(a)[...,None]-c)/.1)**2,axis=-1)

def analytic_q_jax(a,parameters):
    import jax.numpy as jnp
    import jax.scipy as jsp
    c,w=parameters
    return TAU*jsp.special.logsumexp(jnp.log(w)-jnp.log(.1*jnp.sqrt(2*jnp.pi))-.5*((a[...,0,None]-c)/.1)**2,axis=-1)

def analytic_cdf(x,parameters):
    c,w=np.asarray(parameters,float)
    lower=ndtr((-1-c)/.1);normalizer=np.sum(w*(ndtr((1-c)/.1)-lower))
    return np.sum(w*(ndtr((np.asarray(x)[...,None]-c)/.1)-lower),axis=-1)/normalizer

def reference(q_grid,parameters=None):
    q=np.asarray(q_grid,float);dx=GRID[1]-GRID[0]
    scaled=np.exp((q-q.max())/TAU)
    segments=(scaled[:-1]+scaled[1:])*dx/2
    cdf=np.r_[0,np.cumsum(segments)];z=cdf[-1];cdf/=z
    pdf=scaled/z
    peaks=find_peaks(pdf)[0]
    # Endpoints can be maxima of the learned bounded Q.
    if pdf[0]>pdf[1]:peaks=np.r_[0,peaks]
    if pdf[-1]>pdf[-2]:peaks=np.r_[peaks,len(pdf)-1]
    valleys=[GRID[i+int(np.argmin(pdf[i:j+1]))] for i,j in zip(peaks[:-1],peaks[1:])]
    boundaries=np.unique(np.r_[-1,valleys,1.])
    F=(lambda x:analytic_cdf(x,parameters)) if parameters is not None else (lambda x:np.interp(x,GRID,cdf))
    bin_mass=np.maximum(np.diff(F(EDGES)),0);bin_mass/=bin_mass.sum()
    basin_mass=np.maximum(np.diff(F(boundaries)),0);basin_mass/=basin_mass.sum()
    value=np.trapz(pdf*q,GRID)
    return dict(grid=GRID,q=q,pdf=pdf,edges=EDGES,target_bin_mass=bin_mass,
                boundaries=boundaries,target_basin_mass=basin_mass,peaks=GRID[peaks],target_backup=value)

def reward(state,action):
    a=np.asarray(action)
    return np.exp(-.5*((a[...,None]-CENTERS)/.1)**2).sum(-1)-.25*(a-np.asarray(state))**2

class PositionChoice:
    """Time limit resets to zero but does not remove continuing TD bootstrap."""
    def __init__(self):self.state=0.;self.episode_step=0
    def step(self,a):
        a=float(a);assert -1<=a<=1 and np.isfinite(a)
        r=float(reward(self.state,a));self.episode_step+=1
        timeout=self.episode_step==200
        next_state=a
        self.state=0. if timeout else a
        if timeout:self.episode_step=0
        return next_state,r,False,timeout
