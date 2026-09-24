"""Explicit non-Gaussian densities: logistic spikes, tilted shelves and ripples."""
from functools import lru_cache
import json
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from scipy.integrate import quad, cumulative_simpson
from scipy.special import logsumexp
from scipy.signal import find_peaks
from scipy.optimize import minimize_scalar


def np_log_shape(x,t):
    x=np.asarray(x,float)
    if t['kind']=='logistic':
        v=(x-t['center'])/t['width']
        return -np.log(t['width'])-np.logaddexp(0,-v)-np.logaddexp(0,v)
    v=(x-t['left'])/t['edge_left'];u=(t['right']-x)/t['edge_right']
    result=-np.logaddexp(0,-v)-np.logaddexp(0,-u)+t.get('tilt',0)*(x-(t['left']+t['right'])/2)
    if t.get('ripple',0):result+=np.log1p(t['ripple']*np.cos(t['frequency']*(x-t.get('phase_center',0))))
    return result


def jax_log_shape(x,t):
    if t['kind']=='logistic':
        v=(x-t['center'])/t['width']
        return -jnp.log(t['width'])+jax.nn.log_sigmoid(v)+jax.nn.log_sigmoid(-v)
    v=(x-t['left'])/t['edge_left'];u=(t['right']-x)/t['edge_right']
    result=jax.nn.log_sigmoid(v)+jax.nn.log_sigmoid(u)+t.get('tilt',0)*(x-(t['left']+t['right'])/2)
    if t.get('ripple',0):result+=jnp.log1p(t['ripple']*jnp.cos(t['frequency']*(x-t.get('phase_center',0))))
    return result


@lru_cache(None)
def term_norms(key):
    terms=json.loads(key)
    return np.array([quad(lambda x:float(np.exp(np_log_shape(x,t))),-10,10,epsabs=1e-11,epsrel=1e-11,limit=500)[0] for t in terms])


def logs_and_weights(cfg):
    return np.log(np.array(cfg['shape_masses'])/term_norms(json.dumps(cfg['shapes'],sort_keys=True)))


def np_pdf(cfg,x):
    terms=np.stack([np_log_shape(x,t) for t in cfg['shapes']],-1)
    return np.exp(logsumexp(terms+logs_and_weights(cfg),axis=-1))


@lru_cache(None)
def reference_grid(key):
    cfg=json.loads(key);x=np.linspace(-10,10,262145);pdf=np_pdf(cfg,x)
    cdf=cumulative_simpson(pdf,x=x,initial=0)
    assert abs(cdf[-1]-1)<1e-8
    return x,cdf


def reference(cfg,x):
    arr=np.asarray(x,float);grid,cdf=reference_grid(json.dumps(dict(shapes=cfg['shapes'],shape_masses=cfg['shape_masses']),sort_keys=True))
    return np.where(np.abs(arr)<=10,np_pdf(cfg,arr),0),np.interp(arr,grid,cdf,left=0,right=1)


@lru_cache(None)
def cached_geometry(key):
    cfg=json.loads(key);x=np.linspace(-10,10,131073);y=np_pdf(cfg,x)
    idx,_=find_peaks(y,prominence=y.max()*1e-4,distance=100)
    peaks=np.array([minimize_scalar(lambda a:-float(np_pdf(cfg,a)),bounds=(x[i-1],x[i+1]),method='bounded').x for i in idx])
    assert len(peaks)==cfg['expected_modes'],(cfg['id'],peaks)
    valleys=np.array([minimize_scalar(lambda a:float(np_pdf(cfg,a)),bounds=(a,z),method='bounded').x for a,z in zip(peaks[:-1],peaks[1:])])
    bounds=np.r_[-10,valleys,10]
    # Fixed geometric core windows declared with targets before seeing any run.
    cores=np.asarray(cfg['core_intervals']);cl=cores[:,0];cr=cores[:,1]
    assert np.all(cl<cr) and np.all(cl>=bounds[:-1]) and np.all(cr<=bounds[1:])
    target_core=reference(cfg,cr)[1]-reference(cfg,cl)[1]
    target_basin=np.diff(reference(cfg,bounds)[1])
    vh=np.full(len(valleys),.15);vl=np.maximum(valleys-vh,cr[:-1]);vr=np.minimum(valleys+vh,cl[1:])
    assert np.all(vl<vr)
    tv=reference(cfg,vr)[1]-reference(cfg,vl)[1]
    return dict(peaks=peaks,basin_bounds=bounds,core_left=cl,core_right=cr,target_core=target_core,
                target_basin=target_basin,valley_left=vl,valley_right=vr,target_valley=tv,
                target_valley_core_ratio=tv/np.minimum(target_core[:-1],target_core[1:]))


def geometry(cfg):return cached_geometry(json.dumps({k:cfg[k] for k in ['id','shapes','shape_masses','expected_modes','core_intervals']},sort_keys=True))
