"""Float64 target reference and actual density-mode boundaries, evaluation only."""
import numpy as np
from scipy.special import ndtr,logsumexp
from scipy.signal import find_peaks
from scipy.optimize import minimize_scalar


def parameters(cfg):
    c=np.asarray(cfg['target_centers'],dtype=float);h=np.asarray(cfg['target_widths'],dtype=float)
    w=np.asarray(cfg['target_masses'],dtype=float);b=float(cfg['action_bound'])
    assert c.ndim==1 and c.shape==h.shape==w.shape and np.all(np.diff(c)>0)
    assert np.all(h>0) and np.all(w>0) and np.isclose(w.sum(),1)
    return c,h,w,b


def reference(cfg,x):
    c,h,w,b=parameters(cfg);x=np.asarray(x,dtype=float)
    norm=np.sum(w*(ndtr((b-c)/h)-ndtr((-b-c)/h)))
    ell=np.log(w)-np.log(h)-.5*np.log(2*np.pi)-.5*((x[...,None]-c)/h)**2
    pdf=np.exp(logsumexp(ell,axis=-1))/norm
    cdf=np.sum(w*(ndtr((x[...,None]-c)/h)-ndtr((-b-c)/h)),axis=-1)/norm
    return pdf,cdf


def geometry(cfg):
    c,h,w,b=parameters(cfg);x=np.linspace(-b,b,65537);y=reference(cfg,x)[0]
    idx,_=find_peaks(y,prominence=y.max()*1e-5)
    assert len(idx)==len(c),f'{cfg["id"]}: {len(c)} components but {len(idx)} density peaks'
    peaks=np.array([minimize_scalar(lambda a:-float(reference(cfg,a)[0]),bounds=(x[i-1],x[i+1]),method='bounded').x for i in idx])
    valleys=np.array([minimize_scalar(lambda a:float(reference(cfg,a)[0]),bounds=(a,z),method='bounded').x for a,z in zip(peaks[:-1],peaks[1:])])
    bounds=np.r_[-b,valleys,b]
    core_left=np.maximum(peaks-h,bounds[:-1]);core_right=np.minimum(peaks+h,bounds[1:])
    target_core=reference(cfg,core_right)[1]-reference(cfg,core_left)[1]
    target_basin=np.diff(reference(cfg,bounds)[1])
    vh=np.minimum(h[:-1],h[1:]);vl=np.maximum(valleys-vh,core_right[:-1]);vr=np.minimum(valleys+vh,core_left[1:])
    assert np.all(vl<vr),'Core windows overlap valley windows'
    tv=reference(cfg,vr)[1]-reference(cfg,vl)[1]
    return dict(peaks=peaks,basin_bounds=bounds,core_left=core_left,core_right=core_right,
                target_core=target_core,target_basin=target_basin,valley_left=vl,valley_right=vr,
                target_valley=tv,target_valley_core_ratio=tv/np.minimum(target_core[:-1],target_core[1:]))
