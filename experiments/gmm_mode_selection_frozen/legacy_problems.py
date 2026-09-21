"""Stationary 1D energies with analytically normalized Boltzmann references."""
import functools
import numpy as np
from scipy.special import ndtr, logsumexp
from scipy.integrate import quad
from scipy.signal import find_peaks
from scipy.optimize import minimize_scalar

TAU = .25
LANDSCAPES = {
    'needle3': dict(label='Narrow and broad: 3 modes',
        centers=[-.7,0.,.7], scales=[.035,.18,.06], weights=[.3,.4,.3]),
    'comb6': dict(label='Six narrow, equally spaced modes',
        centers=[-.75,-.45,-.15,.15,.45,.75], scales=[.035]*6, weights=[1/6]*6),
    'rugged7': dict(label='Unequal widths and masses: 7 modes',
        centers=[-.9,-.6,-.32,.02,.3,.58,.9],
        scales=[.025,.075,.02,.11,.035,.06,.018], weights=[.12,.18,.1,.2,.12,.16,.12]),
}

def log_energy(x, name):
    c=LANDSCAPES[name];mu=np.array(c['centers']);sd=np.array(c['scales']);w=np.array(c['weights'])
    return logsumexp(np.log(w/sd)-.5*np.square((np.asarray(x)[...,None]-mu)/sd)-.5*np.log(2*np.pi),axis=-1)

def cdf(x,name):
    c=LANDSCAPES[name];mu=np.array(c['centers']);sd=np.array(c['scales']);w=np.array(c['weights'])
    lo=ndtr((-1-mu)/sd);hi=ndtr((1-mu)/sd)
    return np.sum(w*(ndtr((np.asarray(x)[...,None]-mu)/sd)-lo),axis=-1)/np.sum(w*(hi-lo))

def pdf(x,name):
    c=LANDSCAPES[name];mu=np.array(c['centers']);sd=np.array(c['scales']);w=np.array(c['weights'])
    z=np.sum(w*(ndtr((1-mu)/sd)-ndtr((-1-mu)/sd)))
    return np.exp(log_energy(x,name))/z

@functools.lru_cache(None)
def reference(name):
    c=LANDSCAPES[name];x=np.linspace(-1,1,65537);p=pdf(x,name)
    peaks=find_peaks(p)[0];assert len(peaks)==len(c['centers']),(name,len(peaks))
    valleys=[minimize_scalar(lambda a:float(pdf(a,name)),bounds=(x[i],x[j]),method='bounded',
                            options={'xatol':1e-13}).x for i,j in zip(peaks[:-1],peaks[1:])]
    boundaries=np.array([-1,*valleys,1.]);mass=np.diff(cdf(boundaries,name))
    expectation,error=quad(lambda a:float(pdf(a,name)*TAU*log_energy(a,name)), -1,1,
                           points=sorted([*c['centers'],*valleys]),epsabs=1e-10,epsrel=1e-10,limit=300)
    # Independent fine midpoint quadrature validates the nonuniform adaptive integral.
    xm=(x[:-1]+x[1:])/2;qm=np.sum(pdf(xm,name)*TAU*log_energy(xm,name))*2/(len(x)-1)
    assert abs(qm-expectation)<2e-7 and error<1e-8
    edges=np.linspace(-1,1,513)
    return dict(name=name,**c,tau=TAU,boundaries=boundaries.tolist(),mode_mass=mass.tolist(),
                q=expectation,q_quad_error=error,q_midpoint_delta=abs(qm-expectation),
                peaks=x[peaks].tolist(),edges=edges.tolist(),bin_mass=np.diff(cdf(edges,name)).tolist())

def q_jax(a,name):
    import jax.numpy as jnp
    import jax.scipy as jsp
    c=LANDSCAPES[name];mu=jnp.asarray(c['centers']);sd=jnp.asarray(c['scales']);w=jnp.asarray(c['weights'])
    return TAU*jsp.special.logsumexp(jnp.log(w/sd)-.5*jnp.square((a[...,0,None]-mu)/sd)-.5*jnp.log(2*jnp.pi),axis=-1)

METHODS=('sinkhorn_nll','gmm_nll','exact_ot_nll','barycentric_mse','argmax_mse')
def tasks():
    return [dict(name=f'{p}_{m}_s{s}',landscape=p,method=m,seed=s,updates=20000)
            for s in range(4) for p in LANDSCAPES for m in METHODS]
