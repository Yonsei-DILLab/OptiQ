"""Population 3-GMM, fixed equal weights and sigma; only means are optimized."""
import json
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import jax.scipy as jsp
from scipy.integrate import quad_vec
from scipy.special import logsumexp,ndtr


def load_config():return json.loads(Path(__file__).with_name('config.json').read_text())


def cases(cfg):
    out=[dict(id=f'R{r:g}_D{d:g}',centers=[-r,r,d]) for r in cfg['near_R'] for d in cfg['far_D']]
    out.append(dict(id='old_symmetric_control',centers=[-4.25,0.,4.25]))
    return out


def mixture_np(x,mu,sigma):
    t=(np.asarray(x)[...,None]-np.asarray(mu))/sigma
    return np.exp(logsumexp(-.5*t*t,axis=-1)-np.log(3*sigma*np.sqrt(2*np.pi))),ndtr(t).mean(axis=-1)


def quadrature(c,sigma,n):
    x=np.linspace(min(c)-10*sigma,max(c)+10*sigma,n)
    a=np.ones(n);a[1:-1:2]=4;a[2:-1:2]=2
    mass=mixture_np(x,c,sigma)[0]*a*(x[1]-x[0])/3
    assert n%2==1 and abs(mass.sum()-1)<1e-12
    return x,mass


def loss(mu,x,mass,sigma):
    logq=jsp.special.logsumexp(-.5*((x[:,None]-mu)/sigma)**2,axis=-1)-np.log(3*sigma*np.sqrt(2*np.pi))
    return -jnp.sum(mass*logq)


def initial_means(case,cfg):
    c=np.asarray(case['centers']);mid=(c[0]+c[1])/2;delta=cfg['duplicate_initial_offset']
    start=np.array([mid,c[2]-delta,c[2]+delta])
    rows=[];meta=[]
    for kind in ['bad_structure','good_structure']:
        for seed in cfg['seeds']:
            jitter=np.random.default_rng(seed).normal(0,cfg['initial_jitter'],3)
            rows.append((start if kind=='bad_structure' else c)+jitter)
            meta.append(dict(kind=kind,seed=seed))
    return np.stack(rows),meta


class Trainer:
    def __init__(self,case,cfg):
        self.cfg=cfg;self.case=case
        x,m=quadrature(case['centers'],cfg['sigma'],cfg['quadrature_points']);self.x=jnp.array(x);self.mass=jnp.array(m)
        self.fn=lambda mu:loss(mu,self.x,self.mass,cfg['sigma'])
        self.vg=jax.jit(jax.vmap(jax.value_and_grad(self.fn)))
        mu,self.meta=initial_means(case,cfg);self.mu=jnp.array(mu)
        def step(mu,_):
            val,g=self.vg(mu);return mu-cfg['learning_rate']*g,None
        self.advance=jax.jit(lambda mu,n:jax.lax.scan(step,mu,None,length=n)[0],static_argnums=(1,))


def adaptive(mu,centers,sigma):
    """Independent analytic gradient/Hessian, integrated under each true Gaussian."""
    mu=np.asarray(mu,float);total=np.zeros(13);errors=[]
    for c in centers:
        def integrand(t):
            x=c+sigma*t;d=mu-x;logit=-.5*(d/sigma)**2
            r=np.exp(logit-logsumexp(logit));g=r*d/sigma**2
            h=np.diag(r/sigma**2-r*d*d/sigma**4)+np.outer(r*d,r*d)/sigma**4
            val=-logsumexp(logit)+np.log(3*sigma*np.sqrt(2*np.pi))
            return np.r_[val,g,h.ravel()]*np.exp(-.5*t*t)/np.sqrt(2*np.pi)/3
        boundaries=[(v-c)/sigma for v in [*mu,*((mu[:,None]+mu[None,:])/2).ravel()]]
        pts=sorted(set([-8.,-4.,0.,4.,8.]+[float(v) for v in boundaries if -12<v<12]))
        val,err=quad_vec(integrand,-12,12,epsabs=2e-13,epsrel=2e-13,points=pts,limit=1000)
        total+=val;errors.append(float(err))
    return float(total[0]),total[1:4],total[4:].reshape(3,3),sum(errors)
