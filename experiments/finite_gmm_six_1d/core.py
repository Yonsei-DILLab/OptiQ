"""Finite component identities, direct parameter GD; no network, EM, OT or SNIS."""
import json
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import jax.scipy as jsp
import optax
from experiments.kl_diverse_targets_1d.target import reference as gmm_reference
from experiments.kl_nongmm_targets_1d.target import reference as other_reference


def reference(cfg, x):
    if cfg.get('unbounded', False):
        from scipy.special import ndtr
        c=np.asarray(cfg['target_centers']);h=np.asarray(cfg['target_widths']);w=np.asarray(cfg['target_masses'])
        t=(np.asarray(x)[...,None]-c)/h
        return np.sum(w*np.exp(-t*t/2)/(h*np.sqrt(2*np.pi)),axis=-1),np.sum(w*ndtr(t),axis=-1)
    return (other_reference if cfg['target_kind']=='nongmm' else gmm_reference)(cfg,x)


def quadrature(cfg, count):
    assert count%2==1
    bound=cfg.get('integration_bound',cfg['action_bound'])
    x=np.linspace(-bound,bound,count)
    coeff=np.ones(count);coeff[1:-1:2]=4;coeff[2:-1:2]=2
    mass=reference(cfg,x)[0]*coeff*(x[1]-x[0])/3
    assert abs(mass.sum()-1)<1e-8, (cfg['id'],mass.sum())
    return x,mass/mass.sum()


def log_density(p, x, cfg):
    mu=p[0];ls=p[1] if cfg['variance']=='learned' else jnp.full_like(mu,np.log(cfg['fixed_sigma']))
    lw=jax.nn.log_softmax(p[2]) if cfg['weights']=='learned' else jnp.full_like(mu,-np.log(mu.size))
    z=(x[:,None]-mu)/jnp.exp(ls)
    ell=-.5*z*z-ls-.5*np.log(2*np.pi)
    if not cfg.get('unbounded',False):
        b=cfg['action_bound'];norm=jsp.special.ndtr((b-mu)/jnp.exp(ls))-jsp.special.ndtr((-b-mu)/jnp.exp(ls))
        ell=ell-jnp.log(norm)
    return jsp.special.logsumexp(ell+lw,axis=-1)


def loss(p,x,mass,cfg):return -jnp.sum(mass*log_density(p,x,cfg))


def initialize(cfg):
    rows=[]
    for seed in cfg['seeds']:
        rng=np.random.default_rng(seed)
        mu=rng.uniform(-8,8,15)[:cfg['k']]
        if cfg.get('initialization')=='bad_structure':mu=np.array([-2.125,4.25,4.25])+rng.normal(0,.025,3)
        elif cfg.get('initialization')=='good_structure':mu=np.array([-4.25,0,4.25])+rng.normal(0,.025,3)
        rows.append(np.stack([mu,np.full(cfg['k'],np.log(cfg['fixed_sigma'])),np.zeros(cfg['k'])]))
    return jnp.asarray(np.stack(rows),dtype=jnp.float64)


class Trainer:
    def __init__(self,cfg):
        self.cfg=cfg
        x,m=quadrature(cfg,cfg['quad_points']);self.x=jnp.asarray(x);self.mass=jnp.asarray(m)
        self.opt=optax.adam(cfg['learning_rate']) if cfg['optimizer']=='adam' else optax.sgd(cfg['learning_rate'])
        p=initialize(cfg);self.state=(p,self.opt.init(p))
        self.vg=jax.jit(jax.vmap(jax.value_and_grad(lambda p:loss(p,self.x,self.mass,cfg))))
        def step(state,_):
            p,os=state;vals,g=self.vg(p)
            updates,os=self.opt.update(g,os,p);p=optax.apply_updates(p,updates)
            if not cfg.get('unbounded',False):p=p.at[:,0].set(jnp.clip(p[:,0],-cfg['action_bound'],cfg['action_bound']))
            if cfg['variance']=='learned':p=p.at[:,1].set(jnp.clip(p[:,1],np.log(cfg['sigma_min']),np.log(cfg['sigma_max'])))
            else:p=p.at[:,1].set(np.log(cfg['fixed_sigma']))
            if cfg['weights']=='learned':p=p.at[:,2].set(p[:,2]-jnp.mean(p[:,2],axis=-1,keepdims=True))
            else:p=p.at[:,2].set(0)
            return (p,os),vals
        self.advance=jax.jit(lambda state,n:jax.lax.scan(step,state,None,length=n),static_argnums=(1,))


def configurations(config):
    out=[]
    for case in config['cases']:
        for k in config['components']:
            for variance in ['fixed','learned']:
                cfg=dict(config,**case,k=k,variance=variance,weights='equal',stage='main',initialization='uniform')
                cfg.pop('cases');cfg['name']=f"{case['id']}/K{k}_{variance}_equal";out.append(cfg)
        for k in [3,15]:
            cfg=dict(config,**case,k=k,variance='learned',weights='learned',stage='weight_control',initialization='uniform')
            cfg.pop('cases');cfg['name']=f"{case['id']}/K{k}_learned_learned";out.append(cfg)
    for init in ['uniform','bad_structure','good_structure']:
        cfg=dict(config,**config['cases'][0],k=3,variance='fixed',weights='equal',stage='paper_control',initialization=init,
                 unbounded=True,integration_bound=12.,optimizer='sgd',learning_rate=.01)
        cfg.pop('cases');cfg['name']=f'paper_control/K3_{init}';out.append(cfg)
    return out


def load_config():return json.loads(Path(__file__).with_name('config.json').read_text())
