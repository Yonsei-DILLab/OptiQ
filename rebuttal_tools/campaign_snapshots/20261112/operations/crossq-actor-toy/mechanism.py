"""GPU-only terminal-bandit intervention with a fitted BRN/ReLU critic and paired actors.
Run with the rebuttal iBOLT checkout on PYTHONPATH. Specification and results are in report.md.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import freeze
from ibolt.box_gaussian import mixture_log_prob, sample_box
from ibolt.distillation import direct_gmm_nll
from models.critic import BatchRenorm

TEMPERATURE = 0.5
SIGMA = 0.055
CENTER = 0.65
N = 64
SEEDS = 32
STATES = 256
ACTOR_STEPS = 80
LR = 0.001
METHODS = ('ibolt', 'pathwise')
CONDITIONS = ('aligned', 'lagged', 'constant', 'recalibrated')


def reward(a):
    return 1.25*a - 2*jax.nn.relu(a + 0.25)


def measure(bridge, stats, batch):
    _, changed = bridge.apply({'params': bridge_params, 'batch_stats': stats},
                              batch[:, None], mutable=['batch_stats'])
    return changed['batch_stats']


def calibrate():
    # Replay is uniform on [-1,0]. Policy successor actions are drawn as an exactly
    # symmetric stratified sample. Reflection changes the mean but preserves variance.
    u = (jnp.arange(256) + 0.5) / 256
    replay = u - 1
    eps = jax.scipy.special.ndtri((jnp.arange(128) + 0.5)/128) * SIGMA
    successor = jnp.concatenate([CENTER+eps, -CENTER-eps])
    joint = jnp.concatenate([replay, successor])
    bridge = BatchRenorm(use_running_average=False)
    init = bridge.init(jax.random.PRNGKey(0), joint[:, None])
    global bridge_params
    bridge_params = init['params']
    def accumulate(stats, x, count):
        return jax.lax.fori_loop(0, count, lambda _, s: measure(bridge,s,x), stats)
    aligned = accumulate(init['batch_stats'], joint, 2048)
    old = accumulate(init['batch_stats'], -joint, 2048)
    lagged = measure(bridge, old, joint)
    restored = accumulate(lagged, joint, 2048)
    norm, _ = bridge.apply({'params': bridge_params,'batch_stats': lagged},
                           joint[:,None],mutable=['batch_stats'])
    x = norm[:len(replay),0]
    design = jnp.stack([jnp.ones_like(x),jax.nn.relu(-x),jax.nn.relu(x)],1)
    coef = jnp.linalg.lstsq(design, reward(replay), rcond=None)[0]
    infer = BatchRenorm(use_running_average=True)
    def make_q(stats, offset=0.):
        def q(a):
            v = infer.apply({'params':bridge_params,'batch_stats':stats},a[...,None])[...,0]
            return coef[0]+coef[1]*jax.nn.relu(-v)+coef[2]*jax.nn.relu(v)+offset
        return q
    qs = (make_q(aligned),make_q(lagged),make_q(aligned,1.),make_q(restored))
    grid = jnp.linspace(-1,1,2001)
    stats = {name:{'mean':float(s['mean'][0]), 'variance':float(s['var'][0])}
             for name,s in [('aligned',aligned),('lagged',lagged),('recalibrated',restored)]}
    stats['max_aligned_error'] = float(jnp.max(jnp.abs(qs[0](grid)-reward(grid))))
    stats['max_restored_error'] = float(jnp.max(jnp.abs(qs[0](grid)-qs[-1](grid))))
    stats['train_rmse'] = float(jnp.sqrt(jnp.mean((design@coef-reward(replay))**2)))
    return qs, stats


def samples(key, theta, batch):
    latent, choice, noise = jax.random.split(key,3)
    signs = 2*jax.random.bernoulli(latent,shape=(batch,N,1)).astype(jnp.float32)-1
    means = jnp.tanh(jnp.arctanh(signs*CENTER)+theta)
    logs = jnp.full_like(means,jnp.log(SIGMA))
    idx = jax.random.randint(choice,(batch,N),0,N)
    centers = jnp.take_along_axis(means, idx[...,None], axis=1)
    actions = sample_box(noise, centers, logs)
    return means,logs,actions


def objectives(q,key,theta,batch,method):
    means,logs,actions=samples(key,theta,batch)
    if method=='ibolt':
        actions=jax.lax.stop_gradient(actions)
    logp=mixture_log_prob(actions,means,logs)
    values=q(actions[...,0])
    weights=jax.nn.softmax(values/TEMPERATURE-logp,axis=1)
    if method=='ibolt':
        loss,_=direct_gmm_nll(means,logs,actions,weights)
    else:
        loss=(TEMPERATURE*logp-values).mean()
    right=(actions[...,0]>0)
    return loss, jnp.mean(jnp.sum(weights*right,axis=1))


def exact_policy_return(theta):
    # Quadrature over a reparameterized, box-truncated Gaussian for each latent.
    u=(jnp.arange(4096)+.5)/4096
    mu=jnp.tanh(jnp.arctanh(jnp.array([-CENTER,CENTER]))+theta)[:,None]
    lo=jax.scipy.special.ndtr((-1-mu)/SIGMA)
    hi=jax.scipy.special.ndtr((1-mu)/SIGMA)
    a=mu+SIGMA*jax.scipy.special.ndtri(lo+(hi-lo)*u)
    return reward(a).mean()


def summarize(x):
    x=np.asarray(x,dtype=float)
    return dict(mean=float(x.mean()),sd=float(x.std(ddof=1)),se=float(x.std(ddof=1)/np.sqrt(len(x))),
                positive=int((x>0).sum()),per_seed=x.tolist())


def run():
    device=jax.devices()[0]
    assert jax.default_backend()=='gpu' and device.platform=='gpu', device
    qs, calibration=calibrate()
    a=jnp.array([-CENTER,CENTER])
    assert a.device.platform=='gpu'
    exact=[]
    for q in qs:
        exact.append({'q':np.asarray(q(a)).tolist(),'dqda':np.asarray(jax.vmap(jax.grad(q))(a)).tolist(),
                      'right_mass':{str(t):float(jax.nn.softmax(q(a)/t)[1]) for t in (.25,.5,1.)}})
    native_ibolt=jax.jit(jax.value_and_grad(lambda th: exact_policy_return(th)))
    base, oracle_grad=map(float,native_ibolt(0.))
    stats={}
    # Separate compilation for each condition; common keys across every method/arm.
    for name,q in zip(CONDITIONS,qs):
        stats[name]={}
        for method in METHODS:
            def deriv(th,k,b):
                (_,mass),g=jax.value_and_grad(lambda p: objectives(q,k,p,b,method),has_aux=True)(th)
                return -g,mass
            def trial(seed):
                key=jax.random.PRNGKey(seed)
                grad,mass=deriv(0.,jax.random.fold_in(key,1777),STATES)
                def update(carry,i):
                    theta,m,v=carry
                    g,_=deriv(theta,jax.random.fold_in(key,i),32)
                    m=.5*m+.5*g
                    v=.999*v+.001*g**2
                    theta=theta+LR*(m/(1-.5**(i+1)))/(jnp.sqrt(v/(1-.999**(i+1)))+1e-8)
                    return (theta,m,v),None
                (theta,_,_),_=jax.lax.scan(update,(0.,0.,0.),jnp.arange(ACTOR_STEPS))
                return grad,mass,theta,exact_policy_return(theta)-base
            trial=jax.jit(jax.vmap(trial))
            vals=[[],[],[],[]]
            for first in range(0,SEEDS,4):
                block=jax.device_get(trial(jnp.arange(first,first+4)))
                for v,b in zip(vals,block):v.extend(b)
            stats[name][method]={k:summarize(v) for k,v in zip(
                ('initial_ascent_gradient','target_right_mass','final_theta','true_return_change'),vals)}
            print(name,method,[round(float(np.mean(v)),6) for v in vals],flush=True)
    return dict(gpu=device.device_kind,device=str(device),jax=jax.__version__,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                seeds=SEEDS,gradient_states=STATES,actions_per_state=N,adam_steps=ACTOR_STEPS,
                calibration=calibration,analytic_pair=exact,conditions=CONDITIONS,
                true_initial_return=base,true_initial_gradient=oracle_grad,results=stats)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    output=parser.parse_args().output
    result=run()
    temp=output.with_suffix('.tmp')
    temp.write_text(json.dumps(result,indent=2,allow_nan=False))
    os.replace(temp,output)
