"""Out-of-construction test on a trained native twin CrossQ critic, on the GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from models.critic import CrossQVectorCritic
from mechanism import CENTER, SIGMA, TEMPERATURE, objectives, reward, summarize

BATCH=256
WIDTH=64
SEEDS=12
STAGES=(0,5,50,250)
NET=CrossQVectorCritic(net_arch=(WIDTH,WIDTH),n_critics=2,momentum=.99)
CAL=CrossQVectorCritic(net_arch=(WIDTH,WIDTH),n_critics=2,momentum=0.)
OPT=optax.adam(.001,b1=.5)


def joint_batch(key,regime,batch):
    k1,k2,k3,k4=jax.random.split(key,4)
    x=jax.random.uniform(k1,(batch,))
    narrow=.05+ .95*x
    wide=2*x-1
    replay=jnp.where(jax.random.uniform(k2,(batch,))<.85,regime*narrow,wide)
    signs=2*jax.random.bernoulli(k3,shape=(batch,))-1
    successor=jnp.clip(signs*CENTER+SIGMA*jax.random.normal(k4,(batch,)),-1,1)
    a=jnp.concatenate([replay,successor])[:,None]
    return jnp.zeros_like(a),a,reward(replay)


@jax.jit
def initialize(key):
    obs,a,_=joint_batch(key,1,BATCH)
    variables=NET.init(key,obs,a,train=True)
    return variables['params'],variables['batch_stats'],OPT.init(variables['params'])


def train_step(state,key,regime):
    p,stats,opt=state
    obs,a,r=joint_batch(key,regime,BATCH)
    def loss(params):
        out,updated=NET.apply({'params':params,'batch_stats':stats},obs,a,train=True,mutable=['batch_stats'])
        # Terminal bandit. The next half contributes to joint BRN statistics.
        err=out[:,:BATCH,0]-r
        return (err**2).mean(axis=1).sum(),updated['batch_stats']
    (_,stats),g=jax.value_and_grad(loss,has_aux=True)(p)
    updates,opt=OPT.update(g,opt,p)
    return optax.apply_updates(p,updates),stats,opt


@jax.jit
def train_block(state,keys,regime):
    def step(s,k):
        return train_step(s,k,regime),None
    return jax.lax.scan(step,state,keys)[0]


@jax.jit
def aligned_stats(p,stats,key,regime):
    obs,a,_=joint_batch(key,regime,8192)
    _,changed=CAL.apply({'params':p,'batch_stats':stats},obs,a,train=True,mutable=['batch_stats'])
    return changed['batch_stats']


def gaussian_sac(q,theta,key):
    # A single tanh-Gaussian as in SAC. Bulk policy spans the hinge.
    noise=jax.random.normal(key,(16384,))
    pre=theta+.5*noise
    a=jnp.tanh(pre)
    log_pi=-.5*noise**2-jnp.log(.5)-.5*jnp.log(2*jnp.pi)-jnp.log1p(-a*a+1e-6)
    return (q(a)-TEMPERATURE*log_pi).mean()


def readings(q,qmin,key):
    v=jnp.array([-CENTER,CENTER])
    pair=q(v)
    # Same actions/entropy/actor parameter for the iBOLT and matched pathwise controls.
    gi=jax.grad(lambda th:-objectives(q,key,th,64,'ibolt')[0])(0.)
    gp=jax.grad(lambda th:-objectives(q,key,th,64,'pathwise')[0])(0.)
    gs=jax.grad(lambda th:gaussian_sac(qmin,th,key))(0.)
    grid=jnp.linspace(-1,1,1001)
    return jnp.stack([jnp.sqrt(jnp.mean((q(grid)-reward(grid))**2)),
                      pair[1]-pair[0],jax.nn.sigmoid((pair[1]-pair[0])/TEMPERATURE),gi,gp,gs])


@jax.jit
def evaluate(p,stats,key):
    def values(a):
        flat=a.reshape(-1,1)
        return NET.apply({'params':p,'batch_stats':stats},jnp.zeros_like(flat),flat,train=False)[...,0].reshape((2,)+a.shape)
    return readings(lambda a:values(a).mean(0),lambda a:values(a).min(0),key)


def run():
    dev=jax.devices()[0];assert dev.platform=='gpu' and jax.default_backend()=='gpu'
    raw={}
    # Switch the replay distribution after the first 1000 optimizer updates.
    # Keep the same frozen actor for collecting successor actions in both regimes.
    for seed in range(SEEDS):
        key=jax.random.PRNGKey(seed)
        init,old,new,cal,measure=jax.random.split(key,5)
        state=train_block(initialize(init),jax.random.split(old,1000),1.)
        keys=jax.random.split(new,STAGES[-1])
        previous=0
        oracle=jax.jit(lambda k:readings(reward,reward,k))(measure)
        for stage in STAGES:
            if stage>previous: state=train_block(state,keys[previous:stage],-1.)
            previous=stage
            p,stats,_=state
            matched=aligned_stats(p,stats,cal,-1.)
            old_matched=aligned_stats(p,stats,cal,1.)
            for arm,s in [('running',stats),('new_batch',matched),('old_batch',old_matched)]:
                metrics=evaluate(p,s,measure)
                raw.setdefault(f'{stage}/{arm}',[]).append(np.asarray(metrics).tolist())
        raw.setdefault('oracle',[]).append(np.asarray(oracle).tolist())
        print('completed seed',seed,flush=True)
    names=('q_grid_rmse','right_minus_left','pair_right_mass','ibolt_ascent','matched_pathwise_ascent','gaussian_sac_ascent')
    summary={key:{name:summarize(col) for name,col in zip(names,np.array(rows).T)} for key,rows in raw.items()}
    return dict(gpu=dev.device_kind,device=str(dev),jax=jax.__version__,seeds=SEEDS,critic_width=WIDTH,
                batch_size=BATCH,shift_after=1000,post_shift_steps=STAGES,stats_intervention='hold weights fixed, replace BRN stats with moments on 8192 replay + 8192 successor samples',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                dependency_sha256=hashlib.sha256((Path(__file__).parent/'mechanism.py').read_bytes()).hexdigest(),
                summary=summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    f=parser.parse_args().output
    r=run()
    tmp=f.with_suffix('.tmp')
    tmp.write_text(json.dumps(r,indent=2,allow_nan=False))
    os.replace(tmp,f)
