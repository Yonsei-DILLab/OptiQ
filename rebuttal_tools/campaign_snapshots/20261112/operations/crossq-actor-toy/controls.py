"""Intervene on the BRN warmup gate and mean-shift magnitude with actual BRN code."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from models.critic import BatchRenorm
from mechanism import CENTER, SIGMA, TEMPERATURE, objectives, reward, summarize


def gaussian_grad(q,key):
    noise=jax.random.normal(key,(8192,))
    def obj(theta):
        a=jnp.tanh(theta+.5*noise)
        # Only the Q contribution differs when the critic is perturbed.
        return (q(a)+TEMPERATURE*jnp.log1p(-a*a+1e-6)).mean()
    return jax.grad(obj)(0.)


@jax.jit
def experiment(shift,warmed):
    u=(jnp.arange(256)+.5)/256
    replay=u-1
    eps=jax.scipy.special.ndtri((jnp.arange(128)+.5)/128)*SIGMA
    joint=jnp.concatenate([replay,CENTER+eps,-CENTER-eps])
    mu=joint.mean(); sd=jnp.sqrt(joint.var()+.001)
    train=BatchRenorm(use_running_average=False)
    infer=BatchRenorm(use_running_average=True)
    # Old context adds a constant to a hidden feature. Its stationary EMA has the
    # same variance as the new context and an offset measured in running SDs.
    variables=train.init(jax.random.PRNGKey(2),joint[:,None])
    stats=dict(variables['batch_stats'],mean=(mu+shift*sd)[None],
               var=joint.var()[None],steps=jnp.where(warmed,100000,99999))
    train_x,changed=train.apply({'params':variables['params'],'batch_stats':stats},
                                joint[:,None],mutable=['batch_stats'])
    # Fit a common hinge location from the joint mean and the readout to replay rewards.
    # Every evaluation below shares these fitted parameters.
    intercept=train_x.mean()
    x=train_x[:256,0]-intercept
    design=jnp.stack([jnp.ones_like(x),jax.nn.relu(-x),jax.nn.relu(x)],1)
    coef=jnp.linalg.lstsq(design,reward(replay),rcond=None)[0]
    def readout(x):
        return coef[0]+coef[1]*jax.nn.relu(-x)+coef[2]*jax.nn.relu(x)
    slope=(train_x[200,0]-train_x[10,0])/(joint[200]-joint[10])
    def q_batch(a):
        return readout(slope*(a-joint[10])+train_x[10,0]-intercept)
    def q_run(a):
        x=infer.apply({'params':variables['params'],'batch_stats':changed['batch_stats']},a[...,None])
        return readout(x[...,0]-intercept)
    centers=jnp.array([-CENTER,CENTER])
    pair0=q_batch(centers);pair1=q_run(centers)
    def seed_metrics(seed):
        key=jax.random.PRNGKey(seed)
        def shifts(q):
            gi=-jax.grad(lambda th:objectives(q,key,th,96,'ibolt')[0])(0.)
            gp=-jax.grad(lambda th:objectives(q,key,th,96,'pathwise')[0])(0.)
            return jnp.stack([gi,gp,gaussian_grad(q,key)])
        return shifts(q_run)-shifts(q_batch)
    changes=jax.lax.map(seed_metrics,jnp.arange(32))
    return pair0,pair1,jnp.max(jnp.abs(q_batch(joint)-reward(joint))),changes


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    path=parser.parse_args().output
    dev=jax.devices()[0];assert jax.default_backend()=='gpu' and dev.platform=='gpu'
    rows=[]
    for warm in [False,True]:
        for shift in (0.,1.,4.,5.,6.,8.):
            ref,run,error,changes=experiment(shift,warm)
            expected=ref[1]-ref[0];observed=run[1]-run[0]
            row=dict(warmup_passed=warm,old_mean_shift_sd=shift,train_pair=np.asarray(ref).tolist(),
                     inference_pair=np.asarray(run).tolist(),gap_error=float(observed-expected),
                     train_right_mass=float(jax.nn.sigmoid(expected/TEMPERATURE)),
                     inference_right_mass=float(jax.nn.sigmoid(observed/TEMPERATURE)),train_fit_error=float(error))
            row['gradient_difference']={name:summarize(v) for name,v in zip(
                ('ibolt','matched_pathwise','gaussian_sac'),np.asarray(changes).T)}
            rows.append(row)
            print(warm,shift,round(row['gap_error'],4),round(row['inference_right_mass'],4),flush=True)
    result=dict(gpu=dev.device_kind,device=str(dev),seeds=32,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                dependency_sha256=hashlib.sha256((Path(__file__).parent/'mechanism.py').read_bytes()).hexdigest(),rows=rows)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False));os.replace(tmp,path)
