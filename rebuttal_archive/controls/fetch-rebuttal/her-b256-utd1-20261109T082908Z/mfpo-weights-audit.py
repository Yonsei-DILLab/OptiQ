"""Read-only CPU diagnostic of existing Fetch MFPO checkpoints."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', OMP_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                  XLA_PYTHON_CLIENT_PREALLOCATE='false', MPLCONFIGDIR='/tmp/mfpo-her-audit-mpl')
import sys, pathlib, json, pickle, time
import numpy as np
base = pathlib.Path('/workspace/iBOLT/outputs/fetch-rebuttal/her-b256-utd1-20261109T082908Z/host2')
sys.path.insert(0, str(base/'source'))
import jax, jax.numpy as jnp, flax.serialization
from fetch_rebuttal.baselines.native import MFPO
from fetch_rebuttal.env import make_env

output = pathlib.Path('/tmp/mfpo-her-weights.json')
result = dict(started=time.time(), backend=jax.default_backend(), checkpoints=[], evaluations=[])
def save(): output.write_text(json.dumps(result, indent=2, allow_nan=False))
def stats(x):
    a=np.asarray(x);return dict(mean=float(a.mean()), std=float(a.std()), min=float(a.min()), max=float(a.max()))
holder=MFPO(0, pathlib.Path('/tmp'), 1000000, 28, batch_size=256, action_dim=4)
from jaxrl5.networks.mean_flow import action_sampler, action_sampler_with_logp
template=holder.agent
sampler=jax.jit(lambda a,o,n: action_sampler(a.actor.apply_fn,a.actor.params,a.T,n,o,False))
withlogp=jax.jit(lambda a,o,n: action_sampler_with_logp(a.actor.apply_fn,a.actor.params,a.logp_mvel.apply_fn,a.logp_mvel.params,a.T,n,o,a.clip_sampler))
value=jax.jit(lambda a,o,u:a.calc_value(o,u))
env=make_env('pick-and-place')
raw=np.stack([env.reset(seed=900000+i)[0] for i in range(100)])
env.close()
from jaxrl5.networks.mean_flow import tr_sampler
rows=[]
for seed in range(5):
 run=base/'runs'/f'pick-and-place-mfpo-s{seed}'
 step=300000;path=run/'checkpoints'/f'policy_{step:09d}.msgpack'
 agent=flax.serialization.from_bytes(template,path.read_bytes())
 norm=pickle.load((path.parent/f'normalizer_{step:09d}.pkl').open('rb'))
 obs=norm.normalize_obs(raw).astype(np.float32);B=len(obs);K1=16;K2=32
 key=jax.random.PRNGKey(2741);key,k=jax.random.split(key);t,r=tr_sampler(k,B,agent.time_dist_name,agent.data_proportion)
 key,k=jax.random.split(key);noise=jax.random.normal(k,(B*K1,4))
 policy,logp=withlogp(agent,np.repeat(obs,K1,axis=0),noise);policy=policy.reshape(B,K1,4);logp=logp.reshape(B,K1)
 key,k=jax.random.split(key);noisy=(1-t)*policy[:,0]+t*jax.random.normal(k,(B,4))
 low=(noisy[:,None]-(1-t)[:,:,None])/t[:,:,None];high=(noisy[:,None]+(1-t)[:,:,None])/t[:,:,None]
 key,k=jax.random.split(key);tn=jax.random.truncated_normal(k,low,high,shape=(B,K2,4))
 key,k=jax.random.split(key);fallback=jnp.clip(jax.random.normal(k,(B,K2,4)),low,high);tn=jnp.where(jnp.isnan(tn),fallback,tn)
 clean=(noisy[:,None]-t[:,:,None]*tn)/(1-t)[:,:,None]
 cat=jnp.concatenate([policy,clean],axis=1)
 q=value(agent,np.repeat(obs,K1+K2,axis=0),cat.reshape(-1,4)).reshape(B,K1+K2)
 qp=q[:,:K1];qc=q[:,K1:];alpha=float(agent.temp.apply_fn({'params':agent.temp.params}))
 def weightstats(logw):
  w=np.asarray(jax.nn.softmax(logw,axis=1));return dict(ess_mean=float((1/(w*w).sum(1)).mean()),ess_median=float(np.median(1/(w*w).sum(1))),maxweight_mean=float(w.max(1).mean()),over099=float((w.max(1)>.99).mean()))
 row=dict(seed=seed,alpha=alpha,policy_q_range=stats(jnp.max(qp,1)-jnp.min(qp,1)),clean_q_range=stats(jnp.max(qc,1)-jnp.min(qc,1)),at_temperature={})
 kernel=jnp.sum((noisy[:,None]-(1-t[...,None])*policy)**2,axis=-1)/(2*t**2)
 for a in (alpha,.01,.1,1.):
  row['at_temperature'][str(a)]=dict(policy=weightstats(qp/a-kernel-logp),clean=weightstats(qc/a))
 rows.append(row);output.write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
