"""Read-only matched-checkpoint probes: no optimizer step and no training replay mutation."""
from pathlib import Path
import os,json,time,hashlib
import numpy as np
import jax,jax.numpy as jnp
import flax.serialization
from omegaconf import OmegaConf
from gymnasium import spaces
from ibolt.policy import IBOLTPolicy
from ibolt.box_gaussian import sample_box,mixture_log_prob
from fetch_rebuttal.env import make_env
P=Path(__file__).resolve().parent;spec=json.loads((P/'probe-spec.json').read_text())
assert jax.default_backend()=='gpu'
for name,digest in json.loads((P/'checkpoint-sha256.json').read_text()).items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest,name
cfg=OmegaConf.create(json.loads((P/spec['runs'][0]['id']/'mujoco-config.json').read_text()))
policy=IBOLTPolicy(spaces.Box(-np.inf,np.inf,(28,),dtype=np.float32),spaces.Box(-1.,1.,(4,),dtype=np.float32),cfg=cfg)
policy.build(jax.random.PRNGKey(15171),lambda _:3e-4,3e-4)
A,C=policy.actor_state,policy.qf_state

def load(run,step):
 p=P/run['id']/'checkpoints'
 a=flax.serialization.from_bytes(A,(p/f'actor_state_{step}.msgpack').read_bytes())
 c=flax.serialization.from_bytes(C,(p/f'critic_state_{step}.msgpack').read_bytes())
 normal=json.loads((p/f'normalizer_{step:09d}.json').read_text())
 return a,c,normal

def norm(raw,n):
 return np.clip((np.clip(raw,-200,200)-np.asarray(n['mean']))/np.maximum(np.sqrt(np.asarray(n['variance'])),.01),-5,5).astype(np.float32)

# Symmetric state bank: every checkpoint family/seed contributes the same reset seeds and times.
raws=[];origins=[];env=make_env('slide');assert env.unwrapped.distance_threshold==.20
for r in spec['runs']:
 a,c,normal=load(r,spec['steps'][-1]);rng=jax.random.PRNGKey(195302)
 for ep in range(4):
  obs,_=env.reset(seed=271000+ep)
  for t in range(50):
   if t in [0,7,14,21,28,35,42,49]:raws.append(obs.copy());origins.append(dict(run=r['id'],episode_seed=271000+ep,time=t))
   rng,key=jax.random.split(rng)
   action=np.asarray(IBOLTPolicy.sample_action(a,jnp.asarray(norm(obs[None],normal)),key,sample_conditional_noise=False))[0]
   obs,reward,terminated,truncated,info=env.step(action)
   assert reward in (-1.,0.)
   if terminated or truncated:assert t==49
 del a,c
env.close();raw=np.stack(raws);assert raw.shape==(128,28)
np.save(P/'raw-state-bank.npy',raw);(P/'state-bank-origins.json').write_text(json.dumps(origins,indent=2))
print('STATE_BANK_READY',raw.shape,flush=True)

@jax.jit
def candidates(actor,critic,obs,key):
 batch,odim=obs.shape;keys=jax.random.split(key,3)
 z=jax.random.normal(keys[0],(batch,64,4));ob=jnp.broadcast_to(obs[:,None],(batch,64,odim)).reshape(-1,odim)
 mu,ls=actor.apply_fn({'params':actor.params},ob,z.reshape(-1,4));mu=mu.reshape(batch,64,4);ls=ls.reshape(batch,64,4)
 ix=jax.random.randint(keys[1],(batch,256),0,64)
 act=sample_box(keys[2],jnp.take_along_axis(mu,ix[:,:,None],axis=1),jnp.take_along_axis(ls,ix[:,:,None],axis=1))
 ob=jnp.broadcast_to(obs[:,None],(batch,256,odim)).reshape(-1,odim)
 values=critic.apply_fn({'params':critic.params,'batch_stats':critic.batch_stats},ob,act.reshape(-1,4),train=False).reshape(2,batch,256,-1)[...,0]
 density=mixture_log_prob(act,mu,ls)
 return values,density,act

def softmax(v):
 a=np.exp(v-v.max(-1,keepdims=True));return a/a.sum(-1,keepdims=True)
def reduce(values,density,m):
 q=values[:,:,:m].mean(0).astype(np.float64);gap=np.abs(values[0,:,:m]-values[1,:,:m]).astype(np.float64);lp=density[:,:m].astype(np.float64)
 w=softmax(q/.1-lp);wq=softmax(q/.1);wd=softmax(-lp)
 ess=1/(w*w).sum(-1);idx=w.argmax(-1);topq=np.take_along_axis(q,idx[:,None],axis=-1)[:,0]
 return dict(ess_mean=float(ess.mean()),ess_median=float(np.median(ess)),ess_p10=float(np.quantile(ess,.1)),max_weight_mean=float(w.max(-1).mean()),q_only_ess_mean=float((1/(wq*wq).sum(-1)).mean()),density_only_ess_mean=float((1/(wd*wd).sum(-1)).mean()),q_mean=float(q.mean()),q_std_over_temperature=float(q.std(-1).mean()/.1),log_density_std=float(lp.std(-1).mean()),weighted_q_mean=float((w*q).sum(-1).mean()),weighted_twin_gap_mean=float((w*gap).sum(-1).mean()),unweighted_twin_gap_mean=float(gap.mean()),top_weight_q_positive_fraction=float((topq>0).mean()),top_weight_q_mean=float(topq.mean()),q_above_zero_fraction=float((q>0).mean()),q_below_minus100_fraction=float((q< -100).mean()))

rows=[];began=time.time()
for step in spec['steps']:
 for run in spec['runs']:
  a,c,normal=load(run,step);obs=jnp.asarray(norm(raw,normal));replicates=[]
  for draw in range(4):
   # Same latent mixture and nested candidate pool for counterfactual M64/M256.
   v,d,act=candidates(a,c,obs,jax.random.PRNGKey(20261112+draw));v,d=np.asarray(v),np.asarray(d)
   assert np.isfinite(v).all() and np.isfinite(d).all()
   replicates.append({str(m):reduce(v,d,m) for m in [64,256]})
  means={str(m):{k:float(np.mean([r[str(m)][k] for r in replicates])) for k in replicates[0][str(m)]} for m in [64,256]}
  row=dict(id=run['id'],seed=run['seed'],trained_M=run['M'],step=step,replicates=replicates,mean=means)
  rows.append(row)
  with (P/'checkpoint-probes.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
  print('PROBED',run['seed'],run['M'],step,means[str(run['M'])]['ess_mean'],flush=True)
  del a,c
out=dict(passed=True,backend=jax.default_backend(),read_only=True,optimizer_updates=0,state_bank_size=len(raw),state_bank_sha256=hashlib.sha256((P/'raw-state-bank.npy').read_bytes()).hexdigest(),method='128 shared raw states from equal rollouts of each trained policy at 1.5M; each checkpoint uses own frozen normalizer; 4 independent latent/proposal keys; M64 is prefix of same M256 candidates; no historical minibatch replay',elapsed_seconds=time.time()-began,rows=rows)
(P/'probe-results.json').write_text(json.dumps(out,indent=2,allow_nan=False));print('FINISHED',len(rows),flush=True)
