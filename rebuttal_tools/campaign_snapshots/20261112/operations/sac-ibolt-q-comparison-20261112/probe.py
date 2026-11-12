"""Read-only common-input critic comparison. No learner or optimizer mutation."""
from pathlib import Path
import json,hashlib,time
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
import numpy as np
import gymnasium as gym
import gymnasium_robotics
import jax,jax.numpy as jnp
import flax.serialization
from omegaconf import OmegaConf
from ibolt.policy import IBOLTPolicy
from baselines.jax_sac import critic_values,sample_actor
P=Path(__file__).resolve().parent;spec=json.loads((P/'probe-spec.json').read_text())
assert jax.default_backend()=='cpu'
for f,digest in json.loads((P/'checkpoint-sha256.json').read_text()).items():assert hashlib.sha256((P/f).read_bytes()).hexdigest()==digest,f
def restore_tuples(value):
 if isinstance(value,dict):
  if value and all(str(k).isdigit() for k in value):return tuple(restore_tuples(value[str(i)]) for i in range(len(value)))
  return {k:restore_tuples(v) for k,v in value.items()}
 return value
@jax.jit
def ibolt_q(critic,obs,act):
 return critic.apply_fn({'params':critic.params,'batch_stats':critic.batch_stats},obs,act,train=False)[...,0]
@jax.jit
def sac_q(params,obs,act):return critic_values(params,obs,act)[...,0]
@jax.jit
def actor_logp(actor,obs,noise):return sample_actor(actor,obs,noise)[1][...,0]
def normalize(obs,n):
 return np.clip((np.clip(obs,-200,200)-np.asarray(n['mean']))/np.maximum(np.sqrt(np.asarray(n['variance'])),.01),-5,5).astype(np.float32)
tasks=[('fetch',t) for t in ['reach','push','pick-and-place','slide']]+[('adroit',t) for t in ['door','hammer','pen','relocate']]
banks={};results=[];started=time.time();gym.register_envs(gymnasium_robotics)
fetch_ids=dict(reach='FetchReach-v4',push='FetchPush-v4',slide='FetchSlide-v4',**{'pick-and-place':'FetchPickAndPlace-v4'})
for index,(family,task) in enumerate(tasks):
 rr=[r for r in spec['rows'] if (r['family'],r['task'])==(family,task)];assert rr
 if family=='fetch':
  env=gym.make(fetch_ids[task],reward_type='sparse')
  if task=='slide':env.unwrapped.distance_threshold=.20
  assert env.spec.max_episode_steps==50
  rng=np.random.default_rng(910000+index);keep=set(np.linspace(0,49,20,dtype=int));obs=[];act=[];rew=[];nobs=[]
  for ep in range(25):
   o,_=env.reset(seed=4100000+index*1000+ep)
   for t in range(50):
    a=rng.uniform(-1,1,4).astype(np.float32);n,r,terminated,truncated,info=env.step(a)
    assert r in (-1.,0.)
    if t in keep:
     obs.append(np.concatenate((o['observation'],o['desired_goal'])));act.append(a);rew.append(r);nobs.append(np.concatenate((n['observation'],n['desired_goal'])))
    o=n
  obs=np.asarray(obs,dtype=np.float32);act=np.asarray(act,dtype=np.float32)
  note=dict(kind='Fixed independent uniform-action environment rollouts',environment=fetch_ids[task],reset_seeds=list(range(4100000+index*1000,4100000+index*1000+25)),distance_threshold_m=float(env.unwrapped.distance_threshold),horizon=50)
  env.close()
 else:
  assert len({r['demo_sha256'] for r in rr})==1
  path=Path(rr[0]['demo_path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==rr[0]['demo_sha256']
  with np.load(path,allow_pickle=False) as d:
   offsets=d['offsets'];assert len(offsets)==26
   ix=np.concatenate([np.linspace(a,b-1,20,dtype=int) for a,b in zip(offsets[:-1],offsets[1:])])
   obs=np.asarray(d['observations'][ix],dtype=np.float32);act=np.asarray(d['actions'][ix],dtype=np.float32);rew=d['rewards'][ix];nobs=d['next_observations'][ix]
  assert np.all(np.isclose(rew,-.1)|np.isclose(rew,10.))
  note=dict(kind='Verified expert transitions:20 evenly spaced samples from each25 episode',dataset_sha256=rr[0]['demo_sha256'])
 assert len(obs)==len(act)==500
 np.savez_compressed(P/(family+'-'+task+'-bank.npz'),observations=obs,actions=act,next_observations=nobs,rewards=rew)
 banks[family+'-'+task]=dict(**note,samples=500,state_action_sha256=hashlib.sha256(obs.tobytes()+act.tobytes()).hexdigest())
 ref=next(r for r in rr if r['method']=='ibolt');cfgname='mujoco-config.json' if family=='fetch' else 'ibolt-config.json'
 cfg=OmegaConf.create(json.loads((P/ref['id']/cfgname).read_text()))
 assert cfg.alg.gamma==.99 and bool(cfg.alg.critic.get('layer_norm',False))==(family=='adroit')
 policy=IBOLTPolicy(gym.spaces.Box(-np.inf,np.inf,(obs.shape[1],),dtype=np.float32),gym.spaces.Box(-1.,1.,(act.shape[1],),dtype=np.float32),cfg=cfg)
 policy.build(jax.random.PRNGKey(971),lambda _:3e-4,3e-4)
 for r in rr:
  for chunk in r['chunks']:
   step=chunk['step'];folder=P/r['id'];state=folder/chunk['file'];x=obs
   if family=='fetch':
    norm=json.loads((folder/chunk['normalizer']).read_text());assert norm['steps']==step;x=normalize(obs,norm)
   bonus=None;alpha=None
   if r['method']=='ibolt':
    critic=flax.serialization.from_bytes(policy.qf_state,state.read_bytes());q=np.asarray(ibolt_q(critic,jnp.asarray(x),jnp.asarray(act)))
    updates=int(critic.step)
   else:
    raw=flax.serialization.msgpack_restore(state.read_bytes());params=restore_tuples(raw['critic']);q=np.asarray(sac_q(params,jnp.asarray(x),jnp.asarray(act)))
    alpha=0. if r['method']=='sac_zero' else float(np.asarray(jnp.exp(jnp.asarray(raw['log_alpha']))).reshape(-1)[0])
    actor=restore_tuples(raw['actor']);entropy=[]
    for draw in range(4):
     noise=jax.random.normal(jax.random.PRNGKey(20261112+draw),(len(obs),act.shape[-1]));lp=np.asarray(actor_logp(actor,jnp.asarray(x),noise));assert np.isfinite(lp).all();entropy.append(float((-alpha*lp).mean()))
    bonus=float(np.mean(entropy));updates=int(raw['updates'])
   assert updates==int((step-12800)*(.125 if family=='fetch' else .25)),(r['id'],step,updates)
   assert q.shape==(2,500) and np.isfinite(q).all()
   qmin=q.min(0);row=dict(id=r['id'],family=family,task=task,method=r['method'],seed=r['seed'],step=step,updates=updates,q_min_mean=float(qmin.mean()),q_min_p05=float(np.quantile(qmin,.05)),q_min_p95=float(np.quantile(qmin,.95)),q_min_min=float(qmin.min()),q_min_max=float(qmin.max()),twin_gap_mean=float(np.abs(q[0]-q[1]).mean()),alpha=alpha,mean_entropy_term=bonus)
   results.append(row)
   print(json.dumps(row),flush=True)
 del policy
report=dict(passed=True,backend='cpu',read_only=True,optimizer_updates=0,seeds=[0,1],ibolt='Original temperature-selected confirmation; N64/M64; DACER H/d=-0.9',q_semantics={'ibolt':'Unregularized discounted reward Q','sac_auto':'Soft Q; includes future -alpha log pi terms','sac_zero':'Unregularized discounted reward Q'},method='Identical500 raw state-action pairs per task; each Fetch checkpoint uses its own frozen normalizer; all paired comparisons use same training seed and checkpoint step. This is a frozen-input diagnostic, not historical training minibatch Q or success evaluation.',entropy_term_note='Mean -alpha*log pi on shared states with four fixed Gaussian draws; not the discounted future entropy contribution and cannot simply be subtracted from Q.',banks=banks,rows=results,elapsed_seconds=time.time()-started)
(P/'q-comparison.json').write_text(json.dumps(report,indent=2)+'\n');print('FINISHED',len(results),flush=True)
