"""Frozen critic evaluation on identical verified expert transitions; no updates."""
from pathlib import Path
import json,hashlib,time
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
import numpy as np
import jax,jax.numpy as jnp
import flax.serialization
from gymnasium import spaces
from omegaconf import OmegaConf
from ibolt.policy import IBOLTPolicy
P=Path(__file__).resolve().parent;spec=json.loads((P/'probe-spec.json').read_text())
assert jax.default_backend()=='cpu'
for f,h in json.loads((P/'checkpoint-sha256.json').read_text()).items():assert hashlib.sha256((P/f).read_bytes()).hexdigest()==h,f
@jax.jit
def values(critic,obs,act):
 return critic.apply_fn({'params':critic.params,'batch_stats':critic.batch_stats},obs,act,train=False)[...,0]
out=[];banks={};began=time.time()
for task in ['door','hammer','pen','relocate']:
 rows=[r for r in spec['rows'] if r['task']==task];ref=rows[0]
 assert len({r['demo_sha256'] for r in rows})==1
 dataset=Path(ref['demo_path']);assert hashlib.sha256(dataset.read_bytes()).hexdigest()==ref['demo_sha256']
 with np.load(dataset,allow_pickle=False) as data:
  offsets=data['offsets'];assert len(offsets)==26
  indices=np.concatenate([np.linspace(a,b-1,20,dtype=int) for a,b in zip(offsets[:-1],offsets[1:])]);assert len(indices)==500 and len(set(indices))==500
  obs=np.asarray(data['observations'][indices],dtype=np.float32);act=np.asarray(data['actions'][indices],dtype=np.float32)
 banks[task]=dict(dataset_sha256=ref['demo_sha256'],transitions=500,indices=indices.tolist(),state_action_sha256=hashlib.sha256(obs.tobytes()+act.tobytes()).hexdigest())
 cfg=OmegaConf.create(json.loads((P/ref['id']/'ibolt-config.json').read_text()))
 assert cfg.alg.gamma==.99 and cfg.alg.critic.layer_norm and list(cfg.alg.critic.hs)==[256,256]
 policy=IBOLTPolicy(spaces.Box(-np.inf,np.inf,(obs.shape[1],),dtype=np.float32),spaces.Box(-1,1,(act.shape[1],),dtype=np.float32),cfg=cfg)
 policy.build(jax.random.PRNGKey(971),lambda _:3e-4,3e-4)
 for r in rows:
  for step in r['steps']:
   critic=flax.serialization.from_bytes(policy.qf_state,(P/r['id']/'checkpoints'/f'critic_state_{step}.msgpack').read_bytes())
   q=np.asarray(values(critic,jnp.asarray(obs),jnp.asarray(act)));assert np.isfinite(q).all()
   qmin=q.min(0);gap=np.abs(q[0]-q[1])
   v=dict(task=task,seed=r['seed'],N=r['N'],M=r['M'],id=r['id'],step=step,q_min_mean=float(qmin.mean()),q_min_p05=float(np.quantile(qmin,.05)),q_min_p95=float(np.quantile(qmin,.95)),q_min_max=float(qmin.max()),above1000_fraction=float((qmin>1000).mean()),below_minus10_fraction=float((qmin< -10).mean()),twin_gap_mean=float(gap.mean()))
   out.append(v);print(json.dumps(v),flush=True)
 del policy
report=dict(passed=True,backend='cpu',optimizer_updates=0,read_only=True,method='500 identical expert state-action pairs per task: 20 evenly spaced transitions from each of25 verified complete trajectories; no observation normalization in A1; min of twin critics; checkpoint diagnostic, not historical replay minibatch mean',banks=banks,rows=out,elapsed_seconds=time.time()-began)
(P/'adroit-checkpoint-q.json').write_text(json.dumps(report,indent=2)+'\n');print('FINISHED',len(out),flush=True)
