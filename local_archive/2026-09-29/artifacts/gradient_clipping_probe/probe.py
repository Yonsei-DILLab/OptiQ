"""Read saved checkpoints; collect a small diagnostic pool; no model training/writes."""
import argparse,json,sys,time,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--campaign',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--out',required=True);args=p.parse_args()
sys.path.insert(0,args.source)
import jax,jax.numpy as jnp,optax,gymnasium as gym
from flax.serialization import from_bytes,to_bytes
from omegaconf import OmegaConf
from run_optiq_dime import create_algorithm
from optiq_dime.algorithm import OptiQDIME
from optiq_dime.policy import OptiQPolicy
out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
run=next(p for p in (Path(args.campaign)/'runs').glob('*/config.json') if json.loads(p.read_text())['seed']==args.seed)
cfg=OmegaConf.create(json.loads(run.read_text()));cfg.output_root=str(out/'runtime');cfg.alg.buffer_size=4096
model,cbs=create_algorithm(cfg);a=cfg.alg.actor;c=cfg.alg.critic
checkpoint_dir=next((run.parent/'checkpoints').iterdir())
steps=sorted(int(x.stem.split('_')[-1]) for x in checkpoint_dir.glob('actor_state_*.msgpack'))
latest=max(s for s in steps if (checkpoint_dir/f'critic_state_{s}.msgpack').exists())
selected=sorted(set([min(steps), min(steps,key=lambda s:abs(s-50000)), min(steps,key=lambda s:abs(s-latest/2)),latest]))
actor_template=model.policy.actor_state;qf_template=model.policy.qf_state
def load(kind,step,template):return from_bytes(template,(checkpoint_dir/f'{kind}_state_{step}.msgpack').read_bytes())
actor=load('actor',latest,actor_template);critic=load('critic',latest,qf_template)
orig_hash=hashlib.sha256(to_bytes(actor)+to_bytes(critic)).hexdigest()
env=gym.make(cfg.env_name);data=[];key=jax.random.PRNGKey(712000+args.seed);start=time.time()
try:
 for step in selected:
  behavior=load('actor',step,actor_template);obs,_=env.reset(seed=730000+step+args.seed)
  for i in range(512):
   key,ak=jax.random.split(key)
   action=np.asarray(OptiQPolicy.sample_action(behavior,jnp.asarray(obs[None],dtype=jnp.float32),ak,False,True))[0]
   native=env.action_space.low+(action+1)*.5*(env.action_space.high-env.action_space.low)
   nxt,reward,terminated,truncated,info=env.step(native)
   data.append((obs.copy(),action.copy(),nxt.copy(),float(reward),float(terminated)))
   obs=nxt
   if terminated or truncated:obs,_=env.reset()
  print(json.dumps({'phase':'collected','seed':args.seed,'policy_step':step,'transitions':len(data),'seconds':time.time()-start}),flush=True)
finally:env.close()
arrays=[np.asarray([d[i] for d in data],dtype=np.float32) for i in range(5)]
# Capture the exact gradients passed to the existing optimizer without applying them.
def capture_init(params):return jax.tree.map(jnp.zeros_like,params)
def capture_update(grads,state,params=None):return jax.tree.map(jnp.zeros_like,grads),grads
capture=optax.GradientTransformation(capture_init,capture_update)
actor_probe=actor.replace(tx=capture,opt_state=capture.init(actor.params))
critic_probe=critic.replace(tx=capture,opt_state=capture.init(critic.params))
z_atoms=jnp.linspace(c.v_min,c.v_max,c.n_atoms)
actor_kwargs={k:a[k] for k in ['num_policy_samples','proposals_per_policy_sample','proposal_sampling_mode','proposal_std','proposal_clip','include_anchor','density_correction','density_beta','adaptive_density_beta','minimum_source_ess','density_beta_grid_size','temperature','sinkhorn_epsilon','sinkhorn_iterations','source_q_eval','transport_target_mode']}
actor_kwargs.update(semi_implicit=True,normalize_ot_cost=a.get('normalize_ot_cost',True),distillation_loss=a.distillation_loss,teacher_distribution=a.teacher_distribution,soft_proximal_ess_fraction=a.get('soft_proximal_ess_fraction',0.),entropy_diagnostics=a.get('entropy_diagnostics',True),ot_student_action=a.ot_student_action)
@jax.jit
def probe(astate,qstate,original_a,original_q,obs,actions,nxt,rewards,dones,key):
 ka,kc=jax.random.split(key)
 ag,aloss,_,am=OptiQDIME.update_actor(astate,original_q,obs,ka,z_atoms,**actor_kwargs)
 qg,qm,_=OptiQDIME.update_critic(model.crossq_style,model.use_bnstats_from_live_net,model.gamma,original_a,qstate,obs,actions,nxt,rewards,dones,c.n_atoms,z_atoms,c.v_min,c.v_max,c.entr_coeff,a.td_noise_std,a.td_noise_clip,kc,True,int(a.get('entropy_samples',16)),a.temperature,model.backup_mode)
 return jnp.array([optax.global_norm(ag.opt_state),optax.global_norm(qg.opt_state),aloss,qm['critic_loss']])
rng=np.random.default_rng(90000+args.seed);records=[]
for i in range(128):
 ix=rng.integers(0,len(data),256);key,k=jax.random.split(key)
 values=np.asarray(probe(actor_probe,critic_probe,actor,critic,*[jnp.asarray(x[ix]) for x in arrays],k)).tolist()
 assert np.isfinite(values).all(),values
 records.append(values)
 if i in [0,31,127]:print(json.dumps({'phase':'gradients','seed':args.seed,'batches':i+1,'values':values,'seconds':time.time()-start}),flush=True)
assert orig_hash==hashlib.sha256(to_bytes(actor)+to_bytes(critic)).hexdigest()
candidates=([2,5,10,20,50,100,250,500,1000,2000] if cfg.env_name=='Ant-v4' else [2,10,25,50,100,200,500,1000,5000,10000,20000,50000])
summary={}
for idx,kind in enumerate(['actor','critic']):
 norms=np.asarray(records)[:,idx]
 summary[kind]={'min':float(norms.min()),'p50':float(np.quantile(norms,.5)),'p90':float(np.quantile(norms,.9)),'p95':float(np.quantile(norms,.95)),'p99':float(np.quantile(norms,.99)),'max':float(norms.max()),'rms':float(np.sqrt(np.mean(norms**2))),'candidates':{str(t):{'clipped_fraction':float(np.mean(norms>t)),'mean_scale':float(np.mean(np.minimum(1,t/np.maximum(norms,1e-30))))} for t in candidates}}
result={'seed':args.seed,'env':cfg.env_name,'source':args.source,'run_config':str(run),'checkpoint_step':latest,'behavior_checkpoint_steps':selected,'transitions':len(data),'batches':128,'batch_size':256,'sampling':'bootstrap from equal-length fresh full-policy rollouts at saved checkpoints; not original replay','parameters_unchanged':True,'seconds':time.time()-start,'summary':summary,'raw_rows_actor_norm_critic_norm_actor_loss_critic_loss':records}
(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({'phase':'done','summary':summary,'seconds':time.time()-start}),flush=True)
for cb in cbs.callbacks:
 if hasattr(cb,'eval_env'):cb.eval_env.close()
model.get_env().close();model.logger.close()
