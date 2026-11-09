"""Private read-only Fetch failure diagnostics; no training or source changes."""
from pathlib import Path
import os,sys,json,time,pickle,gc
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='1',JAX_PLATFORMS='cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false')
import numpy as np
ROOT=Path('/workspace/iBOLT')
CAM=ROOT/'outputs/fetch-rebuttal/pick-and-place-20261108T140900Z/host1'
OUT=ROOT/'outputs/fetch-rebuttal/diagnostic-20261109'
OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(CAM/'source'))
def save(name,value):
 (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def emit(value):print(json.dumps(value,allow_nan=False),flush=True)
def replay():
 rows=[]
 for seed in range(5):
  run=CAM/'runs'/f'pick-and-place-ibolt-T0.1-s{seed}'
  b=pickle.load((run/'checkpoints/final-replay.pkl').open('rb'))
  assert b.full and b.pos==0 and b.size()==1000000
  x=b.observations[:,0];y=b.next_observations[:,0];a=b.actions[:,0];r=b.rewards[:,0]
  initial=x[::50];init_hit=np.linalg.norm(initial[:,3:6]-initial[:,-3:],axis=1)<.05
  hit=(r==0).reshape(-1,50);anyhit=hit.any(1);finalhit=hit[:,-1]
  air=initial[:,-1]>.47
  lift=(y[:,5]>.47).reshape(-1,50).any(1)
  close=(np.linalg.norm(x[:,6:9],axis=1)<.04).reshape(-1,50).any(1)
  dist=np.linalg.norm(y[:,3:6]-y[:,-3:],axis=1)
  reward_expected=-(dist>.05).astype(np.float32)
  continuous=np.ones(len(x)-1,bool);continuous[49::50]=False
  row=dict(seed=seed,transitions=int(b.size()),episodes=len(initial),initial_success_episodes=int(init_hit.sum()),any_success_episodes=int(anyhit.sum()),final_success_episodes=int(finalhit.sum()),new_success_episodes=int((anyhit&~init_hit).sum()),new_success_air_episodes=int((anyhit&~init_hit&air).sum()),new_success_table_episodes=int((anyhit&~init_hit&~air).sum()),lift_episodes=int(lift.sum()),gripper_within4cm_episodes=int(close.sum()),reward_mismatches=int(np.count_nonzero(r!=reward_expected)),within_episode_goal_max_change=float(np.abs(x[:,-3:]-y[:,-3:]).max()),next_observation_link_max_error=float(np.max(np.abs(y[:-1][continuous]-x[1:][continuous]))),timeouts=int(b.timeouts.sum()),raw_dones=int(b.dones.sum()),effective_terminal_masks=int((b.dones*(1-b.timeouts)).sum()),action_std=np.std(a,axis=0).tolist(),action_boundary_fraction=np.mean(np.abs(a)>.99,axis=0).tolist(),windows=[])
  for start,end in [(0,5000),(5000,100000),(100000,200000),(900000,1000000)]:
   sl=slice(start//50,end//50);hr=hit[sl];ih=init_hit[sl];aa=air[sl]
   row['windows'].append(dict(start=start,end=end,episodes=len(ih),initial_success=int(ih.sum()),new_success=int((hr.any(1)&~ih).sum()),new_air_success=int((hr.any(1)&~ih&aa).sum()),lift_episodes=int(lift[sl].sum()),reward_zero_fraction=float((r[start:end]==0).mean()),mean_return=float(hr.sum(1).mean()-50)))
  rows.append(row);emit(row);save('replay.json',rows)
  row.update(new_final_success_episodes=int((finalhit&~init_hit).sum()),new_final_air_success_episodes=int((finalhit&~init_hit&air).sum()),reward_zero_transitions=int(hit.sum()),reward_zero_from_initially_failed=int(hit[~init_hit].sum()),reward_zero_from_initially_failed_air=int(hit[~init_hit&air].sum()),air_any_success_events=[dict(first_transition=int(ep*50+np.flatnonzero(hit[ep])[0]+1),goal=initial[ep,-3:].tolist(),episode_final_success=bool(finalhit[ep])) for ep in np.flatnonzero(anyhit&~init_hit&air)])
  save('replay.json',rows)
  if seed==0:
   rng=np.random.default_rng(119);ids=np.sort(rng.choice(len(x),256,replace=False))
   success_ids=np.flatnonzero(r==0);sid=rng.choice(success_ids,128,replace=False)
   np.savez(OUT/'probes.npz',obs=x[ids],actions=a[ids],next_obs=y[ids],rewards=r[ids],success_obs=x[sid],success_actions=a[sid],initial_obs=initial[:128])
  del b,x,y,a,r;gc.collect()
def scripted():
 from fetch_rebuttal.env import make_env
 env=make_env('pick-and-place');rows=[];states=[];expert_actions=[];phases=[]
 for ep in range(30):
  obs,_=env.reset(seed=900000+ep);initial=float(np.linalg.norm(obs[3:6]-obs[-3:])<.05);phase=0;counter=0;total=0;hit=False;maxz=0;trans=[]
  for step in range(50):
   grip=obs[:3];obj=obs[3:6];goal=obs[-3:];gripper=1.
   if phase==0:
    target=obj+np.array([0,0,.07]);
    if np.linalg.norm(target-grip)<.012:phase=1;trans.append(step)
   if phase==1:
    target=obj.copy()
    if np.linalg.norm(target-grip)<.012:phase=2;counter=0;trans.append(step)
   if phase==2:
    target=obj.copy();gripper=-1.;counter+=1
    if counter>=7:phase=3;trans.append(step)
   if phase==3:target=goal;gripper=-1.
   action=np.r_[20*(target-grip),gripper].astype(np.float32)
   states.append(obs.copy());expert_actions.append(np.clip(action,-1,1));phases.append(phase)
   obs,r,te,tr,info=env.step(np.clip(action,-1,1));total+=r;hit|=bool(info['is_success']);maxz=max(maxz,float(obs[5]))
  rows.append(dict(episode=ep,initial_success=initial,any_success=hit,final_success=bool(info['is_success']),return_=total,max_object_z=maxz,final_distance=float(np.linalg.norm(obs[3:6]-obs[-3:])),phase=phase,phase_steps=trans))
 env.close();np.savez(OUT/'scripted-probes.npz',obs=np.asarray(states),actions=np.asarray(expert_actions),phases=np.asarray(phases));save('scripted.json',rows);emit(dict(scripted_episodes=len(rows),initial_success=sum(r['initial_success'] for r in rows),final_success=sum(r['final_success'] for r in rows),any_success=sum(r['any_success'] for r in rows),new_success=sum(r['any_success'] and not r['initial_success'] for r in rows),lift=sum(r['max_object_z']>.47 for r in rows)))
def network():
 import jax,jax.numpy as jnp
 from omegaconf import OmegaConf
 from exploration import ExplorationIBOLT
 from fetch_rebuttal.env import make_env
 from ibolt.policy import IBOLTPolicy
 from ibolt.box_gaussian import mixture_log_prob,sample_box
 data=np.load(OUT/'probes.npz');rng=np.random.default_rng(42)
 # No collector, optimizer update, or checkpoint write is invoked.
 cfg=OmegaConf.create(json.loads((CAM/'runs/pick-and-place-ibolt-T0.1-s0/mujoco-config.json').read_text()));cfg.require_gpu=False
 model=ExplorationIBOLT('MlpPolicy',env=make_env('pick-and-place'),cfg=cfg,model_save_path=None,save_every_n_steps=0)
 rows=[]
 def qvals(state,obs,acts):
  return state.apply_fn({'params':state.params,'batch_stats':state.batch_stats},jnp.asarray(obs),jnp.asarray(acts),train=False)[...,0]
 for temp,seed,step in [(0.1,0,5001),(0.1,0,100000),(0.1,0,1000000),(0.1,1,1000000),(0.05,0,1000000),(1,0,1000000),(10,0,1000000)]:
  run=CAM/'runs'/f'pick-and-place-ibolt-T{temp:g}-s{seed}'
  model.load_model(run/'checkpoints',step,step);actor=model.policy.actor_state;critic=model.policy.qf_state
  initial=data['initial_obs'][:64];B=len(initial);N=64;D=4
  obs=jnp.broadcast_to(jnp.asarray(initial)[:,None],(B,N,28)).reshape(B*N,28)
  latent=jax.random.normal(jax.random.key(991),(B*N,D));mu,ls=actor.apply_fn({'params':actor.params},obs,latent)
  center=np.asarray(mu).reshape(B,N,D);std=np.exp(np.asarray(ls)).reshape(B,N,D)
  # Mirror the learner candidate streams and density correction exactly.
  key,latent_key,proposal_key,dropout_key=jax.random.split(jax.random.key(42),4);zk,_=jax.random.split(latent_key)
  latent=jax.random.normal(zk,(B,N,D));mu,ls=actor.apply_fn({'params':actor.params},obs,latent.reshape(B*N,D));mu=mu.reshape(B,N,D);ls=ls.reshape(B,N,D)
  ck,nk=jax.random.split(proposal_key);ids=jax.random.randint(ck,(B,N),0,N)
  candidates=sample_box(nk,jnp.take_along_axis(mu,ids[:,:,None],axis=1),jnp.take_along_axis(ls,ids[:,:,None],axis=1))
  q=qvals(critic,obs,candidates.reshape(B*N,D)).reshape(2,B,N);meanq=q.mean(0);density=mixture_log_prob(candidates,mu,ls)
  weights=jax.nn.softmax(meanq/temp-density,axis=-1);wdensity=jax.nn.softmax(-density,axis=-1)
  qu=np.asarray(q);w=np.asarray(weights);diff=np.asarray(weights-wdensity)
  # State-controlled goal counterfactual: object position vs a distant target.
  pp=data['obs'].copy();aa=data['actions'];base=np.asarray(qvals(critic,pp,aa)).mean(0)
  pp[:,-3:]=pp[:,3:6];near=np.asarray(qvals(critic,pp,aa)).mean(0)
  pp[:,-3:]=pp[:,3:6]+np.array([.15,.15,.2]);far=np.asarray(qvals(critic,pp,aa)).mean(0)
  obsbatch=data['obs'];nxt=data['next_obs'];act=data['actions'];rr=data['rewards']
  qa=np.asarray(qvals(critic,obsbatch,act));nextact=IBOLTPolicy.sample_action(actor,jnp.asarray(nxt),jax.random.key(72));targetstate=critic.replace(params=critic.target_params,batch_stats=critic.target_batch_stats);target=rr+.99*np.asarray(qvals(targetstate,nxt,nextact)).min(0)
  row=dict(temperature=temp,seed=seed,checkpoint=step,actor_optimizer_step=int(actor.step),critic_optimizer_step=int(critic.step),actor_center_mean=center.mean((0,1)).tolist(),actor_center_latent_sd=center.std(1).mean(0).tolist(),actor_center_abs_gt_09=(np.abs(center)>.9).mean((0,1)).tolist(),conditional_std_mean=float(std.mean()),conditional_std_min=float(std.min()),candidate_q_mean=float(qu.mean()),candidate_q_within_state_sd_mean=float(np.asarray(meanq).std(1).mean()),candidate_q_within_state_range_mean=float(np.ptp(np.asarray(meanq),axis=1).mean()),twin_q_abs_disagreement=float(np.abs(qu[0]-qu[1]).mean()),teacher_ess_mean=float((1/(w*w).sum(1)).mean()),teacher_ess_median=float(np.median(1/(w*w).sum(1))),teacher_q_contribution_mean_L1=float(np.abs(diff).sum(1).mean()),q_goal_near_mean=float(near.mean()),q_goal_far_mean=float(far.mean()),q_goal_near_minus_far_mean=float((near-far).mean()),q_goal_near_better_fraction=float((near>far).mean()),replay_q_mean=float(qa.mean()),sample_td_rmse=float(np.sqrt(((qa-target[None])**2).mean())),parameters_finite=bool(all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves((actor.params,critic.params)))))
  rows.append(row);save('network.json',rows);emit(row)
  row.update(replay_q_below_lower_bound_fraction=float((qa < -100).mean()),replay_q_above_upper_bound_fraction=float((qa > 0).mean()),replay_q_min=float(qa.min()),replay_q_max=float(qa.max()))
  expected=center.mean(1)[:,:3];toward_object=initial[:,3:6]-initial[:,:3]
  cosine=np.sum(expected*toward_object,axis=1)/(np.linalg.norm(expected,axis=1)*np.linalg.norm(toward_object,axis=1)+1e-12)
  row.update(expected_initial_motion_toward_object_cosine_mean=float(cosine.mean()),expected_initial_motion_away_fraction=float((cosine<0).mean()))
  scripted_data=OUT/'scripted-probes.npz'
  if scripted_data.exists():
   sp=np.load(scripted_data);idx=np.arange(0,len(sp['obs']),5);xx=sp['obs'][idx];expert=sp['actions'][idx];phase=sp['phases'][idx];bs=len(xx)
   random_actions=rng.uniform(-1,1,(bs,64,4)).astype(np.float32)
   qrandom=np.asarray(qvals(critic,np.repeat(xx,64,axis=0),random_actions.reshape(-1,4))).mean(0).reshape(bs,64)
   qe=np.asarray(qvals(critic,xx,expert)).mean(0);qa0=np.asarray(qvals(critic,xx,-expert)).mean(0)
   percentile=(qrandom<=qe[:,None]).mean(1)
   row['scripted_action_probe']=[dict(phase=int(ph),n=int((phase==ph).sum()),expert_action_q_percentile_mean=float(percentile[phase==ph].mean()),expert_minus_opposite_q_mean=float((qe-qa0)[phase==ph].mean())) for ph in np.unique(phase)]
  save('network.json',rows)
 model.get_env().close()
if __name__=='__main__':
 {'replay':replay,'scripted':scripted,'network':network}[sys.argv[1]]()
