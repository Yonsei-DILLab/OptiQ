
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR='/tmp/sac-diagnostic-mpl')
import pathlib,sys,json,time,numpy as np,torch
from stable_baselines3 import SAC
root=pathlib.Path('/workspace/iBOLT/outputs/fetch-rebuttal/sac-b2048-utd1-20261109T044224Z')
sys.path.insert(0,str(root/'source'))
from fetch_rebuttal.env import make_env
ckpt=root/'runs/pick-and-place-sac-b2048-s0/checkpoints/policy_000400000.zip'
torch.set_num_threads(1)
model=SAC.load(ckpt,device='cpu')
env=make_env('pick-and-place');model.policy.set_training_mode(False)
result={'checked_unix':time.time(),'checkpoint':str(ckpt),'protocol':'Read-only CPU diagnostic: same 100 environment seeds 700000..700099 per mode, horizon 50. No training or replay updates.', 'model':{'gamma':model.gamma,'tau':model.tau,'batch_size':model.batch_size,'gradient_steps':model.gradient_steps,'target_update_interval':model.target_update_interval,'target_entropy':model.target_entropy,'entropy_coefficient':float(model.log_ent_coef.exp().detach().cpu()),'num_updates':model._n_updates,'learning_rate':model.learning_rate,'use_sde':model.use_sde,'policy_class':type(model.policy).__name__,'actor_network':str(model.actor),'critic_network':str(model.critic),'optimizer_actor':str(model.actor.optimizer),'optimizer_critic':str(model.critic.optimizer),'parameters_finite':all(bool(torch.isfinite(p).all()) for p in model.policy.parameters())},'modes':{}}
for mode in ['deterministic','stochastic','zero_action']:
 torch.manual_seed(700000);episodes=[];traces=[];act_all=[];std_all=[];qs=[]
 for ep in range(100):
  obs,info=env.reset(seed=700000+ep);initial=obs.copy();d0=float(np.linalg.norm(obs[3:6]-obs[-3:]));records=[];total=0.;zero_rewards=0
  for step in range(50):
   if mode=='zero_action':action=np.zeros(4,dtype=np.float32)
   else:action=model.predict(obs,deterministic=mode=='deterministic')[0]
   if ep<10 and mode=='deterministic':
    with torch.no_grad():
     x=torch.as_tensor(obs[None]);mu,ls,_=model.actor.get_action_dist_params(x);std_all.append(ls.exp().cpu().numpy()[0].tolist());qs.append(float(torch.minimum(*model.critic(x,torch.as_tensor(action[None]))).item()))
   next_obs,reward,terminated,truncated,info=env.step(action);total+=float(reward);zero_rewards+=int(reward==0)
   rec={'t':step+1,'success':bool(info['is_success']),'distance':float(np.linalg.norm(next_obs[3:6]-next_obs[-3:])),'object_z':float(next_obs[5]),'object_displacement':float(np.linalg.norm(next_obs[3:6]-initial[3:6])),'gripper_object_distance':float(np.linalg.norm(next_obs[6:9])),'action':action.tolist()}
   records.append(rec);act_all.append(action.tolist());obs=next_obs
   if terminated or truncated:break
  hit=[r['t'] for r in records if r['success']]
  episodes.append({'seed':700000+ep,'initial_success':d0<.05,'initial_distance':d0,'initial_object_position':initial[3:6].tolist(),'goal':initial[-3:].tolist(),'any_success':bool(hit),'final_success':records[-1]['success'],'first_success_step':hit[0] if hit else None,'successful_steps':len(hit),'zero_reward_steps':zero_rewards,'return':total,'final_distance':records[-1]['distance'],'minimum_distance':min(r['distance'] for r in records),'max_object_lift':max(r['object_z']-float(initial[5]) for r in records),'max_object_displacement':max(r['object_displacement'] for r in records),'min_gripper_object_distance':min(r['gripper_object_distance'] for r in records)})
  if ep<10:traces.append({'seed':700000+ep,'steps':records})
 a=np.asarray(act_all);valid=[e for e in episodes if not e['initial_success']];high=[e for e in episodes if e['goal'][2]>=.50]
 summary={'episodes':100,'initial_success_count':sum(e['initial_success'] for e in episodes),'any_success_count':sum(e['any_success'] for e in episodes),'final_success_count':sum(e['final_success'] for e in episodes),'new_any_success_count':sum(e['any_success'] for e in valid),'new_final_success_count':sum(e['final_success'] for e in valid),'initially_failed_episodes':len(valid),'high_goal_episodes':len(high),'high_goal_any_success_count':sum(e['any_success'] for e in high),'high_goal_final_success_count':sum(e['final_success'] for e in high),'lift_over_5cm_episodes':sum(e['max_object_lift']>.05 for e in episodes),'object_moved_over_5cm_episodes':sum(e['max_object_displacement']>.05 for e in episodes),'gripper_within_4cm_episodes':sum(e['min_gripper_object_distance']<.04 for e in episodes),'mean_return':float(np.mean([e['return'] for e in episodes])),'zero_reward_steps':sum(e['zero_reward_steps'] for e in episodes),'action_mean':a.mean(0).tolist(),'action_std':a.std(0).tolist(),'boundary_fraction':(np.abs(a)>.99).mean(0).tolist(),'max_object_lift_max':max(e['max_object_lift'] for e in episodes)}
 if std_all:summary.update(gaussian_pretanh_std_mean=np.mean(std_all,axis=0).tolist(),critic_q_mean=float(np.mean(qs)),critic_q_min=float(np.min(qs)),critic_q_max=float(np.max(qs)))
 result['modes'][mode]={'summary':summary,'episodes':episodes,'traces_first10':traces}
 print(json.dumps({'mode':mode,'summary':summary}),flush=True)
env.close();dest=pathlib.Path('/tmp/sac-b2048-400k-diagnostic.json');dest.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'saved':str(dest),'model':result['model']}),flush=True)
