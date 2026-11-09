import argparse,json,os,time,traceback
from pathlib import Path
from functools import partial
import importlib.metadata
import numpy as np
import gymnasium as gym
import gymnasium_robotics

class CompactGoal(gym.ObservationWrapper):
 def __init__(self,env):
  super().__init__(env)
  dim=env.observation_space['observation'].shape[0]+3
  self.observation_space=gym.spaces.Box(-np.inf,np.inf,shape=(dim,),dtype=np.float32)
 def observation(self,obs):
  return np.concatenate([obs['observation'],obs['desired_goal']]).astype(np.float32)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--profile',required=True);args=parser.parse_args()
 started=time.time()
 report=dict(profile=args.profile,packages={p:importlib.metadata.version(p) for p in ['gymnasium-robotics','gymnasium','mujoco','stable-baselines3','numpy']},
  training_performed=False,environments=[],vectors=[],baseline_predict=[])
 if hasattr(gym,'register_envs'):gym.register_envs(gymnasium_robotics)
 ids={}
 for name in ['FetchReach','FetchPush','FetchSlide','FetchPickAndPlace']:
  candidates=[key for key in gym.envs.registry if key.startswith(name+'-v') and key.rsplit('-v',1)[1].isdigit()]
  env_id=max(candidates,key=lambda key:int(key.rsplit('-v',1)[1]));ids[name]=env_id
  env=gym.make(env_id)
  try:
   obs,info=env.reset(seed=123)
   shapes={k:list(v.shape) for k,v in obs.items()}
   assert env.action_space.shape==(4,)
   env.action_space.seed(17)
   term=trunc=False;values=[]
   for t in range(50):
    obs,reward,term,trunc,info=env.step(env.action_space.sample())
    assert all(np.isfinite(v).all() for v in obs.values()) and reward in (-1.,0.)
    assert float(env.unwrapped.compute_reward(obs['achieved_goal'],obs['desired_goal'],info))==reward
    assert 'is_success' in info
    values.append(float(reward))
   assert not term and trunc
   row=dict(id=env_id,observation_shapes=shapes,action_shape=list(env.action_space.shape),compact_goal_dim=shapes['observation'][0]+3,
    default_horizon=env.spec.max_episode_steps,terminated_at_50=bool(term),truncated_at_50=bool(trunc),sparse_reward_values=sorted(set(values)),success_info_available=True)
   report['environments'].append(row)
  finally:env.close()
  vec=gym.vector.SyncVectorEnv([partial(gym.make,env_id) for _ in range(4)])
  try:
   obs,info=vec.reset(seed=45)
   for t in range(52):obs,r,term,trunc,info=vec.step(np.zeros((4,4),np.float32))
   assert obs['observation'].shape[0]==4 and np.isfinite(obs['observation']).all()
   report['vectors'].append(dict(id=env_id,type='SyncVectorEnv',num_envs=4,steps=52,passed=True))
  finally:vec.close()
 vec=gym.vector.AsyncVectorEnv([partial(gym.make,ids['FetchPush']) for _ in range(4)],context='spawn')
 try:
  obs,info=vec.reset(seed=45)
  for t in range(52):obs,r,term,trunc,info=vec.step(np.zeros((4,4),np.float32))
  assert obs['observation'].shape==(4,25)
  report['vectors'].append(dict(id=ids['FetchPush'],type='AsyncVectorEnv',num_envs=4,steps=52,passed=True))
 finally:vec.close()
 try:
  env=gym.make(ids['FetchReach'],render_mode='rgb_array',width=160,height=120)
  try:
   env.reset(seed=5);frame=env.render()
   assert frame.shape==(120,160,3)
   report['headless_rgb_render']=dict(passed=True,shape=list(frame.shape))
  finally:env.close()
 except Exception as error:report['headless_rgb_render']=dict(passed=False,error=repr(error))
 from stable_baselines3 import SAC,TD3
 for cls in [SAC,TD3]:
  env=CompactGoal(gym.make(ids['FetchPush']))
  try:
   model=cls('MlpPolicy',env,buffer_size=32,batch_size=4,device='cpu',seed=0)
   obs,_=env.reset(seed=5);action,_=model.predict(obs,deterministic=False)
   assert action.shape==(4,) and np.isfinite(action).all()
   env.step(action)
   report['baseline_predict'].append(dict(method=cls.__name__,observation_dim=len(obs),action_dim=len(action),passed=True))
  finally:env.close()
 try:
  import sys
  sys.path.insert(0,'/workspace/iBOLT')
  import train
  from ibolt.algorithm import IBOLT
  cfg=train.compose_config(['benchmark=hopper','wandb.activate=false','require_gpu=false','dacer.enabled=false',
   'alg.buffer_size=32','alg.batch_size=4'])
  env=CompactGoal(gym.make(ids['FetchPush']))
  try:
   model=IBOLT('MlpPolicy',env,model_save_path=None,save_every_n_steps=0,cfg=cfg,gradient_steps=1)
   obs,_=env.reset(seed=5);action,_=model.predict(obs,deterministic=False)
   assert action.shape==(4,) and np.isfinite(action).all()
   env.step(action)
   report['baseline_predict'].append(dict(method='iBOLT',observation_dim=len(obs),action_dim=len(action),passed=True))
  finally:env.close()
 except Exception as error:report['baseline_predict'].append(dict(method='iBOLT',passed=False,error=repr(error),traceback=traceback.format_exc()[-1800:]))
 report['elapsed_seconds']=time.time()-started
 output=Path('/tmp/ibolt-fetch-feasibility-20261108')/(args.profile+'-report.json')
 output.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report),flush=True)

if __name__=='__main__':main()

