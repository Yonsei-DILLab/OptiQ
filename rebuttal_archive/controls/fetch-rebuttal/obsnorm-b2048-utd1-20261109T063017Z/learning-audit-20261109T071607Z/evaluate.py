
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR='/tmp/fetch-learning-audit-mpl',XLA_PYTHON_CLIENT_PREALLOCATE='false')
import sys,pathlib,json,time,pickle,argparse
import numpy as np
campaign=pathlib.Path('/workspace/iBOLT/outputs/fetch-rebuttal/obsnorm-b2048-utd1-20261109T063017Z')
parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['controls','sac','mfpo']);args=parser.parse_args()
host=2 if args.mode=='mfpo' else 1
base=campaign/f'host{host}';sys.path.insert(0,str(base/'source'))
from fetch_rebuttal.env import make_env
output=pathlib.Path('/tmp/fetch-learning-audit-'+args.mode+'.json')
result={'mode':args.mode,'started_unix':time.time(),'protocol':'Read-only CPU evaluation, no learning. Original: fixed 50 environment seeds used by the five training seeds. Heldout: common seeds 900000..900099. Random control draws independent uniform actions in [-1,1].','cases':[]}
def save():output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
def evaluate(label,seeds,act,repeats=1,extra=None):
 env=make_env('pick-and-place');episodes=[]
 for repeat in range(repeats):
  for seed in seeds:
   obs,_=env.reset(seed=seed);init=obs.copy();dist=float(np.linalg.norm(obs[3:6]-obs[-3:]));anyhit=False;positive=0;maxlift=0.;closest=dist;ret=0.;steps=0
   for step in range(50):
    action=np.asarray(act(obs,seed,repeat,step),np.float32).reshape(4)
    assert np.isfinite(action).all()
    obs,reward,terminated,truncated,info=env.step(np.clip(action,-1,1));ret+=float(reward);steps+=1
    anyhit|=bool(info['is_success']);positive+=int(reward==0);maxlift=max(maxlift,float(obs[5]-init[5]));closest=min(closest,float(np.linalg.norm(obs[3:6]-obs[-3:])))
    if terminated or truncated:break
   final=bool(info['is_success']);initial=dist<.05
   episodes.append(dict(seed=seed,repeat=repeat,initial_success=initial,goal=init[-3:].tolist(),initial_distance=dist,final_success=final,any_success=anyhit,new_final_success=final and not initial,new_any_success=anyhit and not initial,return_=ret,zero_reward_steps=positive,max_object_lift=maxlift,minimum_distance=closest,final_distance=float(np.linalg.norm(obs[3:6]-obs[-3:]))))
 env.close();N=len(episodes);ninitial=sum(e['initial_success'] for e in episodes);nf=sum(e['new_final_success'] for e in episodes);na=sum(e['new_any_success'] for e in episodes);high=[e for e in episodes if e['goal'][2]>=.50]
 summary=dict(episodes=N,unique_initial_conditions=len(seeds),repeats=repeats,initial_success_count=ninitial,final_success_count=sum(e['final_success'] for e in episodes),any_success_count=sum(e['any_success'] for e in episodes),new_final_success_count=nf,new_any_success_count=na,final_success_rate=sum(e['final_success'] for e in episodes)/N,new_final_rate_among_initial_failures=nf/max(1,N-ninitial),mean_return=float(np.mean([e['return_'] for e in episodes])),high_goal_episodes=len(high),high_goal_final_success_count=sum(e['final_success'] for e in high),lift_over5cm_episodes=sum(e['max_object_lift']>.05 for e in episodes))
 row=dict(label=label,summary=summary,episodes=episodes,extra=extra);result['cases'].append(row);save();print(json.dumps({'label':label,'summary':summary,'extra':extra}),flush=True)
original=[700000+s*1000+i for s in range(5) for i in range(10)];heldout=list(range(900000,900100))
if args.mode=='controls':
 zero=lambda obs,seed,repeat,step:np.zeros(4,np.float32)
 evaluate('zero-original',original,zero)
 evaluate('zero-heldout',heldout,zero)
 for cohort,seeds,repeats in [('original',original,20),('heldout',heldout,5)]:
  rngs={}
  def random_action(obs,seed,repeat,step):
   key=(seed,repeat)
   if key not in rngs:rngs[key]=np.random.default_rng(np.random.SeedSequence([seed,repeat,276513]))
   return rngs[key].uniform(-1,1,4).astype(np.float32)
  evaluate('random-'+cohort,seeds,random_action,repeats)
elif args.mode=='sac':
 import torch
 from stable_baselines3 import SAC
 torch.set_num_threads(1)
 for seed in range(5):
  run=base/'runs'/f'pick-and-place-sac-s{seed}';files=sorted((run/'checkpoints').glob('policy_*.zip'))
  if not files:continue
  ckpt=files[-1];step=int(ckpt.stem.split('_')[-1]);norm=pickle.load((run/'checkpoints'/f'normalizer_{step:09d}.pkl').open('rb'));model=SAC.load(ckpt,device='cpu')
  def action(obs,seed,repeat,step):return model.predict(norm.normalize_obs(obs),deterministic=True)[0]
  extra=dict(seed=seed,checkpoint_step=step,alpha=float(model.log_ent_coef.exp().item()),parameters_finite=all(bool(torch.isfinite(p).all()) for p in model.policy.parameters()))
  evaluate(f'sac-s{seed}-original',[700000+seed*1000+i for i in range(10)],action,extra=extra)
  evaluate(f'sac-s{seed}-heldout',heldout,action,extra=extra)
  del model
else:
 import jax,flax.serialization
 from fetch_rebuttal.baselines.native import MFPO
 template=None
 for seed in range(5):
  run=base/'runs'/f'pick-and-place-mfpo-s{seed}';files=sorted((run/'checkpoints').glob('policy_*.msgpack'))
  if not files:continue
  ckpt=files[-1];step=int(ckpt.stem.split('_')[-1]);norm=pickle.load((run/'checkpoints'/f'normalizer_{step:09d}.pkl').open('rb'))
  if template is None:
   agent=MFPO(seed,pathlib.Path('/tmp'),1000000,28,batch_size=2048,action_dim=4);template=agent.agent
  agent.agent=flax.serialization.from_bytes(template,ckpt.read_bytes()).replace(rng=jax.random.PRNGKey(800000+seed))
  def action(obs,seed_,repeat,step_):
   a,agent.agent=agent.agent.eval_actions(norm.normalize_obs(obs));return np.asarray(a)
  extra=dict(seed=seed,checkpoint_step=step,backend=jax.default_backend())
  evaluate(f'mfpo-s{seed}-original',[700000+seed*1000+i for i in range(10)],action,extra=extra)
  evaluate(f'mfpo-s{seed}-heldout',heldout,action,extra=extra)
result['finished_unix']=time.time();save();print(json.dumps({'saved':str(output),'elapsed_seconds':result['finished_unix']-result['started_unix']}),flush=True)
