"""Read-only CPU checkpoint comparison on identical Fetch evaluation conditions."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR='/tmp/fetch-ibolt-paired-mpl')
import sys,pathlib,json,pickle,time
import numpy as np
b=pathlib.Path('/workspace/iBOLT/outputs/fetch-rebuttal/her-b256-utd1-20261109T082908Z/host1')
sys.path.insert(0,str(b/'source'))
import jax,flax.serialization
from omegaconf import OmegaConf
from ibolt.policy import IBOLTPolicy
from fetch_rebuttal.env import make_env
output=pathlib.Path('/tmp/fetch-ibolt-1m-paired-audit.json')
result=dict(started_unix=time.time(),backend=jax.default_backend(),training_modified=False,cases=[])
def save():output.write_text(json.dumps(result,indent=2,allow_nan=False))
env=make_env('pick-and-place')
for seed in (0,1):
    run=b/'runs'/f'pick-and-place-ibolt-T0.25-s{seed}'
    cfg=OmegaConf.load(run/'mujoco-config.json')
    policy=IBOLTPolicy(env.observation_space,env.action_space,cfg=cfg)
    policy.build(jax.random.key(seed),None,cfg.alg.optimizer.lr_critic)
    template=policy.actor_state
    for step in (900000,1000000):
        ckpt=run/'checkpoints'
        policy.actor_state=flax.serialization.from_bytes(template,(ckpt/f'actor_state_{step}.msgpack').read_bytes())
        policy.key=jax.random.key(700000+seed*1000);policy.evaluation_mu_only=True
        norm=pickle.load((ckpt/f'normalizer_{step:09d}.pkl').open('rb'))
        count=float(norm.obs_rms.count)
        episodes=[]
        for e in range(100):
            obs,_=env.reset(seed=700000+seed*1000+e);initial=obs.copy();anyhit=False;lift=0.;ret=0.
            for t in range(50):
                action,_=policy.predict(norm.normalize_obs(obs),deterministic=False)
                obs,reward,terminated,truncated,info=env.step(np.clip(action,-1,1));ret+=float(reward)
                anyhit|=bool(info['is_success']);lift=max(lift,float(obs[5]-initial[5]))
                if terminated or truncated:break
            episodes.append(dict(final=bool(info['is_success']),any=anyhit,return_=ret,
                height_offset=float(initial[-1]-initial[5]),lift=lift))
        assert float(norm.obs_rms.count)==count
        data=json.loads((run/'evaluations'/f'{step:09d}.json').read_text())['stochastic_z']
        high=[e for e in episodes if e['height_offset']>.05];ground=[e for e in episodes if abs(e['height_offset'])<.001]
        row=dict(seed=seed,step=step,episodes=100,final_success=sum(e['final'] for e in episodes),
            any_success=sum(e['any'] for e in episodes),first10_success=sum(e['final'] for e in episodes[:10]),
            ground_episodes=len(ground),ground_success=sum(e['final'] for e in ground),
            air_over5cm_episodes=len(high),air_over5cm_success=sum(e['final'] for e in high),
            lift_over5cm=sum(e['lift']>.05 for e in episodes),logged_episodes=data['episodes'],
            logged_success=data['success_rate'],logged_success_array_matches=(
                [int(e['final']) for e in episodes[:data['episodes']]]==data['final_success']),records=episodes)
        result['cases'].append(row);save();print(json.dumps({k:v for k,v in row.items() if k!='records'}),flush=True)
env.close();result['elapsed_seconds']=time.time()-result['started_unix'];save()
