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

output = pathlib.Path('/tmp/mfpo-her-audit.json')
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
for seed in range(5):
    run=base/'runs'/f'pick-and-place-mfpo-s{seed}'
    for step in (100000,300000):
        path=run/'checkpoints'/f'policy_{step:09d}.msgpack'
        if not path.exists(): continue
        agent=flax.serialization.from_bytes(template,path.read_bytes())
        norm=pickle.load((path.parent/f'normalizer_{step:09d}.pkl').open('rb'))
        leaves=jax.tree_util.tree_leaves(agent)
        row=dict(seed=seed,step=step,alpha=float(agent.temp.apply_fn({'params':agent.temp.params})),
                 finite=all(bool(np.isfinite(np.asarray(x)).all()) for x in leaves),
                 optimizer_steps={k:int(getattr(agent,k).step) for k in ('actor','critic_1','critic_2','logp_mvel','temp')})
        obs=norm.normalize_obs(raw).astype(np.float32)
        repeated=np.repeat(obs,16,axis=0)
        noise=jax.random.normal(jax.random.PRNGKey(411),(len(repeated),4))
        unclipped=np.asarray(sampler(agent,repeated,noise));actions=np.clip(unclipped,-1,1)
        _,logp=withlogp(agent,repeated,noise)
        row.update(action_mean=actions.mean(axis=0).tolist(),action_std=actions.std(axis=0).tolist(),
                   within_state_action_std=actions.reshape(100,16,4).std(axis=1).mean(axis=0).tolist(),
                   abs_action_over099=float((np.abs(actions)>.99).mean()),unclipped=stats(unclipped),
                   estimated_entropy=stats(-logp),policy_q=stats(value(agent,repeated,actions)))
        alt=raw.copy();alt[:,-3:]=raw[::-1,-3:]
        altobs=np.repeat(norm.normalize_obs(alt).astype(np.float32),16,axis=0)
        alta=np.clip(np.asarray(sampler(agent,altobs,noise)),-1,1)
        row['same_noise_goal_change_action_mae']=float(np.abs(alta-actions).mean())
        qzero=np.asarray(value(agent,obs,np.zeros((100,4),np.float32)))
        row['zero_action_q']=stats(qzero)
        result['checkpoints'].append(row);save();print(json.dumps(row),flush=True)
        if step!=300000:continue
        for mode in ('native','sample'):
            agent=agent.replace(rng=jax.random.PRNGKey(81234))
            evalenv=make_env('pick-and-place');episodes=[]; actionrows=[]
            # A common 50 held-out conditions for each seed and both evaluation modes.
            for envseed in range(900000,900050):
                state,_=evalenv.reset(seed=envseed);init=state.copy();initial=np.linalg.norm(init[3:6]-init[-3:])<.05
                anyhit=False;lift=0.;ret=0.
                for t in range(50):
                    fn=agent.eval_actions if mode=='native' else agent.sample_actions
                    action,agent=fn(norm.normalize_obs(state));action=np.asarray(action);actionrows.append(action)
                    state,reward,terminated,truncated,info=evalenv.step(np.clip(action,-1,1))
                    ret+=float(reward);anyhit|=bool(info['is_success']);lift=max(lift,float(state[5]-init[5]))
                    if terminated or truncated:break
                episodes.append(dict(initial=bool(initial),final=bool(info['is_success']),any=anyhit,
                                     new_final=bool(info['is_success']) and not initial,new_any=anyhit and not initial,
                                     lift=lift,return_=ret,object_displacement=float(np.linalg.norm(state[3:6]-init[3:6])),
                                     grip_object_distance=float(np.linalg.norm(state[:3]-state[3:6]))))
            evalenv.close()
            e=dict(seed=seed,step=step,mode=mode,episodes=len(episodes),
                   **{k:sum(x[k] for x in episodes) for k in ('initial','final','any','new_final','new_any')},
                   lift_over5cm=sum(x['lift']>.05 for x in episodes),mean_return=float(np.mean([x['return_'] for x in episodes])),
                   mean_object_displacement=float(np.mean([x['object_displacement'] for x in episodes])),
                   mean_final_grip_object_distance=float(np.mean([x['grip_object_distance'] for x in episodes])),
                   action_mean=np.mean(actionrows,axis=0).tolist(),action_std=np.std(actionrows,axis=0).tolist())
            result['evaluations'].append(e);save();print(json.dumps(e),flush=True)
result['elapsed_seconds']=time.time()-result['started'];save()
print(json.dumps(dict(output=str(output),elapsed=result['elapsed_seconds'])),flush=True)
