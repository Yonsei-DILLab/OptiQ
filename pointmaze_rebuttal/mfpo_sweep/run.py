"""MFPO target-entropy/initial-temperature sweep, fixed common PointMaze budget."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback

from ..guards import assert_work_branch, assert_output_boundary, code_hashes
from ..baselines.profiles import configuration
from ..baselines.run import save_json, runtime


def measure(agent, config, output, step, episodes):
    import numpy as np
    from ..evaluation import evaluate
    from .metrics import goal_metrics
    row=dict(steps=step,updates=agent.updates,method='mfpo',
        primary_evaluation_mode='official',secondary_evaluation_mode='single_sample')
    numpy_state,python_state=np.random.get_state(),random.getstate()
    previous=agent.evaluation_mode
    try:
        for mode in ('official','single_sample'):
            agent.evaluation_mode=mode
            for obstacle,offset in ((False,17000),(True,27000)):
                key=('obstacle_' if obstacle else '')+mode
                row[key]=goal_metrics(evaluate(agent,'pm_'+config['maze'],
                    config['seed']+offset+step,episodes,mode,
                    output/'evaluations'/f'{step:09d}_{key}.npz',obstacle))
    finally:
        agent.evaluation_mode=previous
        np.random.set_state(numpy_state);random.setstate(python_state)
    save_json(output/'evaluations'/f'{step:09d}_summary.json',row)
    with (output/'evaluations.jsonl').open('a') as stream:
        stream.write(json.dumps(row,allow_nan=False)+'\n')
    print(json.dumps(dict(steps=step,updates=agent.updates,
        official_success=row['official']['success'],
        single_sample_success=row['single_sample']['success'],
        official_coverage=row['official']['reachable_goals'])),flush=True)
    return row


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maze',choices=('medium','hard'),required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--target-entropy-coeff',type=float,required=True)
    parser.add_argument('--initial-temperature',type=float,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args(argv)
    branch=assert_work_branch();assert_output_boundary(args.output)
    if not args.target_entropy_coeff<0 or args.initial_temperature<=0:
        raise ValueError('Negative entropy coefficient and positive temperature required')
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    os.environ.setdefault('MUJOCO_GL','egl')
    if args.device=='cpu':os.environ['JAX_PLATFORMS']='cpu'
    import numpy as np
    import jax
    import flax.serialization
    from .agent import Agent
    from ..envs.batch import TaskBatch
    config=configuration('mfpo',args.maze,args.seed,False,'matched')
    config['native_settings']['mfpo_target_entropy_coeff']=args.target_entropy_coeff
    config.update(initial_temperature=args.initial_temperature,
        sweep_parameters=dict(target_entropy_coeff=args.target_entropy_coeff,temp=args.initial_temperature),
        evaluation_mode='official',evaluation_secondary='single_sample',
        official_candidate_count=10,branch=branch,device=args.device,
        timeout_bootstrap=False,save_replay=True,
        training_policy='native direct MFPO sample; collection unchanged',
        checkpoint_evaluation_does_not_update_learner=True)
    if args.smoke:
        config.update(smoke=True,steps=1024,num_envs=256,batch_size=256,
            updates_per_collect=2,warmup=512,eval_every=1024,eval_episodes=5,
            final_episodes=5,replay_capacity=2048,expected_updates=4,utd=2/256)
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    for name in ('checkpoints','evaluations'):(output/name).mkdir()
    save_json(output/'config.json',config)
    save_json(output/'source-hashes.json',code_hashes())
    save_json(output/'progress.json',dict(steps=0,updates=0,status='initializing',observed_unix=time.time()))
    np.random.seed(args.seed);random.seed(args.seed);rng=np.random.default_rng(args.seed)
    agent=env=None;step=0;started=time.time()
    try:
        agent=Agent(config,output,device=args.device)
        config.update(agent=agent.config,runtime=runtime('mfpo',args.device))
        save_json(output/'config.json',config)
        save_json(output/'initialization-audit.json',agent.initialization_audit)
        if not np.isclose(float(agent.native.agent.temp.apply_fn({'params':agent.native.agent.temp.params})),args.initial_temperature,rtol=1e-5):
            raise RuntimeError('Initial MFPO temperature was not applied')
        if not np.isclose(agent.native.agent.target_entropy,args.target_entropy_coeff*2):
            raise RuntimeError('MFPO target entropy was not applied')
        if args.smoke:
            obs=np.random.default_rng(918).normal(size=(5,4)).astype(np.float32)
            with agent.evaluation_rng(919):a=agent.act(obs,'train')
            with agent.evaluation_rng(919):b=agent.act(obs,'single_sample')
            np.testing.assert_array_equal(a,b)
            with agent.evaluation_rng(920):
                state=agent.native.agent;keys=jax.random.split(state.rng,len(obs)+1)
                reference=np.stack([np.asarray(state.replace(rng=key).eval_actions_select(row,cand_num=10)[0])
                                    for row,key in zip(obs,keys[1:])])
            with agent.evaluation_rng(920):selected=agent.act(obs,'official')
            np.testing.assert_allclose(selected,reference,rtol=1e-5,atol=1e-5)
            save_json(output/'evaluation-path-audit.json',dict(training_sample_unchanged=True,
                official_vector_selection_matches_upstream=True,max_absolute_error=float(np.max(abs(selected-reference)))))
        env=TaskBatch(args.maze,config['num_envs'],args.seed)
        for step in range(config['num_envs'],config['steps']+1,config['num_envs']):
            obs=env.current.copy()
            actions=(rng.uniform(-1.,1.,(config['num_envs'],2)).astype(np.float32)
                     if step<=config['warmup'] else agent.act(obs,'train'))
            if not np.isfinite(actions).all():raise FloatingPointError('Nonfinite MFPO actions')
            actions=np.clip(actions,-1.,1.).astype(np.float32)
            nxt,rewards,terminated,truncated,_=env.step(actions)
            if not np.isfinite(nxt).all() or not np.isfinite(rewards).all():
                raise FloatingPointError('Nonfinite environment transition')
            done=terminated|truncated
            agent.observe(obs,actions,rewards,nxt,done)
            env.reset(np.flatnonzero(done))
            if step>config['warmup']:
                for _ in range(config['updates_per_collect']):agent.update(step)
            save_json(output/'progress.json',dict(steps=step,updates=agent.updates,
                status='running',observed_unix=time.time(),elapsed_seconds=time.time()-started))
            if agent.updates and (agent.updates%2048==0 or step==config['steps']):
                record=dict(steps=step,updates=agent.updates,**agent.last_info)
                with (output/'diagnostics.jsonl').open('a') as stream:
                    stream.write(json.dumps(record,allow_nan=False)+'\n')
            if step%config['eval_every']<config['num_envs'] or step==config['steps']:
                agent.assert_finite()
                digest=hashlib.sha256(flax.serialization.to_bytes(agent.native.agent)).hexdigest()
                episodes=config['final_episodes'] if step==config['steps'] else config['eval_episodes']
                measure(agent,config,output,step,episodes)
                if digest!=hashlib.sha256(flax.serialization.to_bytes(agent.native.agent)).hexdigest():
                    raise RuntimeError('Evaluation mutated MFPO learner state or RNG')
                agent.save(output/'checkpoints'/f'{step:09d}.msgpack')
        if agent.updates!=config['expected_updates']:raise RuntimeError('UTD/update count mismatch')
        agent.save_replay(output/'checkpoints/final-replay.npz')
        result=dict(completed=True,method='mfpo',maze=args.maze,seed=args.seed,
            target_entropy_coeff=args.target_entropy_coeff,initial_temperature=args.initial_temperature,
            steps=step,updates=agent.updates,backend=jax.default_backend(),smoke=args.smoke,
            elapsed_seconds=time.time()-started,finished_unix=time.time())
        save_json(output/'completed.json',result)
        save_json(output/'progress.json',dict(result,status='complete'))
        print(json.dumps(result),flush=True)
    except BaseException as error:
        save_json(output/'failure.json',dict(steps=step,updates=agent.updates if agent else 0,
            error=repr(error),traceback=traceback.format_exc()))
        raise
    finally:
        if env is not None:env.close()
        if agent is not None:agent.close()


if __name__=='__main__':main()
