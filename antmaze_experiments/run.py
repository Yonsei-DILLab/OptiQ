"""256 official Gym environments with native sparse baseline accounting."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time
import traceback
import numpy as np

from .settings import (CAMPAIGN, BUDGETS, REWARD, DENSE_REWARD, NUM_ENVS, EVAL_NUM_ENVS, UPDATES, WARMUP,
                       PREFLIGHT_STEPS, total_budget, expected_updates, WANDB_ENTITY, WANDB_PROJECT)


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def evaluate(learner, task, folder, step, episodes, mode, fixed=False):
    from .envs import vector, transition
    from .learners import evaluation_rng
    label = f'{mode}-' + ('fixed' if fixed else 'natural')
    count = min(EVAL_NUM_ENVS, episodes)
    destination = folder/'evaluations'/f'{step:010d}'/label
    destination.mkdir(parents=True, exist_ok=True)
    paths, returns, goals, lengths, starts = [], [], [], [], []
    with evaluation_rng(learner, 700000 + step):
        env = vector(task, count, seed=87231, asynchronous=False, fixed=fixed,
                     reward_profile=learner.reward_profile,
                     random_init=True if getattr(learner,'eval_random_starts',False) else None)
        try:
            for batch in range((episodes+count-1)//count):
                obs = env.reset(); active = np.arange(count)+batch*count < episodes
                if fixed:
                    common = env.envs[0].initial
                    for i,e in enumerate(env.envs):
                        e.initial = common
                        obs[i] = e.restore(common)
                tracks = [[o[:2].copy()] for o in obs]
                full_starts = [e.state() for e in env.envs]
                rets = np.zeros(count); lens = np.zeros(count, int)
                goal = np.zeros(count, int)
                for _ in range(500 if task in ('v1','v2') else 700):
                    actions = learner.act(obs, mode)
                    nxt, reward, done, infos = env.step(actions)
                    final, _ = transition(nxt, done, infos)
                    for i in np.flatnonzero(active):
                        tracks[i].append(final[i,:2].copy()); rets[i] += reward[i]; lens[i] += 1
                        if done[i]:
                            active[i] = False
                            goal[i] = int(infos[i].get('success',0))
                    obs = nxt
                    if not active.any(): break
                for i in range(min(count, episodes-batch*count)):
                    paths.append(np.asarray(tracks[i])); returns.append(rets[i])
                    goals.append(goal[i]); lengths.append(lens[i])
                    starts.append(np.r_[full_starts[i]['qpos'],full_starts[i]['qvel']])
        finally:
            env.close()
    max_length = max(map(len, paths))
    xy = np.full((episodes,max_length,2), np.nan, np.float32)
    for i,p in enumerate(paths): xy[i,:len(p)] = p
    np.savez_compressed(destination/'rollouts.npz', xy=xy, returns=returns,
        goals=goals, lengths=lengths, initial_full_state=starts,
        env_steps=np.array(step), mode=mode, fixed=fixed)
    result = dict(step=step, mode=mode, fixed=fixed, episodes=episodes,
        success_rate=float(np.mean(np.array(goals)>0)), mean_return=float(np.mean(returns)),
        goal_counts={str(g):int(np.sum(np.array(goals)==g)) for g in sorted(set(goals))},
        mean_length=float(np.mean(lengths)),
        identical_initial_full_state=bool(np.all(np.array(starts)==starts[0])))
    result['start_distribution'] = ('fixed first sampled full state' if fixed else
        'xy uniform[-2,2], original pose/velocity' if task=='v1' or getattr(learner,'eval_random_starts',False)
        else 'original fixed full state')
    if fixed: assert result['identical_initial_full_state']
    write(destination/'summary.json',result)
    return result


def checkpoint(learner, env, obs, folder, step, rng, config):
    import torch
    memory = learner.replay
    replay = {k:v[:memory.cur_capacity].cpu() for k,v in vars(memory).items()
              if isinstance(v,torch.Tensor) and k.startswith('buf_')}
    intrinsic = learner.intrinsic
    state = dict(config=config,step=step,updates=learner.updates,learner=learner.state(),
        replay=replay,replay_metadata=dict(next_p=memory.next_p,if_full=memory.if_full,
            cur_capacity=memory.cur_capacity,total_samples=memory.total_samples),
        intrinsic=(dict(model=intrinsic.rnd_model.state_dict(),
            optimizer=intrinsic.rnd_optimizer.state_dict(),updates=intrinsic.update_step)
            if learner.noveld_enabled else dict(enabled=False,updates=0)),
        env_states=env.call('state'),observations=obs,behavior_rng=rng.bit_generator.state,
        python_rng=random.getstate(),numpy_rng=np.random.get_state(),
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
    path = folder/'checkpoint-final.pt'
    torch.save(state,path)
    # Read back serialized model/replay/simulator data before marking complete.
    loaded = torch.load(path, map_location='cpu', weights_only=False)
    assert loaded['step']==step and loaded['updates']==learner.updates
    assert loaded['replay']['buf_obs'].shape[0]==min(step,memory.capacity)
    assert loaded['replay_metadata']['total_samples']==step
    for key,value in replay.items(): assert torch.equal(loaded['replay'][key],value),key
    assert np.array_equal(loaded['observations'],obs)
    with path.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
    from .envs import make_one
    reference=make_one(config['task'],0)
    target_goals=np.asarray(reference.physics_env.target_goal).reshape(-1,2)
    reference.close()
    positions=loaded['replay']['buf_next_obs'][:,:2].numpy()
    distances=np.linalg.norm(positions[:,None,:]-target_goals[None,:,:],axis=-1)
    actual=loaded['replay']['buf_reward'].numpy().ravel()
    if config['reward_profile']=='dense':
        expected=-distances.min(axis=1)
        assert np.allclose(actual,expected,rtol=2e-6,atol=2e-5)
        boundary=np.zeros(len(actual),bool)
    else:
        expected=np.zeros(len(positions),np.float32)
        for goal in reversed(target_goals):
            reached=np.linalg.norm(positions-goal,axis=-1)<=.5
            expected[reached]=20 if tuple(goal)==(-8,8) else 10
        assert np.isin(actual,[0,10,20]).all()
        boundary=np.min(np.abs(distances-.5),axis=1)<2e-5
        assert np.array_equal(actual[~boundary],expected[~boundary])
    proof = dict(path=path.name,sha256=digest,bytes=path.stat().st_size,
        steps=step,updates=learner.updates,replay_count=memory.cur_capacity,
        simulator_count=len(loaded['env_states']),readback_verified=True,
        environment_reward_verified=True,reward_profile=config['reward_profile'],
        sparse_replay_verified=config['reward_profile']=='sparse',
        dense_replay_verified=config['reward_profile']=='dense',
        intrinsic_enabled=learner.noveld_enabled,
        threshold_roundoff_rows=int(boundary.sum()))
    write(folder/'checkpoint-verification.json',proof)
    return proof


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--method', choices=['optiq','sac','dipo','mfpo'],required=True)
    p.add_argument('--task',choices=['v1','v2','v3','v4'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preflight',action='store_true')
    p.add_argument('--temperature',type=float)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--budget-steps',type=int)
    p.add_argument('--final-eval-episodes',type=int,default=100)
    p.add_argument('--interim-eval-episodes',type=int,default=EVAL_NUM_ENVS)
    p.add_argument('--save-intermediate-policy',action='store_true')
    p.add_argument('--reward-profile',choices=['sparse','dense'],default='sparse')
    p.add_argument('--noveld',choices=['on','off'],default='on')
    p.add_argument('--eval-starts',choices=['upstream','random'],default='upstream')
    a = p.parse_args()
    assert 0 <= a.seed < 2**31
    if a.temperature is not None:
        assert a.method == 'optiq' and a.temperature > 0
    if a.budget_steps is not None:
        assert a.budget_steps >= WARMUP + NUM_ENVS
        assert a.budget_steps % NUM_ENVS == 0
    assert 1 <= a.final_eval_episodes <= 1000
    assert 1 <= a.interim_eval_episodes <= 1000
    if a.save_intermediate_policy:
        assert a.method == 'optiq', 'Intermediate policy saving is currently OptiQ-only'
    root = Path(__file__).resolve().parents[1]
    from .dependencies import verify_dependencies
    dependencies = verify_dependencies(root, (a.method,))
    source = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=root,text=True).strip()
    folder=a.output; folder.mkdir(parents=True,exist_ok=False)
    cpus = sorted(os.sched_getaffinity(0)); gpu=int(os.environ.get('CAMPAIGN_GPU','0'))
    os.sched_setaffinity(0,cpus[gpu::4] or cpus)
    # Fork workers before importing torch/JAX or initializing a GPU context.
    from .envs import vector, transition
    env = vector(a.task,NUM_ENVS,seed=a.seed*NUM_ENVS,reward_profile=a.reward_profile)
    obs = env.reset()
    import torch
    torch.set_num_threads(1);torch.manual_seed(a.seed)
    random.seed(a.seed);np.random.seed(a.seed)
    assert torch.cuda.is_available()
    from .learners import Native,JaxLearner,audit
    planned_budget = a.budget_steps if a.budget_steps is not None else total_budget(a.task)
    if a.method in ('sac','dipo'):
        learner = Native(a.method,(env.single_observation_space,env.single_action_space),a.task,folder,
                         reward_profile=a.reward_profile,noveld=a.noveld=='on',seed=a.seed)
    else:
        learner = JaxLearner(a.method,(env.single_observation_space,env.single_action_space),
            a.task,folder,temperature=a.temperature,budget=planned_budget,
            reward_profile=a.reward_profile,noveld=a.noveld=='on',seed=a.seed)
    initial = audit(learner)
    learner.eval_random_starts = a.eval_starts=='random'
    budget = PREFLIGHT_STEPS if a.preflight else planned_budget
    warmup = WARMUP
    config = dict(source_commit=source,upstream_commit='7edd06c4799abbab0f8fa534c21deb56253b018e',
        method=a.method,task=a.task,seed=a.seed,preflight=a.preflight,steps=budget,
        environment_seed_base=a.seed*NUM_ENVS,
        source_dependencies=dependencies,
        wandb_entity=WANDB_ENTITY,wandb_project=WANDB_PROJECT,
        num_envs=NUM_ENVS,batch_size=4096,updates_per_vector_step=UPDATES,updates_per_transition=1/32,
        expected_updates=expected_updates(budget),warmup_transitions=warmup,
        upstream_max_step=BUDGETS[a.task],native_global_steps=budget-warmup,
        reward=REWARD if a.reward_profile=='sparse' else DENSE_REWARD,
        reward_profile=a.reward_profile,noveld_enabled=a.noveld=='on',
        eval_starts=a.eval_starts,primary_trajectory='policy-natural',
        noveld_coefficient=.01 if a.noveld=='on' else 0.,temperature=a.temperature,
        native=learner.config,random_init=a.task=='v1',eval_interval=250000,
        eval_num_envs=EVAL_NUM_ENVS,interim_eval_episodes=a.interim_eval_episodes,
        final_eval_episodes=a.final_eval_episodes,
        save_intermediate_policy=a.save_intermediate_policy,
        policy_checkpoint_interval=250000 if a.save_intermediate_policy else None,
        policy_checkpoint_kind='evaluation-only; model/optimizer/RNG, no replay or simulator',
        checkpoint='final only',step_definition='step includes warmup; global_steps excludes warmup as upstream; stop global_steps>max_step',
        evaluation='primary=policy-natural; policy: direct draws, no extra exploration noise; native: SAC mean, MFPO Q-best-of10, OptiQ random-z mu-only, DIPO native diffusion; fixed starts are supplementary',
        runtime=dict(python=os.sys.version,torch=torch.__version__,numpy=np.__version__))
    write(folder/'config.json',config)
    import wandb
    temp_name=f'-T{a.temperature:g}' if a.temperature is not None else ''
    run_name=f'{a.task}-{a.method}{temp_name}-{a.reward_profile}-noveld{a.noveld}-s{a.seed}-{budget}steps'
    run = wandb.init(entity=WANDB_ENTITY,project=WANDB_PROJECT,group=os.environ.get('OPTIQ_CAMPAIGN',CAMPAIGN),
        name=run_name,dir=str(folder),config=config,
        mode='disabled' if a.preflight else os.environ.get('WANDB_MODE','online'))
    if not a.preflight:
        offline=os.environ.get('WANDB_MODE','online')=='offline'
        write(folder/'wandb.json',dict(id=run.id,url=None if offline else run.url,
            entity=WANDB_ENTITY,project=WANDB_PROJECT,
            mode='offline' if offline else 'online',sync_pending=offline))
    rng=np.random.default_rng(a.seed);step=0;started=time.monotonic();next_eval=250000
    timing=dict(collection=0.,learner=0.,evaluation=0.,checkpoint=0.)
    xy=np.empty((budget,2),np.float32);successes=[];episodes=0;info={}
    def progress():
        record=dict(step=step,global_steps=max(0,step-warmup),updates=learner.updates,rnd_updates=learner.intrinsic.update_step,
            seconds=time.monotonic()-started,episodes=episodes,successes=len(successes),
            **{f'seconds/{k}':v for k,v in timing.items()})
        write(folder/'progress.json',record);print(json.dumps(record),flush=True)
        run.log(dict(record,**info),step=step)
    try:
        while step<budget:
            t=time.monotonic()
            action=rng.uniform(-1,1,(NUM_ENVS,8)).astype(np.float32) if step<warmup else learner.act(obs)
            assert action.shape==(NUM_ENVS,8) and np.isfinite(action).all()
            nxt,reward,done,infos=env.step(np.clip(action,-1,1))
            final,terminal=transition(nxt,done,infos)
            if isinstance(learner,Native): learner.record_collection(obs,reward,done)
            learner.store(obs,action,reward,final,terminal)
            xy[step:step+NUM_ENVS]=final[:,:2]
            for i,entry in enumerate(infos):
                if done[i]: episodes+=1
                if entry.get('success',0): successes.append(dict(step=step+i+1,env=i,goal=int(entry['success'])))
            obs=nxt;step+=NUM_ENVS;timing['collection']+=time.monotonic()-t
            if step>warmup:
                t=time.monotonic();info=learner.update(step);timing['learner']+=time.monotonic()-t
                assert all(np.isfinite(float(v)) for v in info.values()), info
                assert learner.updates==expected_updates(step)
                assert learner.intrinsic.update_step==(learner.updates if learner.noveld_enabled else 0)
            if step>=next_eval and step<budget:
                if a.save_intermediate_policy:
                    from .policy_checkpoints import save_evaluation_checkpoint
                    t=time.monotonic()
                    save_evaluation_checkpoint(learner,folder,step,config)
                    timing['checkpoint']+=time.monotonic()-t
                t=time.monotonic()
                for mode in ('native','policy'):
                    s=evaluate(learner,a.task,folder,step,a.interim_eval_episodes,mode)
                    run.log({f'eval/{mode}/success_rate':s['success_rate'],
                             f'eval/{mode}/return':s['mean_return']},step=step)
                timing['evaluation']+=time.monotonic()-t
                next_eval+=250000
            if step%4096==0 or step==budget: progress()
        final_audit=audit(learner)
        for key in (('actor','critic','rnd_predictor') if learner.noveld_enabled else ('actor','critic')):
            assert initial[key]['sha256']!=final_audit[key]['sha256'],key
        if learner.noveld_enabled:assert initial['rnd_target']==final_audit['rnd_target']
        write(folder/'parameter-audit.json',dict(initial=initial,final=final_audit,passed=True))
        np.save(folder/'training-xy.npy',xy)
        write(folder/'training-successes.json',successes)
        if a.preflight and a.save_intermediate_policy:
            from .policy_checkpoints import save_evaluation_checkpoint
            save_evaluation_checkpoint(learner,folder,step,config)
        t=time.monotonic();proof=checkpoint(learner,env,obs,folder,step,rng,config)
        timing['checkpoint']+=time.monotonic()-t
        summaries={}
        for mode in (['native','policy','zero_z'] if a.method=='optiq' else ['native','policy']):
            for fixed in (False,True):
                t=time.monotonic()
                label=mode+('-fixed' if fixed else '-natural')
                summaries[label]=evaluate(learner,a.task,folder,step,2 if a.preflight else a.final_eval_episodes,mode,fixed)
                timing['evaluation']+=time.monotonic()-t
        result=dict(completed=True,source_commit=source,method=a.method,task=a.task,
            steps=step,global_steps=step-warmup,updates=learner.updates,rnd_updates=learner.intrinsic.update_step,
            summaries=summaries,checkpoint=proof,timing=timing,seconds=time.monotonic()-started,
            training_successes=len(successes),training_episodes=episodes)
        write(folder/'result.json',result)
        # Keep final 100-episode evaluations separate from periodic 40-episode
        # metrics, including the reset distribution and zero-z control.
        final_summary = dict(completed=True, steps=step, updates=learner.updates,
                             final_evaluations=summaries)
        for label, summary in summaries.items():
            for key in ('success_rate', 'mean_return', 'episodes', 'goal_counts'):
                final_summary[f'final/{label}/{key}'] = summary[key]
        run.summary.update(final_summary)
    except BaseException as error:
        write(folder/'failure.json',dict(type=type(error).__name__,message=str(error),
            traceback=traceback.format_exc(),step=step,source_commit=source))
        raise
    finally:
        env.close();run.finish()


if __name__=='__main__':main()
