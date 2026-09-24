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

from .progress_reward import PROFILES, specification, progress_reward, success_bonus, progress_scale
from .settings import (REWARD_PROFILES, reward_description, CAMPAIGN, BUDGETS, REWARD, DENSE_REWARD, NUM_ENVS, EVAL_NUM_ENVS, UPDATES, WARMUP,
                       PREFLIGHT_STEPS, total_budget, expected_updates, WANDB_ENTITY, WANDB_PROJECT)


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def numeric_metrics(value, prefix):
    """Flatten finite diagnostic scalars for W&B; raw arrays stay on disk."""
    result = {}
    for key, item in value.items():
        name = f'{prefix}/{key}'
        if isinstance(item, dict):
            result.update(numeric_metrics(item, name))
        elif isinstance(item, (int, float)) and not isinstance(item, bool) and np.isfinite(item):
            result[name] = item
    return result


def evaluate(learner, task, folder, step, episodes, mode, fixed=False):
    from .envs import vector, transition
    from .learners import evaluation_rng
    origin_fixed = getattr(learner, 'eval_fixed_starts', False)
    fixed = bool(fixed or origin_fixed)
    label = f'{mode}-' + ('fixed' if fixed else 'natural')
    count = min(EVAL_NUM_ENVS, episodes)
    destination = folder/'evaluations'/f'{step:010d}'/label
    destination.mkdir(parents=True, exist_ok=True)
    paths, returns, goals, lengths, starts = [], [], [], [], []
    recorder = None
    if mode == 'policy' and getattr(learner, 'dynamics_profile', None) is not None:
        from .critic_diagnostics import EvaluationTraceRecorder
        recorder = EvaluationTraceRecorder()
    with evaluation_rng(learner, 700000 + step):
        env = vector(task, count, seed=87231, asynchronous=False, fixed=fixed,
                     reward_profile=learner.reward_profile,
                     random_init=False if origin_fixed else True if getattr(learner,'eval_random_starts',False) else None)
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
                if recorder is not None: recorder.start_batch(active)
                for _ in range(500 if task in ('v1','v2') else 700):
                    actions = learner.act(obs, mode)
                    nxt, reward, done, infos = env.step(actions)
                    final, terminal = transition(nxt, done, infos)
                    if recorder is not None:
                        recorder.record(obs, actions, reward, final, done, terminal, active)
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
                if recorder is not None: recorder.end_batch()
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
    result['start_distribution'] = ('original fixed origin, pose and velocity' if origin_fixed else
        'fixed first sampled full state' if fixed else
        'xy uniform[-2,2], original pose/velocity' if task=='v1' or getattr(learner,'eval_random_starts',False)
        else 'original fixed full state')
    if fixed: assert result['identical_initial_full_state']
    if origin_fixed:
        np.testing.assert_array_equal(np.asarray(starts)[:, :2], np.zeros((episodes, 2)))
    result['original_origin_fixed'] = origin_fixed
    result['reward_profile']=learner.reward_profile
    if learner.reward_profile in PROFILES:
        result['reward_specification']=specification(task,learner.reward_profile)
    if recorder is not None:
        from .dynamics_profiles import get_profile
        result['critic_diagnostics'] = recorder.save(learner, task, destination, step, mode,
            reward_multiplier=get_profile(learner.dynamics_profile)['reward_multiplier'],goals=goals)
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
    if config.get('dacer_enabled') is False and learner.method=='optiq':
        assert loaded['learner']['regulator'] is None
        assert loaded['learner']['regulator_rng'] is None
        assert loaded['learner']['regulator_count']==0
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
    elif config['reward_profile'] in PROFILES:
        # Replay XY is float32; reward was computed from float64 physics.
        reward_atol=2e-5*progress_scale(config['reward_profile'])
        previous=loaded['replay']['buf_obs'][:,:2].numpy()
        bonuses=success_bonus(positions,config['task'])
        expected,_,_=progress_reward(previous,positions,config['task'],config['reward_profile'],bonuses)
        boundary=np.min(np.abs(distances-.5),axis=1)<2e-5
        assert np.allclose(actual[~boundary],expected[~boundary],rtol=2e-6,atol=reward_atol)
        # Float32 replay XY can round across the goal radius. Account for the
        # original physics decision using terminal mask only on those rows.
        if boundary.any():
            terminal=loaded['replay']['buf_done'].numpy().ravel()[boundary]>0
            nearest=distances[boundary].argmin(axis=1)
            values=np.array([20 if tuple(g)==(-8,8) else 10 for g in target_goals])
            bonus=np.where(terminal,values[nearest],0.)
            expected_boundary,_,_=progress_reward(previous[boundary],positions[boundary],
                config['task'],config['reward_profile'],bonus)
            assert np.allclose(actual[boundary],expected_boundary,rtol=2e-6,atol=reward_atol)
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
        progress_replay_verified=config['reward_profile'] in PROFILES,
        progress_reward_absolute_tolerance=reward_atol if config['reward_profile'] in PROFILES else None,
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
    p.add_argument('--temperature-final',type=float)
    p.add_argument('--temperature-anneal-steps',type=int,default=1000000)
    p.add_argument('--temperature-decay',choices=['linear','log_linear'],default='linear')
    p.add_argument('--dacer-target-entropy-per-dim',type=float)
    p.add_argument('--dacer-interval-updates',type=int)
    p.add_argument('--dacer',choices=['on','off'])
    p.add_argument('--budget-steps',type=int)
    p.add_argument('--final-eval-episodes',type=int,default=100)
    p.add_argument('--interim-eval-episodes',type=int,default=EVAL_NUM_ENVS)
    p.add_argument('--save-intermediate-policy',action='store_true')
    p.add_argument('--eval-interval',type=int,default=250000)
    from .dynamics_profiles import PROFILES as DYNAMICS_PROFILES, get_profile
    p.add_argument('--dynamics-profile',choices=tuple(DYNAMICS_PROFILES))
    p.add_argument('--reward-profile',choices=REWARD_PROFILES,default='sparse')
    p.add_argument('--noveld',choices=['on','off'],default='on')
    p.add_argument('--eval-starts',choices=['upstream','random','fixed'],default='upstream')
    a = p.parse_args()
    assert a.eval_interval > 0
    if a.dynamics_profile is not None:
        dynamics = get_profile(a.dynamics_profile)
        assert a.method == 'optiq' and a.task in ('v3', 'v4')
        assert a.reward_profile == dynamics['reward_profile']
        assert a.temperature == dynamics['temperature']
        assert a.noveld == 'off' and a.dacer == 'off' and a.eval_starts == 'upstream'
        assert a.temperature_final is None and a.dacer_target_entropy_per_dim is None
    if a.reward_profile in PROFILES and a.noveld!='off':
        p.error("Progress profiles require --noveld off; intrinsic reward is a separate ablation")
    if a.temperature is not None:
        assert a.method == 'optiq' and a.temperature > 0
    if a.dacer_target_entropy_per_dim is not None:
        assert a.method=='optiq' and np.isfinite(a.dacer_target_entropy_per_dim)
        assert a.dacer!='off', 'A DACER target cannot be set while DACER is off'
    if a.dacer is not None:assert a.method=='optiq'
    if a.dacer_interval_updates is not None:
        assert a.method=='optiq' and a.dacer!='off' and a.dacer_interval_updates > 0
    temperature_schedule=None
    if a.temperature_final is not None:
        assert a.method=='optiq' and a.temperature is not None
        assert np.isfinite(a.temperature) and np.isfinite(a.temperature_final)
        assert 0 < a.temperature_final <= a.temperature and a.temperature_anneal_steps > 0
        temperature_schedule=dict(enabled=True,final_temperature=a.temperature_final,
            anneal_steps=a.temperature_anneal_steps,decay=a.temperature_decay)
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
    env = vector(a.task,NUM_ENVS,seed=0,reward_profile=a.reward_profile)
    obs = env.reset()
    import torch
    torch.set_num_threads(1);torch.manual_seed(0)
    random.seed(0);np.random.seed(0)
    assert torch.cuda.is_available()
    from .learners import Native,JaxLearner,audit
    planned_budget = a.budget_steps if a.budget_steps is not None else total_budget(a.task)
    if a.method in ('sac','dipo'):
        learner = Native(a.method,(env.single_observation_space,env.single_action_space),a.task,folder,
                         reward_profile=a.reward_profile,noveld=a.noveld=='on')
    else:
        learner = JaxLearner(a.method,(env.single_observation_space,env.single_action_space),
            a.task,folder,temperature=a.temperature,budget=planned_budget,
            reward_profile=a.reward_profile,noveld=a.noveld=='on',temperature_schedule=temperature_schedule,
            dacer_target_entropy_per_dim=a.dacer_target_entropy_per_dim,
            dacer_enabled=a.dacer!='off',dynamics_profile=a.dynamics_profile,
            dacer_interval_updates=a.dacer_interval_updates)
    initial = audit(learner)
    learner.eval_random_starts = a.eval_starts=='random'
    learner.eval_fixed_starts = a.eval_starts=='fixed' or (a.eval_starts=='upstream' and a.task!='v1')
    budget = PREFLIGHT_STEPS if a.preflight else planned_budget
    warmup = WARMUP
    config = dict(source_commit=source,upstream_commit='7edd06c4799abbab0f8fa534c21deb56253b018e',
        method=a.method,task=a.task,seed=0,preflight=a.preflight,steps=budget,
        source_dependencies=dependencies,
        wandb_entity=WANDB_ENTITY,wandb_project=WANDB_PROJECT,
        num_envs=NUM_ENVS,batch_size=4096,updates_per_vector_step=UPDATES,updates_per_transition=1/32,
        expected_updates=expected_updates(budget),warmup_transitions=warmup,
        upstream_max_step=BUDGETS[a.task],native_global_steps=budget-warmup,
        reward=reward_description(a.reward_profile),
        reward_specification=specification(a.task,a.reward_profile),
        reward_profile=a.reward_profile,noveld_enabled=a.noveld=='on',
        eval_starts=a.eval_starts,effective_eval_starts='fixed' if learner.eval_fixed_starts else 'random',
        primary_trajectory='policy-fixed' if learner.eval_fixed_starts else 'policy-natural',
        noveld_coefficient=.01 if a.noveld=='on' else 0.,temperature=a.temperature,
        temperature_schedule=temperature_schedule,
        dacer_target_entropy_per_dim=a.dacer_target_entropy_per_dim,
        dacer_interval_updates=int(learner.model.regulator_cfg.interval_updates) if a.method=='optiq' else None,
        dacer_enabled=bool(learner.model.regulator_enabled) if a.method=='optiq' else None,
        native=learner.config,random_init=a.task=='v1',eval_interval=a.eval_interval,
        eval_num_envs=EVAL_NUM_ENVS,interim_eval_episodes=a.interim_eval_episodes,
        final_eval_episodes=a.final_eval_episodes,
        save_intermediate_policy=a.save_intermediate_policy,
        policy_checkpoint_interval=a.eval_interval if a.save_intermediate_policy else None,
        policy_checkpoint_kind='evaluation-only; model/optimizer/RNG, no replay or simulator',
        checkpoint='final only',step_definition='step includes warmup; global_steps excludes warmup as upstream; stop global_steps>max_step',
        evaluation=('primary=policy-fixed at original origin/pose/velocity; ' if learner.eval_fixed_starts else 'primary=policy-natural; ')+
            'policy: direct draws, no extra exploration noise; native: SAC mean, MFPO Q-best-of10, OptiQ random-z mu-only, DIPO native diffusion',
        runtime=dict(python=os.sys.version,torch=torch.__version__,numpy=np.__version__))
    if a.dynamics_profile is not None:
        config.update(dynamics_profile=a.dynamics_profile,dynamics_settings=dynamics,
                      reward_multiplier=dynamics['reward_multiplier'],
                      diagnostic_interval=25000,
                      comparison_reward='100*(d(current)-d(next)); convert rewards/Q by reward_multiplier',
                      requested_global_budget=500000)
    write(folder/'config.json',config)
    import wandb
    temp_name=f'-T{a.temperature:g}' if a.temperature is not None else ''
    if temperature_schedule is not None:
        temp_name+=f'-to{a.temperature_final:g}-{a.temperature_decay}-{a.temperature_anneal_steps}postwarmup'
    if a.dacer_target_entropy_per_dim is not None:
        temp_name+=f'-Hdim{a.dacer_target_entropy_per_dim:g}'
    if a.dacer_interval_updates is not None:temp_name+=f'-Hinterval{a.dacer_interval_updates}'
    if a.dacer is not None:temp_name+='-dacer'+a.dacer
    if a.dynamics_profile is not None:temp_name+='-dyn-'+a.dynamics_profile
    run_name=f'{a.task}-{a.method}{temp_name}-{a.reward_profile}-noveld{a.noveld}-s0-{budget}steps'
    run = wandb.init(entity=WANDB_ENTITY,project=WANDB_PROJECT,group=os.environ.get('OPTIQ_CAMPAIGN',CAMPAIGN),
        name=run_name,dir=str(folder),config=config,
        mode='disabled' if a.preflight else os.environ.get('WANDB_MODE','online'))
    if not a.preflight:
        offline=os.environ.get('WANDB_MODE','online')=='offline'
        write(folder/'wandb.json',dict(id=run.id,url=None if offline else run.url,
            entity=WANDB_ENTITY,project=WANDB_PROJECT,
            mode='offline' if offline else 'online',sync_pending=offline))
    rng=np.random.default_rng(0);step=0;started=time.monotonic();next_eval=a.eval_interval
    next_diagnostic=25000
    timing=dict(collection=0.,learner=0.,evaluation=0.,checkpoint=0.)
    xy=np.empty((budget,2),np.float32);successes=[];episodes=0;info={}
    def progress():
        record=dict(step=step,global_steps=max(0,step-warmup),updates=learner.updates,rnd_updates=learner.intrinsic.update_step,
            seconds=time.monotonic()-started,episodes=episodes,successes=len(successes),
            **{f'seconds/{k}':v for k,v in timing.items()})
        if temperature_schedule is not None:
            record.update(temperature=float(info.get('train/temperature',a.temperature)),
                temperature_anneal_progress=float(info.get('train/temperature_anneal_progress',0.)),
                temperature_post_warmup_steps=max(0,step-warmup))
        if a.dacer_target_entropy_per_dim is not None:
            m=learner.model
            record.update(temperature=float(info.get('train/temperature',learner.config['alg']['actor']['temperature'])),
                dacer_target_entropy_per_dim=float(m.regulator_cfg.target_entropy_per_dim),
                dacer_target_entropy=float(m.regulator_cfg.target_entropy_per_dim)*env.single_action_space.shape[0],
                dacer_updates=int(m.regulator_count),dacer_noise_std=float(m.regulator_noise_std),
                dacer_interval_updates=int(m.regulator_cfg.interval_updates),
                dacer_entropy_proxy=float(m.regulator_entropy) if np.isfinite(m.regulator_entropy) else None)
        if a.dacer=='off':
            assert not learner.model.regulator_enabled
            record.update(dacer_enabled=False,dacer_updates=0,dacer_noise_std=0.)
        if a.dynamics_profile is not None:
            from .dynamics_profiles import expected_actor_updates
            actor_updates = int(learner.model.policy.actor_state.step)
            critic_updates = int(learner.model.policy.qf_state.step)
            assert actor_updates == expected_actor_updates(learner.updates, a.dynamics_profile)
            assert critic_updates == learner.updates == learner.model._n_updates
            record.update(actor_updates=actor_updates,critic_updates=critic_updates,
                          critic_tau=dynamics['tau'],policy_delay=dynamics['policy_delay'])
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
            if a.dynamics_profile is not None and (step>=next_diagnostic or (a.preflight and step==budget)):
                from .critic_diagnostics import replay_diagnostics
                t=time.monotonic()
                diagnostic=replay_diagnostics(learner,step,folder,
                    reward_multiplier=dynamics['reward_multiplier'])
                run.log(numeric_metrics(diagnostic,'diagnostic/replay'),step=step)
                timing['evaluation']+=time.monotonic()-t
                next_diagnostic+=25000
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
                    if 'critic_diagnostics' in s:
                        run.log(numeric_metrics(s['critic_diagnostics'],'diagnostic/policy'),step=step)
                timing['evaluation']+=time.monotonic()-t
                next_eval+=a.eval_interval
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
            for fixed in ((True,) if learner.eval_fixed_starts else (False,) if a.eval_starts=='upstream' else (False,True)):
                t=time.monotonic()
                label=mode+('-fixed' if fixed else '-natural')
                summaries[label]=evaluate(learner,a.task,folder,step,2 if a.preflight else a.final_eval_episodes,mode,fixed)
                timing['evaluation']+=time.monotonic()-t
        result=dict(completed=True,source_commit=source,method=a.method,task=a.task,
            steps=step,global_steps=step-warmup,updates=learner.updates,rnd_updates=learner.intrinsic.update_step,
            summaries=summaries,checkpoint=proof,timing=timing,seconds=time.monotonic()-started,
            training_successes=len(successes),training_episodes=episodes)
        if temperature_schedule is not None:
            result['temperature_schedule']=temperature_schedule
            result['final_temperature']=float(info['train/temperature'])
        if a.dacer_target_entropy_per_dim is not None:
            result['dacer_target_entropy_per_dim']=float(learner.model.regulator_cfg.target_entropy_per_dim)
            result['dacer_regulator']=json.loads((folder/'dacer_regulator.json').read_text())
            result['dacer_interval_updates']=int(learner.model.regulator_cfg.interval_updates)
            result['dacer_next_update']=int(learner.model.regulator_next_update)
        if a.dacer=='off':
            assert not (folder/'dacer_regulator.json').exists()
            result.update(dacer_enabled=False,dacer_updates=0,dacer_noise_std=0.)
        if a.dynamics_profile is not None:
            from .dynamics_profiles import expected_actor_updates
            actor_steps=int(learner.model.policy.actor_state.step)
            critic_steps=int(learner.model.policy.qf_state.step)
            assert actor_steps==expected_actor_updates(learner.updates,a.dynamics_profile)
            assert critic_steps==learner.updates==learner.model._n_updates
            verification=dict(verified=True,dynamics_profile=a.dynamics_profile,settings=dynamics,
                actor_updates=actor_steps,critic_updates=critic_steps,
                runtime_tau=float(learner.model.tau),runtime_policy_delay=int(learner.model.policy_delay),
                initial_parameters=initial,source_commit=source,
                direct_policy_diagnostics=[str(p.relative_to(folder)) for p in
                    sorted((folder/'evaluations').glob('*/policy-*/critic-diagnostics.json'))])
            assert verification['runtime_tau']==dynamics['tau']
            assert verification['runtime_policy_delay']==dynamics['policy_delay']
            write(folder/'dynamics-verification.json',verification)
            result['dynamics_verification']=verification
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
