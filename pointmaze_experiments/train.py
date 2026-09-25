import argparse
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.logger import configure

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def action(model, obs, mode, key=None):
    policy = model.policy
    if key is None:
        policy.reset_noise()
        key = policy.noise_key
    return np.clip(np.asarray(policy.sample_action(policy.actor_state,
        np.asarray(obs, dtype=np.float32)[None], key, deterministic=False,
        sample_conditional_noise=(mode == 'full')))[0], -1., 1.)


def evaluate(model, env, folder, step, episodes, seed):
    import jax
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    target = folder / 'evaluations' / f'{step:07d}'
    target.mkdir(parents=True, exist_ok=False)
    results = {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for mode, ax in zip(('mu', 'full'), axes):
        key = jax.random.PRNGKey(seed + 30000)
        paths, goals, returns = [], [], []
        for i in range(episodes):
            obs, _ = env.reset(seed=seed + 50000 + i)
            xy, total = [obs[:2].copy()], 0.
            for _ in range(env.max_steps):
                key, subkey = jax.random.split(key)
                obs, reward, terminal, trunc, info = env.step(action(model, obs, mode, subkey))
                xy.append(obs[:2].copy()); total += reward
                if terminal or trunc: break
            paths.append(np.array(xy)); goals.append(int(info['target'])); returns.append(total)
        counts = np.bincount(goals, minlength=env.num_goals + 1)[1:]
        p = counts[counts > 0] / max(counts.sum(), 1)
        results[mode] = dict(success_rate=float(np.mean(np.array(goals)>0)),
            mean_return=float(np.mean(returns)), reached_goals=int((counts>0).sum()),
            goal_counts=counts.tolist(), episodes=episodes,
            goal_entropy=float(-(p*np.log(p)).sum()) if len(p) else None)
        env.plot(ax)
        colors = ['gray', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red',
                  'tab:purple','tab:brown','tab:pink','tab:cyan']
        arr = np.full((episodes, env.max_steps + 1, 2), np.nan, dtype=np.float32)
        for i, (xy, goal) in enumerate(zip(paths,goals)):
            arr[i,:len(xy)] = xy
            ax.plot(xy[:,0],xy[:,1],color=colors[goal],alpha=.4,lw=.7)
        ax.scatter(*env.reset_pos, color='red', s=25, zorder=5)
        ax.set_title(f'{mode}: {results[mode]["success_rate"]:.0%} success, {sum(counts>0)}/{env.num_goals} goals')
        np.savez_compressed(target / f'{mode}.npz', xy=arr, goals=goals, returns=returns)
    fig.suptitle(f'iBOLT | {step:,} environment steps')
    fig.tight_layout(); fig.savefig(target / 'rollouts.png', dpi=160); plt.close(fig)
    write(target / 'summary.json', results)
    return results, target / 'rollouts.png'


def checkpoint(model, folder, step):
    import flax.serialization as fs
    import jax
    p = model.policy
    state = dict(actor=p.actor_state, critic=p.qf_state, target_actor=p.target_actor_state)
    data = fs.to_bytes(state)
    restored = fs.from_bytes(state, data)
    assert all(np.array_equal(np.asarray(a),np.asarray(b)) for a,b in
        zip(jax.tree_util.tree_leaves(state),jax.tree_util.tree_leaves(restored)))
    (folder/f'policy-{step:07d}.msgpack').write_bytes(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--maze', default='simple', choices=['simple','medium','hard'])
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--temperature', type=float, default=1.)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eval-every', type=int, default=5000)
    parser.add_argument('--eval-episodes', type=int, default=100)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(Path(args.source).resolve()))
    from envs.mgmaze.point_maze import MultiGoalPointMaze
    env=MultiGoalPointMaze(maze_map=args.maze,reward_type='sparse',maze_eval_mode=False)
    test_env=MultiGoalPointMaze(maze_map=args.maze,reward_type='sparse',maze_eval_mode=False)
    assert env.action_space.shape == (2,) and env.observation_space.shape == (4,)
    assert np.all(env.action_space.low == -1) and np.all(env.action_space.high == 1)
    assert env.compute_reward(env.reset_pos,env.reset_pos)==0
    for goal in env.maze.unique_goal_locations:
        assert env.compute_reward(env.reset_pos,goal)==100 and env.compute_terminated(goal)[0]
    torch.set_num_threads(1); torch.manual_seed(args.seed)
    random.seed(args.seed); np.random.seed(args.seed)
    spec=importlib.util.spec_from_file_location('trg_train',ROOT/'analysis_tools/experiments/20260921_gmm_trg_sweep/train.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    cfg=module.compose_config(['benchmark=ant',f'seed={args.seed}',
        f'alg.actor.temperature={args.temperature}','dacer.enabled=false',f'output_root={out}'])
    cfg.env_name='DrAC-PointMaze-'+args.maze; cfg.task='pointmaze'
    cfg.alg.batch_size=256;cfg.alg.learning_starts=args.warmup
    cfg.alg.actor.learning_starts=args.warmup;cfg.alg.utd=1
    cfg.wandb.project='jaehun-drac-pointmaze'
    model=module.runner.OptiQDIME('MlpPolicy',env=env,cfg=cfg,
        model_save_path=None,save_every_n_steps=args.steps)
    model.set_logger(configure(str(out/'learner'),['csv']))
    model._total_timesteps=args.steps
    replay=ReplayBuffer(1000000,env.observation_space,env.action_space,
        device='cpu',handle_timeout_termination=False)
    model.replay_buffer=replay
    from omegaconf import OmegaConf
    import jax
    assert jax.default_backend()=='gpu',jax.devices()
    config=dict(**vars(args), learner=OmegaConf.to_container(cfg,resolve=True),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        upstream_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.source,text=True).strip(),
        reward='official sparse: goal100, otherwise0', num_envs=1,utd=1,
        observation='x,y,vx,vy; no goal ID', goals=[g.tolist() for g in env.maze.unique_goal_locations],
        primary_eval='random z, mu-only; full conditional-noise policy separately',
        gpu=str(jax.devices()),expert_data=False,discriminator=False)
    write(out/'config.json',config)
    run=None
    if not args.preflight:
        import wandb
        run=wandb.init(entity='OptiQ',project='jaehun-drac-pointmaze',
            name=f'ibolt-{args.maze}-T{args.temperature:g}-s{args.seed}',config=config,dir=str(out))
        write(out/'wandb.json',dict(id=run.id,url=run.url))
    obs,_=env.reset(seed=args.seed);rng=np.random.default_rng(args.seed)
    begin=time.monotonic(); visits=np.zeros(env.num_goals,dtype=int);episodes=0
    initial_actor=np.asarray(jax.tree_util.tree_leaves(model.policy.actor_state.params)[0]).copy()
    with (out/'metrics.jsonl').open('a',buffering=1) as log:
        for step in range(1,args.steps+1):
            a=rng.uniform(-1,1,2).astype('float32') if step<=args.warmup else action(model,obs,'full')
            nxt,r,terminated,truncated,info=env.step(a)
            assert np.isfinite(nxt).all() and r in (0,100)
            replay.add(obs[None],nxt[None],a[None],np.array([r]),np.array([terminated]),[{}])
            obs=nxt
            if terminated or truncated:
                episodes+=1
                if info['target']>0:visits[info['target']-1]+=1
                obs,_=env.reset()
            if step>args.warmup:
                model.num_timesteps=step;model._current_progress_remaining=1-step/args.steps
                model.train(batch_size=256,gradient_steps=1)
            if step%100==0 or step==args.steps:
                metrics={k:float(v) for k,v in model.logger.name_to_value.items()
                         if isinstance(v,(int,float,np.number))}
                assert all(np.isfinite(v) for v in metrics.values()),metrics
                row=dict(env_steps=step,updates=int(model._n_updates),episodes=episodes,
                    train_goal_visits=visits.tolist(),wall_seconds=time.monotonic()-begin,**metrics)
                log.write(json.dumps(row)+'\n');write(out/'progress.json',row)
                if run:run.log(row,step=step)
                if step%1000==0 or step==args.steps:print(json.dumps(row),flush=True)
            if step%args.eval_every==0 or step==args.steps:
                checkpoint(model,out,step)
                result,image=evaluate(model,test_env,out,step,args.eval_episodes,args.seed)
                if run:
                    summary={f'eval/{mode}/{k}':v for mode,stats in result.items()
                             for k,v in stats.items() if isinstance(v,(int,float))}
                    run.log(dict(summary,**{'eval/rollouts':wandb.Image(str(image))}),step=step)
                print('EVAL',step,result,flush=True)
    assert model._n_updates==args.steps-args.warmup
    assert any(np.any(np.asarray(x)!=0) for x in jax.tree_util.tree_leaves(model.policy.actor_state.params))
    changed=not np.array_equal(initial_actor,np.asarray(jax.tree_util.tree_leaves(model.policy.actor_state.params)[0]))
    assert changed,'Actor did not update'
    write(out/'completed.json',dict(steps=args.steps,updates=int(model._n_updates),actor_changed=changed))
    if run:run.finish()
    env.close();test_env.close()


if __name__=='__main__':main()
