"""Vectorized sparse Adroit with native DIPO/MEOW/MFPO learner updates."""
import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import random
import time
import traceback
import numpy as np
from .run import atomic, guard, make_env


def evaluate(agent, method, task, seed, episodes):
    from .baselines.adapters import action
    rng = np.random.default_rng(np.random.SeedSequence([seed, 4404]))
    env_seeds, policy_seeds = rng.integers(0, 2**30, episodes), rng.integers(0, 2**30, episodes)
    env = make_env(task)
    saved_numpy, saved_python = np.random.get_state(), random.getstate()
    rows = []
    try:
        for env_seed, policy_seed in zip(env_seeds, policy_seeds):
            obs, _ = env.reset(seed=int(env_seed))
            rewards, successes = [], []
            with agent.evaluation_rng(int(policy_seed)):
                while True:
                    a = action(agent, method, np.asarray(obs, np.float32)[None], evaluation=True)[0]
                    obs, reward, terminated, truncated, info = env.step(a)
                    rewards.append(float(reward)); successes.append(bool(info['success']))
                    if terminated or truncated:
                        break
            rows.append(dict(return_=sum(rewards), length=len(rewards), success_any=any(successes),
                             success_final=successes[-1], success_steps=sum(successes), success_25_steps=sum(successes) >= 25))
    finally:
        env.close()
        np.random.set_state(saved_numpy); random.setstate(saved_python)
    assert np.isfinite([r['return_'] for r in rows]).all()
    return dict(primary_mode='policy', env_seeds=env_seeds.tolist(), policy_seeds=policy_seeds.tolist(),
        policy=dict(mean_return=float(np.mean([r['return_'] for r in rows])),
            std_return=float(np.std([r['return_'] for r in rows])),
            success_rate=float(np.mean([r['success_any'] for r in rows])),
            final_success_rate=float(np.mean([r['success_final'] for r in rows])),
            success_25_steps_rate=float(np.mean([r['success_25_steps'] for r in rows])), episodes=rows))


def finite(agent, method):
    if method == 'dipo':
        return agent.finite()
    if method == 'meow':
        import torch
        return all(torch.isfinite(p).all().item() for p in agent.policy.parameters())
    import jax
    return all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(agent.agent))


def run(spec, smoke=False):
    guard()
    import torch
    torch.set_num_threads(1)
    from .baselines.adapters import create, action, DIPOReplay
    from .demo_replay import DemonstrationReplay
    from .vector_env import GroupedAdroitVecEnv
    method, task, seed = spec['method'], spec['task'], spec['seed']
    from .verified_demos import resolve
    demo_path, demo_proof = resolve(task, spec)
    assert method in ('dipo', 'meow', 'mfpo')
    total, warmup = (13568 if smoke else 5000192), 12800
    expected = (total-warmup)//8
    output = Path(spec['output']); output.mkdir(parents=True, exist_ok=False)
    if demo_proof is not None:
        atomic(output/'demonstration-validation.json', demo_proof)
    (output/'checkpoints').mkdir()
    atomic(output/'spec.json', dict(spec, smoke=smoke))
    atomic(output/'progress.json', dict(status='initializing', steps=0, updates=0, observed_unix=time.time()))
    vec = agent = None
    started = time.time()
    try:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if method == 'mfpo':
            import jax
            assert jax.default_backend() == 'gpu'
        else:
            assert torch.cuda.is_available()
        vec = GroupedAdroitVecEnv(task, 256, [[], [], [], []])
        replay_class = DIPOReplay if method == 'dipo' else DemonstrationReplay
        replay = replay_class(32768 if smoke else 1000000, vec.observation_space, vec.action_space,
            n_envs=256, device='cpu', demo_seed=seed+9214,
            demo_path=str(demo_path))
        agent = create(method, seed, output, total, vec.observation_space.shape[0], vec.action_space, replay)
        from .baselines.architecture import audit
        architecture = audit(agent, method)
        config = dict(method=method, task=task, seed=seed, steps=total, target_steps=5000000,
            expected_updates=expected, num_envs=256, batch_size=256, utd=.125, warmup=warmup,
            gradient_steps_per_collect=32, demonstration_ratio=.5, sparse=True, her=False,
            normalize_observations=False, normalize_rewards=False, offline_pretraining_updates=0,
            evaluation_every=50176, evaluation_episodes=1 if smoke else 10,
            final_episodes=1 if smoke else 100, native_config=agent.config, architecture=architecture)
        if demo_proof is not None:
            config.update(demonstration_manifest=spec['demonstration_manifest'],
                demonstration_manifest_sha256=spec['demonstration_manifest_sha256'],
                demonstration_dataset_sha256=demo_proof['dataset_sha256'],
                demonstration_provenance=demo_proof['provenance'])
        atomic(output/'config.json', config)
        def record(step, episodes):
            result = dict(steps=step, updates=agent.updates, **evaluate(agent, method, task, seed, episodes))
            with (output/'evaluations.jsonl').open('a') as stream:
                stream.write(json.dumps(result, allow_nan=False)+'\n')
            print(json.dumps(dict(method=method, task=task, steps=step,
                success_rate=result['policy']['success_rate'], mean_return=result['policy']['mean_return'])), flush=True)
        record(0, 1 if smoke else 10)
        vec.seed(seed); vec.action_space.seed(seed)
        obs = vec.reset()
        for step in range(256, total+1, 256):
            actions = (np.stack([vec.action_space.sample() for _ in range(256)]) if step <= warmup else action(agent, method, np.asarray(obs, np.float32)))
            assert actions.shape == (256, vec.action_space.shape[0]) and np.isfinite(actions).all()
            nxt, rewards, done, infos = vec.step(actions)
            terminal_nxt = np.stack([info.get('terminal_observation', nxt[k]) for k, info in enumerate(infos)])
            replay.add(obs, terminal_nxt, actions, rewards, done, infos)
            obs = nxt
            metrics = {}
            if step > warmup:
                for _ in range(32):
                    metrics = agent.update(step)
                assert agent.updates == (step-warmup)//8
            if step % 1024 == 0 or step == total:
                assert all(math.isfinite(float(v)) for v in metrics.values()) and finite(agent, method)
                atomic(output/'progress.json', dict(status='running', steps=step, updates=agent.updates,
                    metrics=metrics, discovery=replay.discovery(), observed_unix=time.time(), elapsed_seconds=time.time()-started))
            if not smoke and step % 50176 == 0 and step < total:
                record(step, 10)
            if step % 100096 == 0:
                agent.save(output/'checkpoints', step)
        assert agent.updates == expected and replay.last_mix['per_optimizer_minibatch'] == [128, 128]
        assert finite(agent, method)
        assert audit(agent, method) == architecture
        agent.save(output/'checkpoints', total)
        record(total, 1 if smoke else 100)
        result = dict(completed=True, smoke=smoke, method=method, task=task, seed=seed,
            steps=total, updates=agent.updates, num_envs=256, observed_utd=agent.updates/(total-warmup),
            replay_mix=replay.last_mix, discovery=replay.discovery(), backend='gpu',
            elapsed_seconds=time.time()-started, observed_unix=time.time())
        atomic(output/'completed.json', result); atomic(output/'progress.json', dict(result, status='completed'))
    except BaseException as error:
        atomic(output/'failure.json', dict(error=repr(error), traceback=traceback.format_exc()))
        raise
    finally:
        if vec is not None:
            vec.close()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--spec', type=Path, required=True); p.add_argument('--smoke', action='store_true')
    a = p.parse_args(); run(json.loads(a.spec.read_text()), a.smoke)


if __name__ == '__main__':
    main()
