"""Online-only runner using the shared JAX SAC and unchanged SB3 TD3 learner."""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time
import traceback

from .config import configuration, run_id

ROOT = Path(__file__).resolve().parents[1]


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def source_provenance():
    p = json.loads((ROOT / 'source-provenance.json').read_text())
    assert p['branch'] == 'heejoon-rebuttal' and len(p['commit']) == 40
    for name, digest in p['files'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    return p


@contextmanager
def evaluation_rng(model, seed):
    import numpy as np
    import torch
    py, np_state = random.getstate(), np.random.get_state()
    jax_key = model.jax_state.key if hasattr(model, 'jax_state') else None
    devices = [torch.cuda.current_device()] if model.device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if jax_key is not None:
                import jax
                model.jax_state = model.jax_state.replace(key=jax.random.PRNGKey(seed))
            yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)
        if jax_key is not None:
            model.jax_state = model.jax_state.replace(key=jax_key)


def evaluate(model, c, episodes, random_actions=False):
    import numpy as np
    from .env import make_env
    assert episodes % 20 == 0
    reset_rng = np.random.default_rng(np.random.SeedSequence([c['seed'], 4404]))
    reset_seeds = reset_rng.integers(0, 2**30, episodes).tolist()
    modes = ['uniform_random'] if random_actions else (
        ['deterministic', 'stochastic'] if c['method'] == 'sac' else ['deterministic'])
    envs = [make_env(c['task']) for _ in range(20)]
    result = {'episodes': episodes, 'reset_seeds': reset_seeds, 'primary': modes[0]}
    try:
        for mode in modes:
            rows = []
            uniform = np.random.default_rng(np.random.SeedSequence([c['seed'], 7090]))
            with evaluation_rng(model, c['seed'] + 900_000):
                for start in range(0, episodes, len(envs)):
                    obs = np.stack([e.reset(seed=s)[0] for e, s in
                                    zip(envs, reset_seeds[start:start + len(envs)], strict=True)])
                    returns = np.zeros(len(envs))
                    counts = np.zeros(len(envs), dtype=np.int64)
                    for t in range(c['horizon']):
                        if random_actions:
                            actions = uniform.uniform(-1, 1, size=(len(envs), *envs[0].action_space.shape)).astype(np.float32)
                        else:
                            actions, _ = model.predict(obs, deterministic=mode == 'deterministic')
                        transitions = [e.step(a) for e, a in zip(envs, actions, strict=True)]
                        obs = np.stack([x[0] for x in transitions])
                        success = np.array([bool(x[4]['success']) for x in transitions])
                        rewards = np.array([x[1] for x in transitions])
                        assert np.allclose(rewards, np.where(success, 10., -.1))
                        assert all(not x[2] and bool(x[3]) == (t == c['horizon'] - 1) for x in transitions)
                        assert np.isfinite(obs).all()
                        returns += rewards
                        counts += success
                    rows.extend(dict(return_=float(r), success_any=bool(n), success_final=bool(f),
                                     success_steps=int(n), success_25_steps=bool(n >= 25))
                                for r, n, f in zip(returns, counts, success, strict=True))
            result[mode] = {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
            result[mode]['episode_records'] = rows
    finally:
        for e in envs:
            e.close()
    return result


def create_model(c, env):
    import numpy as np
    from stable_baselines3 import TD3
    from stable_baselines3.common.noise import NormalActionNoise
    common = dict(learning_rate=c['learning_rate'], buffer_size=c['replay_capacity'],
                  learning_starts=c['warmup'], batch_size=256, tau=c['tau'], gamma=c['gamma'],
                  train_freq=1, gradient_steps=c['gradient_steps'], seed=c['seed'],
                  policy_kwargs=dict(net_arch=c['hidden']), verbose=0)
    if c['method'] == 'sac':
        import jax
        from baselines.jax_sac import JaxSAC
        assert jax.default_backend() == 'gpu', 'Reported SAC must use JAX GPU'
        model = JaxSAC('MlpPolicy', env, ent_coef='auto', target_entropy='auto', **common)
    else:
        import torch
        assert torch.cuda.is_available(), 'Reported TD3 must use CUDA'
        dim = env.action_space.shape[0]
        noise = NormalActionNoise(np.zeros(dim), np.full(dim, c['td3_exploration_std']))
        model = TD3('MlpPolicy', env, device='cuda', action_noise=noise,
                    policy_delay=2, target_policy_noise=.2, target_noise_clip=.5, **common)
    from stable_baselines3.common.buffers import ReplayBuffer
    assert type(model.replay_buffer) is ReplayBuffer
    assert model.replay_buffer.size() == 0 and model.replay_buffer.handle_timeout_termination
    return model


def train(c, output):
    import numpy as np
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    from .vector_env import GroupedAdroitVecEnv

    provenance = source_provenance()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    affinity = sorted(os.sched_getaffinity(0))
    worker_cpus = [[affinity[-4 + i]] for i in range(4)] if len(affinity) >= 5 else [[], [], [], []]
    spec = dict(c, run_id=run_id(c), source_commit=provenance['commit'],
                source_branch=provenance['branch'], pid=os.getpid(),
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), cpu_affinity=affinity,
                worker_cpus=worker_cpus, started_unix=time.time(),
                packages={n: importlib.metadata.version(n) for n in
                          ('gymnasium', 'gymnasium-robotics', 'mujoco', 'stable-baselines3', 'torch', 'jax', 'flax')})
    atomic(output / 'spec.json', spec)
    atomic(output / 'progress.json', dict(status='initializing', steps=0, updated_unix=time.time()))
    env = model = None
    collection_seconds = update_seconds = evaluation_seconds = 0.
    latest_eval = None
    last_progress = 0.

    class Trace(BaseCallback):
        def __init__(self):
            super().__init__()
            self.first_success = None
            self.success_transitions = self.warmup_success_transitions = 0
            self.episodes = self.successful_episodes = 0
            self.episode_successes = np.zeros(c['num_envs'], np.int64)

        def _on_step(self):
            successes = np.array([bool(x['success']) for x in self.locals['infos']])
            self.success_transitions += int(successes.sum())
            if self.num_timesteps <= c['warmup']:
                self.warmup_success_transitions += int(successes.sum())
            if successes.any() and self.first_success is None:
                self.first_success = dict(vector_step_end=self.num_timesteps,
                                          environment_indices=np.flatnonzero(successes).tolist(),
                                          in_warmup=self.num_timesteps <= c['warmup'])
                atomic(output / 'first-success.json', self.first_success)
            self.episode_successes += successes
            for i in np.flatnonzero(self.locals['dones']):
                self.episodes += 1
                self.successful_episodes += int(self.episode_successes[i] > 0)
                self.episode_successes[i] = 0
            return True

    trace = Trace()

    def progress(status):
        elapsed = time.monotonic() - started
        row = dict(status=status, steps=model.num_timesteps if model else 0,
                   updates=model._n_updates if model else 0, elapsed_seconds=elapsed,
                   collection_seconds=collection_seconds, update_seconds=update_seconds,
                   evaluation_seconds=evaluation_seconds, first_success=trace.first_success,
                   success_transitions=trace.success_transitions,
                   warmup_success_transitions=trace.warmup_success_transitions,
                   completed_episodes=trace.episodes, successful_episodes=trace.successful_episodes,
                   last_evaluation=latest_eval, updated_unix=time.time(), pid=os.getpid())
        if model:
            row['transitions_per_second'] = model.num_timesteps / max(elapsed, 1e-6)
            row['learner'] = {k: float(v) for k, v in model.logger.name_to_value.items()
                              if k.startswith('train/') and np.isscalar(v) and np.isfinite(v)}
        atomic(output / 'progress.json', row)
        with (output / 'progress.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps(row, allow_nan=False), flush=True)

    def do_eval(episodes, label, random_actions=False):
        nonlocal evaluation_seconds, latest_eval
        before = time.monotonic()
        result = evaluate(model, c, episodes, random_actions)
        evaluation_seconds += time.monotonic() - before
        result.update(step=model.num_timesteps, updates=model._n_updates,
                      kind=label, elapsed_seconds=time.monotonic() - started)
        atomic(output / 'evaluations' / f'{model.num_timesteps:09d}-{label}.json', result)
        if not random_actions:
            latest_eval = dict(step=model.num_timesteps, episodes=episodes,
                               success=result['deterministic']['success_any'],
                               final_success=result['deterministic']['success_final'],
                               success_25_steps=result['deterministic']['success_25_steps'])
            if 'stochastic' in result:
                latest_eval['stochastic_success'] = result['stochastic']['success_any']
        progress('running')

    def checkpoint(final=False):
        folder = output / 'checkpoints'
        folder.mkdir(exist_ok=True)
        model.save(folder / f'policy-{model.num_timesteps:09d}.zip')
        if final:
            model.save_replay_buffer(folder / 'final-replay.pkl')

    try:
        env = GroupedAdroitVecEnv(c['task'], num_envs=c['num_envs'], worker_cpus=worker_cpus)
        model = create_model(c, env)
        model.set_logger(configure(str(output / 'learner'), ['csv']))
        total, callback = model._setup_learn(c['steps'], callback=trace, reset_num_timesteps=True)
        callback.on_training_start(locals(), globals())
        do_eval(c['eval_episodes'], 'initial')
        do_eval(c['eval_episodes'], 'uniform-random', random_actions=True)
        next_eval, next_checkpoint = c['eval_every'], c['checkpoint_every']
        while model.num_timesteps < total:
            before = time.monotonic()
            rollout = model.collect_rollouts(env, train_freq=model.train_freq,
                action_noise=model.action_noise, callback=callback,
                learning_starts=c['warmup'], replay_buffer=model.replay_buffer, log_interval=None)
            collection_seconds += time.monotonic() - before
            assert rollout.continue_training
            if model.num_timesteps > c['warmup']:
                before = time.monotonic()
                model.train(gradient_steps=c['gradient_steps'], batch_size=256)
                for key, value in model.logger.name_to_value.items():
                    if key.startswith('train/') and np.isscalar(value):
                        assert np.isfinite(value), (key, value)
                update_seconds += time.monotonic() - before
            if model.num_timesteps >= next_eval and model.num_timesteps < total:
                do_eval(c['eval_episodes'], 'regular')
                next_eval += c['eval_every']
            if model.num_timesteps >= next_checkpoint and model.num_timesteps < total:
                checkpoint()
                next_checkpoint += c['checkpoint_every']
            if time.monotonic() - last_progress > 60:
                progress('running')
                model.logger.dump(model.num_timesteps)
                last_progress = time.monotonic()
        callback.on_training_end()
        assert model.num_timesteps == c['steps'] and model._n_updates == c['expected_updates']
        # Native horizons are timeouts, not terminal Bellman targets.
        count = model.replay_buffer.buffer_size if model.replay_buffer.full else model.replay_buffer.pos
        dones = model.replay_buffer.dones[:count]
        timeouts = model.replay_buffer.timeouts[:count]
        assert dones.sum() > 0 and np.array_equal(dones, timeouts)
        do_eval(c['final_episodes'], 'final')
        checkpoint(final=True)
        progress('completed')
        atomic(output / 'completed.json', dict(source_commit=provenance['commit'],
               steps=model.num_timesteps, updates=model._n_updates,
               final_evaluation=latest_eval, completed_unix=time.time()))
    except BaseException:
        atomic(output / 'error.json', dict(traceback=traceback.format_exc(), time=time.time()))
        progress('failed')
        raise
    finally:
        if env is not None:
            env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', required=True, choices=('sac', 'td3'))
    p.add_argument('--task', required=True, choices=('door', 'hammer', 'pen', 'relocate'))
    p.add_argument('--seed', required=True, type=int, choices=(4, 5))
    p.add_argument('--profile', required=True, choices=('e256-u025', 'e256-u1', 'e64-u1'))
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    train(configuration(a.method, a.task, a.seed, a.profile, a.smoke), a.output)


if __name__ == '__main__':
    main()
