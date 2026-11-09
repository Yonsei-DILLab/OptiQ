"""Isolated iBOLT/DIPO PointMaze entry point. No full run starts during setup."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import time

from .guards import ROOT, assert_output_boundary, assert_work_branch, code_hashes
from .protocol import configuration


def save_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def runtime(method, device):
    names = ('jax', 'jaxlib', 'flax', 'optax', 'numpy', 'torch', 'gymnasium', 'mujoco')
    versions = {name: importlib.metadata.version(name) for name in names}
    if method == 'ibolt':
        import jax
        backend, devices = jax.default_backend(), [d.device_kind for d in jax.devices()]
    else:
        import torch
        backend = 'gpu' if device == 'cuda' else 'cpu'
        devices = [torch.cuda.get_device_name()] if device == 'cuda' else ['CPU']
    return dict(backend=backend, devices=devices, packages=versions,
        revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=('ibolt', 'dipo'), required=True)
    parser.add_argument('--maze', choices=('simple', 'medium', 'hard'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--smoke', action='store_true', help='8 transitions / 2 updates, five final episodes per setting')
    for name in ('steps', 'num-envs', 'batch-size', 'updates-per-collect', 'warmup',
                 'eval-every', 'eval-episodes', 'final-episodes', 'replay-capacity'):
        parser.add_argument('--' + name, type=int)
    parser.add_argument('--temperature', type=float)
    parser.add_argument('--no-replay', action='store_true')
    parser.add_argument('--eager-metrics', action='store_true',
                        help='iBOLT diagnostic: use reference per-update CPU metric reads')
    args = parser.parse_args(argv)
    branch = assert_work_branch()
    assert_output_boundary(args.output)
    if args.method == 'dipo' and args.temperature is not None:
        parser.error('Temperature is an iBOLT setting, not a DIPO hyperparameter')
    if args.method == 'dipo' and args.eager_metrics:
        parser.error('--eager-metrics is an iBOLT diagnostic option')
    overrides = {key: getattr(args, key) for key in ('steps', 'num_envs', 'batch_size',
        'updates_per_collect', 'warmup', 'eval_every', 'eval_episodes', 'final_episodes', 'replay_capacity')}
    if args.method == 'ibolt':
        overrides['temperature'] = args.temperature
    config = configuration(args.method, args.maze, args.seed, args.smoke, **overrides)
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    os.environ.setdefault('MUJOCO_GL', 'egl')
    if args.device == 'cpu':
        os.environ['JAX_PLATFORMS'] = 'cpu'
    import numpy as np
    from .envs.batch import TaskBatch
    from .evaluation import evaluate
    if args.method == 'dipo':
        from .agents.dipo import Agent
    else:
        from .agents.ibolt import Agent
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in ('evaluations', 'checkpoints'):
        (output / name).mkdir()
    config.update(branch=branch, device=args.device, save_replay=not args.no_replay,
                  timeout_bootstrap=False, environment='original DrAC sparse +100',
                  collector='synchronous single-process TaskBatch; batched policy inference')
    save_json(output / 'config.json', config)
    save_json(output / 'source-hashes.json', code_hashes())
    np.random.seed(args.seed)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    agent = environment = None
    step = 0
    started = time.monotonic()
    try:
        agent_options = dict(defer_metrics=not args.eager_metrics) if args.method == 'ibolt' else {}
        agent = Agent(config, output, device=args.device, **agent_options)
        config.update(agent=agent.config, runtime=runtime(args.method, args.device),
                      evaluation_mode=agent.evaluation_mode)
        save_json(output / 'config.json', config)
        environment = TaskBatch(args.maze, config['num_envs'], args.seed)
        for step in range(config['num_envs'], config['steps'] + 1, config['num_envs']):
            obs = environment.current.copy()
            actions = (rng.uniform(-1., 1., (config['num_envs'], 2)).astype(np.float32)
                       if step <= config['warmup'] else agent.act(obs, mode='train'))
            if not np.isfinite(actions).all():
                raise FloatingPointError('Nonfinite policy actions')
            actions = np.clip(actions, -1., 1.).astype(np.float32)
            next_obs, rewards, terminated, truncated, _ = environment.step(actions)
            if not np.isfinite(next_obs).all() or not np.isfinite(rewards).all():
                raise FloatingPointError('Nonfinite environment transitions')
            done = terminated | truncated
            agent.observe(obs, actions, rewards, next_obs, done)
            environment.reset(np.flatnonzero(done))
            if step > config['warmup']:
                if args.method == 'ibolt':
                    agent.update_many(step, config['updates_per_collect'])
                else:
                    for _ in range(config['updates_per_collect']):
                        agent.update(step)
            progress = dict(steps=step, updates=agent.updates, status='running')
            save_json(output / 'progress.json', progress)
            evaluation_due = step % config['eval_every'] < config['num_envs'] or step == config['steps']
            if (not evaluation_due and step > config['warmup'] and
                    agent.updates % 1000 < config['updates_per_collect']):
                agent.flush_logs(step)
            if evaluation_due:
                episodes = config['final_episodes'] if step == config['steps'] else config['eval_episodes']
                mode = agent.evaluation_mode
                record = dict(steps=step, updates=agent.updates, method=args.method,
                              primary_evaluation_mode=mode)
                # Restore all training RNGs, including any environment global RNG use.
                numpy_state, python_state = np.random.get_state(), random.getstate()
                try:
                    for obstacle, offset in ((False, 17000), (True, 27000)):
                        key = ('obstacle_' if obstacle else '') + mode
                        record[key] = evaluate(agent, 'pm_' + args.maze,
                            args.seed + offset + step, episodes, mode,
                            output / 'evaluations' / f'{step:09d}_{key}.npz', obstacle)
                finally:
                    np.random.set_state(numpy_state)
                    random.setstate(python_state)
                agent.assert_finite()
                extension = '.pt' if args.method == 'dipo' else '.msgpack'
                agent.save(output / 'checkpoints' / (f'{step:09d}' + extension))
                agent.flush_logs(step)
                save_json(output / 'evaluations' / f'{step:09d}_summary.json', record)
                with (output / 'evaluations.jsonl').open('a') as stream:
                    stream.write(json.dumps(record, allow_nan=False) + '\n')
                print(json.dumps(record, allow_nan=False), flush=True)
        if agent.updates != config['expected_updates']:
            raise RuntimeError(f'Expected {config["expected_updates"]} updates; got {agent.updates}')
        if not args.no_replay:
            agent.save_replay(output / 'checkpoints' / 'final-replay.npz')
        result = dict(steps=step, updates=agent.updates, completed=True, method=args.method,
                      smoke=args.smoke, backend=config['runtime']['backend'],
                      elapsed_seconds=time.monotonic() - started)
        save_json(output / 'completed.json', result)
        save_json(output / 'progress.json', dict(result, status='complete'))
        print('Completed: ' + str(output), flush=True)
    except BaseException as error:
        save_json(output / 'failure.json', dict(steps=step,
            updates=agent.updates if agent is not None else 0, error=repr(error)))
        raise
    finally:
        if environment is not None:
            environment.close()
        if agent is not None:
            agent.close()


if __name__ == '__main__':
    main()
