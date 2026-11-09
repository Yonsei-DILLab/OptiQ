"""Explicit historical profiles; collection budgets are method-specific."""
import json
from pathlib import Path

PROFILES = Path(__file__).resolve().parent / 'profiles'


def configuration(method, maze, seed=0, smoke=False, **overrides):
    if method not in ('ibolt', 'dipo') or maze not in ('simple', 'medium', 'hard'):
        raise ValueError('Unknown method or maze')
    if not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError('Seed must be a nonnegative uint32 integer')
    config = json.loads((PROFILES / f'{method}.json').read_text())
    config.update(maze=maze, seed=seed, smoke=smoke)
    if smoke:
        config.update(steps=8, num_envs=2, batch_size=4, updates_per_collect=1,
                      warmup=4, eval_every=8, eval_episodes=5, final_episodes=5,
                      replay_capacity=32)
    for key, value in overrides.items():
        if value is not None:
            if key not in config:
                raise ValueError(f'Unsupported profile override: {key}')
            config[key] = value
    fields = ('steps', 'num_envs', 'batch_size', 'updates_per_collect', 'warmup',
              'eval_every', 'eval_episodes', 'final_episodes', 'replay_capacity')
    if any(not isinstance(config[k], int) or config[k] <= 0 for k in fields):
        raise ValueError('Budgets and counts must be positive integers')
    if config['steps'] <= config['warmup']:
        raise ValueError('Training must extend beyond warmup')
    if any(config[k] % config['num_envs'] for k in ('steps', 'warmup')):
        raise ValueError('Steps and warmup must contain full collector batches')
    if config['warmup'] < config['batch_size']:
        raise ValueError('Warmup must fill a learner batch')
    if config['replay_capacity'] < max(config['warmup'], config['batch_size'], config['num_envs']):
        raise ValueError('Replay capacity is below the collection/batch requirement')
    if config['eval_episodes'] % 5 or config['final_episodes'] % 5:
        raise ValueError('SR5 evaluation requires episode counts divisible by five')
    if method == 'ibolt':
        import math
        if not math.isfinite(config['temperature']) or config['temperature'] <= 0:
            raise ValueError('Temperature must be positive and finite')
    config['expected_updates'] = ((config['steps'] - config['warmup']) //
                                  config['num_envs'] * config['updates_per_collect'])
    config['utd'] = config['updates_per_collect'] / config['num_envs']
    return config
