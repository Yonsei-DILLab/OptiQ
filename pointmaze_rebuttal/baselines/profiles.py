"""Explicit baseline optimization budgets; native model settings stay separate."""
METHODS = ('sac', 'sql', 'td3', 'dipo', 'mfpo', 'meow')


def configuration(method, maze, seed=0, smoke=False, profile='matched', **overrides):
    if method not in METHODS or maze not in ('simple', 'medium', 'hard'):
        raise ValueError('Unknown baseline or maze')
    if profile not in ('matched', 'native') or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError('Invalid profile or seed')
    native_dipo = profile == 'native' and method == 'dipo'
    config = dict(method=method, maze=maze, seed=seed, smoke=smoke,
        profile='baseline-' + profile, steps=1001472 if native_dipo else 1000192,
        num_envs=2048 if native_dipo else 256,
        batch_size=256 if profile == 'matched' else 4096,
        updates_per_collect=256 if profile == 'matched' else 32 if native_dipo else 16,
        warmup=8192, eval_every=200000, eval_episodes=200, final_episodes=500,
        replay_capacity=1000000,
        native_settings=dict(sql_temperature=1., sql_particles=16,
                             meow_alpha=.2, mfpo_target_entropy_coeff=-.5))
    if smoke:
        config.update(steps=8, num_envs=2, batch_size=4, updates_per_collect=1,
                      warmup=4, eval_every=8, eval_episodes=5, final_episodes=5,
                      replay_capacity=32)
    for key, value in overrides.items():
        if value is not None:
            if key not in config:
                raise ValueError('Unknown override: ' + key)
            config[key] = value
    fields = ('steps', 'num_envs', 'batch_size', 'updates_per_collect', 'warmup',
              'eval_every', 'eval_episodes', 'final_episodes', 'replay_capacity')
    if any(not isinstance(config[k], int) or config[k] <= 0 for k in fields):
        raise ValueError('Counts must be positive integers')
    if (config['steps'] <= config['warmup'] or config['warmup'] < config['batch_size']
        or any(config[k] % config['num_envs'] for k in ('steps', 'warmup'))
        or config['replay_capacity'] < max(config['warmup'], config['batch_size'], config['num_envs'])
        or any(config[k] % 5 for k in ('eval_episodes', 'final_episodes'))):
        raise ValueError('Invalid collector/replay/evaluation budget')
    config['expected_updates'] = (config['steps']-config['warmup']) // config['num_envs'] * config['updates_per_collect']
    config['utd'] = config['updates_per_collect'] / config['num_envs']
    return config
