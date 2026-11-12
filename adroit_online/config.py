"""Frozen scientific configuration, independent of the execution host."""
TASKS = {t: 'AdroitHand' + t.title() + 'Sparse-v2'
         for t in ('door', 'hammer', 'pen', 'relocate')}
ROBOTICS_COMMIT = 'b8da02a89191daf8f004e7f7510ff655f7a90100'
REFERENCE_COMMIT = 'ebdc46e59db1642902e0331c1e3aa662556c6553'
PROFILES = {'e256-u025': (256, .25), 'e256-u1': (256, 1.), 'e64-u1': (64, 1.)}
IBOLT_TEMPERATURES = (.01, .05, .10, .25)


def configuration(method, task, seed, profile, smoke=False, temperature=None):
    assert method in ('sac', 'td3', 'ibolt') and task in TASKS and seed in (4, 5)
    assert temperature is None or method == 'ibolt', 'Actor temperature is an iBOLT-only setting'
    n, utd = PROFILES[profile]
    c = dict(method=method, task=task, seed=seed, profile=profile,
             env_id=TASKS[task], num_envs=n, utd=utd, workers=4,
             steps=3_000_064, requested_steps=3_000_000, warmup=10_240,
             batch_size=256, replay_capacity=1_000_000, learning_rate=3e-4,
             gamma=.99, tau=.005, hidden=[256, 256], critic_layer_norm=False,
             eval_every=50_176, eval_episodes=100, final_episodes=1000,
             checkpoint_every=250_880, horizon=200, demonstration_ratio=0.,
             behavior_cloning=False, her=False, normalize_observations=False,
             normalize_rewards=False, reward_type='sparse',
             entropy='auto', td3_exploration_std=.1, policy_delay=2,
             target_policy_noise=.2, target_noise_clip=.5,
             robotics_commit=ROBOTICS_COMMIT, reference_commit=REFERENCE_COMMIT,
             smoke=smoke)
    if smoke:
        c.update(steps=53_248, warmup=51_200, replay_capacity=65_536, eval_every=53_248,
                 eval_episodes=20, final_episodes=20, checkpoint_every=53_248)
    c['gradient_steps'] = int(n * utd)
    c['expected_updates'] = int((c['steps'] - c['warmup']) * utd)
    if method == 'ibolt':
        assert profile == 'e256-u025', 'Only the current baseline profile is authorized for iBOLT'
        temperature = .25 if temperature is None else float(temperature)
        assert temperature in IBOLT_TEMPERATURES
        for name in ('entropy', 'td3_exploration_std', 'policy_delay', 'target_policy_noise', 'target_noise_clip'):
            c.pop(name)
        c.update(temperature=temperature, num_policy_samples=64, proposals_per_policy_sample=1,
                 activation='gelu', log_std_bounds=[-5., -1.], initial_log_std=-1.,
                 density_correction_beta=1., use_min_candidate_q=False,
                 actor_policy_delay=1, dacer_enabled=True, dacer_noise_scale=.1,
                 dacer_target_entropy_per_dim=-.9, dacer_initial_alpha=.27,
                 dacer_alpha_lr=.03, dacer_interval_updates=10000,
                 primary_evaluation='stochastic_z')
    assert all(c[k] % n == 0 for k in ('steps', 'warmup', 'eval_every'))
    return c


def campaign():
    # Priority only: no task-completion barriers. All 48 runs are eligible.
    return [configuration(m, t, s, p)
            for p in PROFILES for s in (4, 5) for m in ('sac', 'td3') for t in TASKS]


def run_id(c):
    return (f"{c['task']}-{c['method']}-s{c['seed']}-{c['profile']}"
            + (f"-t{round(c['temperature'] * 100):03d}" if c['method'] == 'ibolt' else '')
            + ('-smoke' if c['smoke'] else ''))
