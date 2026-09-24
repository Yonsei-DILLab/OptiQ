"""Official sparse/NovelD baseline profile, with native256-env accounting."""
CAMPAIGN = 'antmaze-upstream-sparse256-nativebudget-s0-20260923'
WANDB_ENTITY = 'OptiQ'
WANDB_PROJECT = 'antmaze'
BUDGETS = dict(v1=3000000, v2=3000000, v3=4000000, v4=5000000)
REWARD = 'unchanged upstream sparse reward:0 except goal bonus10 or20'
DENSE_REWARD = 'negative Euclidean distance from next xy to nearest goal; no sparse bonus'
# Dense rewards are negative. Retain native 51 atoms and upper endpoint.
DIPO_DENSE_V_MIN = -6000.0
NUM_ENVS = 256
EVAL_NUM_ENVS = 20
UPDATES = 8
WARMUP = 32 * NUM_ENVS
PREFLIGHT_STEPS = WARMUP + NUM_ENVS


def total_budget(task):
    # baselines_main excludes warmup from global_steps and stops at >max_step.
    return WARMUP + (BUDGETS[task] // NUM_ENVS + 1) * NUM_ENVS


def expected_updates(total_steps):
    assert (total_steps-WARMUP) % NUM_ENVS == 0
    return (total_steps-WARMUP) // NUM_ENVS * UPDATES


from .progress_reward import PROFILES
REWARD_PROFILES = ('sparse', 'dense') + PROFILES

def reward_description(profile):
    if profile.startswith('progress100_'):
        from .progress_reward import SCALED_PROFILES
        if profile not in SCALED_PROFILES: raise ValueError(profile)
        return reward_description(profile.replace('progress100_', 'progress_')).replace('distance decrease -0.01', 'distance decrease *100 -1')
    return {
        'sparse': REWARD, 'dense': DENSE_REWARD,
        'progress_euclidean': 'nearest-goal Euclidean distance decrease -0.01 + upstream goal bonus10/20',
        'progress_euclidean_no_bonus': 'nearest-goal Euclidean distance decrease -0.01; no success bonus',
        'progress_geodesic_no_bonus': 'nearest-goal XY geodesic distance decrease -0.01; no success bonus',
        'progress_geodesic': 'nearest-goal XY geodesic distance decrease -0.01 + upstream goal bonus10/20',
    }[profile]
