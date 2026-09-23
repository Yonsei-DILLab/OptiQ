"""Official sparse/NovelD baseline profile, with native256-env accounting."""
CAMPAIGN = 'antmaze-upstream-sparse256-nativebudget-s0-20260923'
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
