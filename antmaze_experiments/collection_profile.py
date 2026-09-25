"""Opt-in collection cadence; preserve optimizer updates per transition."""
from .settings import NUM_ENVS, UPDATES, WARMUP

PROFILES = ('env32-update1',)


def get_profile(name=None):
    if name is None:
        return dict(num_envs=NUM_ENVS, updates_per_vector_step=UPDATES)
    if name not in PROFILES:
        raise ValueError(name)
    return dict(num_envs=32, updates_per_vector_step=1)


def expected_updates(total_steps, name=None):
    profile = get_profile(name)
    count = total_steps - WARMUP
    if count < 0 or count % profile['num_envs']:
        raise ValueError('Post-warmup transition count must match the collection batch')
    assert profile['updates_per_vector_step'] / profile['num_envs'] == 1/32
    return count // profile['num_envs'] * profile['updates_per_vector_step']


def aligned_eval_step(index, interval):
    """Match the existing256-env control's actual evaluation transition counts."""
    assert index >= 1 and interval >= NUM_ENVS
    return ((index * interval + NUM_ENVS - 1) // NUM_ENVS) * NUM_ENVS
