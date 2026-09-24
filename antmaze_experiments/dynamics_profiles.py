"""Explicit, opt-in profiles for the 2026-09-25 critic dynamics screen."""
from copy import deepcopy

from .progress_reward import EUCLIDEAN_NO_COST_PROFILE, EUCLIDEAN_SCALE20_PROFILE

_BASE = dict(reward_profile=EUCLIDEAN_NO_COST_PROFILE, reward_multiplier=1.,
             temperature=1., tau=.005, critic_lr=5e-4, policy_delay=1)
PROFILES = {
    'control': {},
    'scale02': dict(reward_profile=EUCLIDEAN_SCALE20_PROFILE,
                    reward_multiplier=.2, temperature=.2),
    'ema01': dict(tau=.01),
    'criticlr2': dict(critic_lr=1e-3),
    'actordelay2': dict(policy_delay=2),
    'scale02ema01': dict(reward_profile=EUCLIDEAN_SCALE20_PROFILE,
                         reward_multiplier=.2, temperature=.2, tau=.01),
}


def get_profile(name):
    if name not in PROFILES:
        raise ValueError(name)
    return deepcopy(dict(_BASE, **PROFILES[name]))


def expected_actor_updates(critic_updates, profile):
    return int(critic_updates) // get_profile(profile)['policy_delay']
