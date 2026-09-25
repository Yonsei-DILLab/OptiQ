"""Wall-free symmetric N-goal point task used for mode-scaling studies.

This keeps the four-way action and reward equations. The new 8/12/16/32-way
family uses one shared ring radius and terminal radius, so only goal count
changes within that family. The terminal discs remain disjoint at N=32.
"""

from __future__ import annotations

import numpy as np


SUPPORTED_GOALS = (8, 12, 16, 32)
GOAL_RADIUS = 6.0
SUCCESS_RADIUS = 0.5
STATE_LIMIT = 8.0
HORIZON = 24


def goal_positions(count: int) -> np.ndarray:
    if count not in SUPPORTED_GOALS:
        raise ValueError(f"unsupported goal count: {count}")
    angles = 2.0 * np.pi * np.arange(count, dtype=np.float64) / count
    return (GOAL_RADIUS * np.stack((np.cos(angles), np.sin(angles)), axis=-1)).astype(np.float32)


class NWayBatch:
    def __init__(self, num_envs: int, goal_count: int, seed: int = 0):
        if num_envs <= 0:
            raise ValueError(num_envs)
        self.num_envs = num_envs
        self.goal_count = goal_count
        self.goal_positions = goal_positions(goal_count)
        self.horizon = HORIZON
        self.state = np.zeros((num_envs, 2), np.float32)
        self.steps = np.zeros(num_envs, np.int32)
        self.rng = np.random.default_rng(seed)

    def reset(self, indices=None):
        indices = np.arange(self.num_envs) if indices is None else np.asarray(indices, int)
        self.state[indices] = 0.
        self.steps[indices] = 0
        return self.state[indices].copy()

    def step(self, actions):
        actions = np.asarray(actions, np.float32)
        if actions.shape != (self.num_envs, 2):
            raise ValueError(actions.shape)
        actions = np.clip(actions, -1., 1.)
        self.state = np.clip(self.state + actions, -STATE_LIMIT, STATE_LIMIT).astype(np.float32)
        squared = np.square(self.state[:, None, :] - self.goal_positions[None]).sum(axis=-1)
        nearest = squared.argmin(axis=-1)
        success = squared[np.arange(self.num_envs), nearest] < SUCCESS_RADIUS ** 2
        reward = (-30. * np.square(actions).sum(axis=-1) - squared.min(axis=-1)
                  + 10. * success)
        self.steps += 1
        truncated = self.steps >= self.horizon
        return (self.state.copy(), reward.astype(np.float32), success, truncated,
                np.where(success, nearest, -1).astype(np.int8))
