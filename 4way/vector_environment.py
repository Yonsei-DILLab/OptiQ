"""Fast batched form of the existing wall-free 2D four-goal task."""

from __future__ import annotations

import numpy as np


GOALS = np.asarray([[5., 0.], [-5., 0.], [0., 5.], [0., -5.]], np.float32)


class FourWayBatch:
    def __init__(self, num_envs: int, seed: int = 0, horizon: int = 20):
        if num_envs <= 0 or horizon <= 0:
            raise ValueError((num_envs, horizon))
        self.num_envs = num_envs
        self.horizon = horizon
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
        self.state = np.clip(self.state + actions, -7., 7.).astype(np.float32)
        delta = self.state[:, None] - GOALS[None]
        squared_distances = np.square(delta).sum(axis=-1)
        goal = squared_distances.argmin(axis=-1)
        success = squared_distances[np.arange(self.num_envs), goal] < 1.
        reward = -30. * np.square(actions).sum(axis=-1)
        reward -= squared_distances.min(axis=-1)
        reward += 10. * success
        self.steps += 1
        truncated = self.steps >= self.horizon
        return (self.state.copy(), reward.astype(np.float32), success, truncated,
                np.where(success, goal, -1).astype(np.int8))
