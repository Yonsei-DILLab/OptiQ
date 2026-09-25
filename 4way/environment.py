"""Symmetric two-dimensional task with four terminal goal modes.

This preserves the environment used by the earlier four-way proposal study;
only the goal id is exposed in ``info`` for exact route accounting.
"""

import gymnasium as gym
import numpy as np


class FourWayEnv(gym.Env):
    metadata = {}

    def __init__(self, horizon=20):
        self.horizon = int(horizon)
        self.observation_space = gym.spaces.Box(-7.0, 7.0, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.goal_positions = np.asarray(
            [[5.0, 0.0], [-5.0, 0.0], [0.0, 5.0], [0.0, -5.0]],
            dtype=np.float32,
        )
        self.goal_names = ("east", "west", "north", "south")
        self.state = np.zeros(2, dtype=np.float32)
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        del options
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
        self.state = np.zeros(2, dtype=np.float32)
        self.steps = 0
        return self.state.copy(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.state = np.clip(self.state + action, -7.0, 7.0).astype(np.float32)
        distances = np.linalg.norm(self.state[None] - self.goal_positions, axis=-1)
        goal_id = int(np.argmin(distances))
        terminated = bool(distances[goal_id] < 1.0)
        reward = -30.0 * float(np.square(action).sum())
        reward -= float(np.square(self.state[None] - self.goal_positions).sum(axis=-1).min())
        if terminated:
            reward += 10.0
        self.steps += 1
        truncated = self.steps >= self.horizon
        return self.state.copy(), reward, terminated, truncated, {
            "goal_id": goal_id if terminated else -1,
            "nearest_goal_id": goal_id,
            "goal_distance": float(distances[goal_id]),
            "success": float(terminated),
        }
