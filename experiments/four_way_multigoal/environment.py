"""Four-way multi-goal environment used for controlled policy validation."""

import gymnasium as gym
import numpy as np


class MultiGoalEnv(gym.Env):
    def __init__(self, horizon: int = 20, reset_std: float = 0.0):
        self.horizon = int(horizon)
        self.reset_std = float(reset_std)
        self.observation_space = gym.spaces.Box(
            low=-7.0,
            high=7.0,
            shape=(2,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )
        self.goal_positions = np.asarray(
            [[5.0, 0.0], [-5.0, 0.0], [0.0, 5.0], [0.0, -5.0]],
            dtype=np.float32,
        )
        self.np_random = np.random.default_rng()
        self.state = np.zeros(2, dtype=np.float32)
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        del options
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
            self.action_space.seed(seed)
            self.observation_space.seed(seed)
        if self.reset_std > 0.0:
            self.state = self.np_random.normal(0.0, self.reset_std, size=2).astype(np.float32)
        else:
            self.state = np.zeros(2, dtype=np.float32)
        self.steps = 0
        return self.state.copy(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        next_state = np.clip(self.state + action, -7.0, 7.0).astype(np.float32)
        goal_distances = np.linalg.norm(next_state[None] - self.goal_positions, axis=-1)
        terminated = bool(goal_distances.min() < 1.0)
        reward = -30.0 * float(np.square(action).sum())
        reward -= float(np.square(next_state[None] - self.goal_positions).sum(axis=-1).min())
        if terminated:
            reward += 10.0
        self.state = next_state
        self.steps += 1
        truncated = self.steps >= self.horizon
        return self.state.copy(), reward, terminated, truncated, {
            "goal_distance": float(goal_distances.min()),
            "success": float(terminated),
        }

    def close(self):
        pass
