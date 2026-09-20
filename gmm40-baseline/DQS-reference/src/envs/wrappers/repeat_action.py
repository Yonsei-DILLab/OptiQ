# Code adapted from https://github.com/ikostrikov/jaxrl

import gymnasium as gym
import numpy as np

from src.envs.wrappers.common import TimeStep


class RepeatAction(gym.Wrapper):

    def __init__(self, env, action_repeat=4):
        super().__init__(env)
        self._action_repeat = action_repeat

    def step(self, action: np.ndarray) -> TimeStep:
        total_reward = 0.0
        terminated = False
        truncated = False
        combined_info = {}

        obs = None
        for _ in range(self._action_repeat):
            obs, reward, term, trunc, info = self.env.step(action)
            total_reward += reward
            combined_info.update(info)
            terminated = terminated or term
            truncated = truncated or trunc
            if term or trunc:
                break

        return obs, total_reward, terminated, truncated, combined_info