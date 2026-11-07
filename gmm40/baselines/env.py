"""A one-step bandit whose maximum-entropy optimum is the bounded GMM40."""
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from gmm40.target import Target, SCALE


class GMM40Bandit(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, target=None):
        super().__init__()
        self.target = target if target is not None else Target()
        self.observation_space = spaces.Box(0., 0., shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(-1., 1., shape=(2,), dtype=np.float32)
        self.reward_evaluations = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        action = np.asarray(action)
        if action.shape != (2,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1.):
            raise ValueError('GMM40 bandit actions must be finite and inside [-1,1]^2')
        reward = float(self.target.log_prob((SCALE*action)[None])[0])
        self.reward_evaluations += 1
        return np.zeros(1, dtype=np.float32), reward, True, False, {}
