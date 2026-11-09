"""Synchronous collectors with explicit final observations and resets."""
import numpy as np
from .drac_paper import PaperPointMaze


class TaskBatch:
    def __init__(self, maze, count, seed, obstacle=False):
        self.count, self.seed = count, seed
        self.envs = [PaperPointMaze(maze, obstacle) for _ in range(count)]
        self.current = np.zeros((count, 4), np.float32)
        self.reset(np.arange(count))

    def reset(self, indices):
        for i in indices:
            self.current[i], _ = self.envs[i].reset(seed=self.seed + 100003 * int(i))
            self.seed += 1
        return self.current

    def step(self, actions, active=None):
        actions = np.asarray(actions, np.float32)
        if actions.shape != (self.count, 2):
            raise ValueError(actions.shape)
        active = np.ones(self.count, bool) if active is None else np.asarray(active, bool)
        if active.shape != (self.count,):
            raise ValueError(active.shape)
        observations = self.current.copy()
        rewards = np.zeros(self.count, np.float32)
        terminated, truncated = np.zeros(self.count, bool), np.zeros(self.count, bool)
        goals = np.full(self.count, -1, np.int8)
        for i in np.flatnonzero(active):
            obs, reward, done, timeout, info = self.envs[i].step(actions[i])
            observations[i], rewards[i] = obs, reward
            terminated[i], truncated[i], goals[i] = done, timeout, info['goal_id']
        self.current = observations.copy()
        return observations, rewards, terminated, truncated, goals

    def close(self):
        for env in self.envs:
            env.close()
