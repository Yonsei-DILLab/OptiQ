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

    def step(self, actions):
        observations, rewards, terminated, truncated, goals = [], [], [], [], []
        for env, action in zip(self.envs, actions):
            obs, reward, done, timeout, info = env.step(action)
            observations.append(obs)
            rewards.append(reward)
            terminated.append(done)
            truncated.append(timeout)
            goals.append(info['goal_id'])
        self.current = np.asarray(observations, np.float32)
        return (self.current.copy(), np.asarray(rewards, np.float32),
                np.asarray(terminated), np.asarray(truncated), np.asarray(goals))

    def close(self):
        for env in self.envs:
            env.close()
