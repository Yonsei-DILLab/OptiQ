"""Batch collectors around the two pinned-source task adapters."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import gymnasium as gym
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load(relative: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_one(task: str, seed: int = 0):
    if task == "4way":
        module = _load("4way/environment.py", "benchmark_fourway")
        return module.FourWayEnv()
    if task == "pointmaze":
        module = _load("pointmaze/environment.py", "benchmark_pointmaze")
        return module.FourGoalPointMaze(reward_type="dense", max_episode_steps=300)
    raise ValueError(task)


def spaces(task: str):
    env = make_one(task)
    try:
        observation = gym.spaces.Box(-np.inf, np.inf, (2 if task == "4way" else 4,), np.float32)
        action = gym.spaces.Box(-1., 1., (2,), np.float32)
        assert env.observation_space.shape == observation.shape
        assert env.action_space.shape == action.shape
        return observation, action
    finally:
        env.close()


class TaskBatch:
    """A batch step returns final observations; resets are explicit."""

    def __init__(self, task: str, count: int, seed: int):
        if count <= 0:
            raise ValueError(count)
        self.task, self.count, self.seed = task, count, seed
        self.horizon = 20 if task == "4way" else 300
        if task == "4way":
            module = _load("4way/vector_environment.py", "benchmark_fourway_batch")
            self.batch = module.FourWayBatch(count, seed=seed)
            self.envs = None
        elif task == "pointmaze":
            self.batch = None
            self.envs = [make_one(task) for _ in range(count)]
        else:
            raise ValueError(task)
        self.current = np.zeros((count, 2 if task == "4way" else 4), np.float32)
        self.reset(np.arange(count))

    def reset(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        if not len(indices):
            return self.current
        if self.batch is not None:
            self.current[indices] = self.batch.reset(indices)
        else:
            for i in indices:
                self.current[i], _ = self.envs[i].reset(seed=self.seed + 100003 * int(i))
                # Subsequent resets must advance the same environment RNG.
                self.seed += 1
        return self.current

    def step(self, actions, active=None):
        actions = np.asarray(actions, np.float32)
        if actions.shape != (self.count, 2):
            raise ValueError(actions.shape)
        active = np.ones(self.count, bool) if active is None else np.asarray(active, bool)
        if active.shape != (self.count,):
            raise ValueError(active.shape)
        if self.batch is not None:
            obs, reward, terminated, truncated, goal = self.batch.step(
                np.where(active[:, None], actions, 0.))
        else:
            obs = self.current.copy()
            reward = np.zeros(self.count, np.float32)
            terminated = np.zeros(self.count, bool)
            truncated = np.zeros(self.count, bool)
            goal = np.full(self.count, -1, np.int8)
            for i in np.flatnonzero(active):
                o, r, d, tr, inf = self.envs[i].step(actions[i])
                obs[i] = o
                reward[i] = r
                terminated[i] = d
                truncated[i] = tr
                goal[i] = inf["goal_id"]
        self.current = obs.copy()
        return obs, reward, terminated, truncated, goal

    def close(self):
        if self.envs is not None:
            for env in self.envs:
                env.close()
