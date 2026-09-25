"""Batch collectors around the two pinned-source task adapters."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import gymnasium as gym
import numpy as np

from .nway import (HORIZON as NWAY_HORIZON, NWayBatch, STATE_LIMIT,
                   SUPPORTED_GOALS)


ROOT = Path(__file__).resolve().parents[1]


class NWayEnv(gym.Env):
    metadata = {}

    def __init__(self, goal_count: int):
        self.batch = NWayBatch(1, goal_count)
        self.goal_positions = self.batch.goal_positions
        self.observation_space = gym.spaces.Box(-STATE_LIMIT, STATE_LIMIT, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1., 1., shape=(2,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        del options
        super().reset(seed=seed)
        return self.batch.reset()[0], {}

    def step(self, action):
        obs, reward, terminated, truncated, goal = self.batch.step(np.asarray(action).reshape(1, 2))
        return obs[0], float(reward[0]), bool(terminated[0]), bool(truncated[0]), {"goal_id": int(goal[0])}


def _load(relative: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_one(task: str, seed: int = 0):
    del seed
    if task == "4way":
        module = _load("4way/environment.py", "benchmark_fourway")
        return module.FourWayEnv()
    if task == "pointmaze":
        module = _load("pointmaze/environment.py", "benchmark_pointmaze")
        return module.FourGoalPointMaze(reward_type="dense", max_episode_steps=300)
    if task.endswith("way") and task[:-3].isdigit() and int(task[:-3]) in SUPPORTED_GOALS:
        return NWayEnv(int(task[:-3]))
    raise ValueError(task)


def spaces(task: str):
    env = make_one(task)
    try:
        observation = gym.spaces.Box(-np.inf, np.inf, (4 if task == "pointmaze" else 2,), np.float32)
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
        self.horizon = 300 if task == "pointmaze" else (20 if task == "4way" else NWAY_HORIZON)
        if task == "4way":
            module = _load("4way/vector_environment.py", "benchmark_fourway_batch")
            self.batch = module.FourWayBatch(count, seed=seed)
            self.envs = None
        elif task.endswith("way") and task[:-3].isdigit() and int(task[:-3]) in SUPPORTED_GOALS:
            self.batch = NWayBatch(count, int(task[:-3]), seed=seed)
            self.envs = None
        elif task == "pointmaze":
            self.batch = None
            self.envs = [make_one(task) for _ in range(count)]
        else:
            raise ValueError(task)
        self.goal_positions = (self.batch.goal_positions.copy()
                               if isinstance(self.batch, NWayBatch)
                               else None)
        self.current = np.zeros((count, 4 if task == "pointmaze" else 2), np.float32)
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
