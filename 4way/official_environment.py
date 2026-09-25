"""Gymnasium-compatible vector port of the ICML 2019 SimplerPathFinding task.

The original Python file is preserved unmodified in ``upstream/``.  This port
keeps its map, force integration, reset distribution, task reward and 500-step
limit.  The paper's autoencoder novelty penalty is zero when no previous
policies have been installed, as in a single-policy baseline run.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


GRID_SIZE = 0.25
MAX_STEPS = 500
GOAL_CODES = (2, 3, 4, 5)  # west, east, north, south


def native_grid() -> np.ndarray:
    """Exact 27×27 wall, hint and goal assignment from the author source."""
    grid = np.zeros((27, 27), dtype=np.int8)
    mid = len(grid) // 2
    grid[0, :] = grid[-1, :] = 1
    grid[:, 0] = grid[:, -1] = 1
    grid[1:mid - 1, 1:mid - 1] = 1
    grid[mid + 2:-1, 1:mid - 1] = 1
    grid[1:mid - 1, mid + 2:-1] = 1
    grid[mid + 2:-1, mid + 2:-1] = 1
    grid[mid - 1:mid + 2, 1] = 2
    grid[mid - 1:mid + 2, -2] = 3
    grid[1, mid - 1:mid + 2] = 4
    grid[-2, mid - 1:mid + 2] = 5
    grid[mid - 1:mid + 2, 2:mid - 1] = 6
    grid[mid, mid - 1] = 6
    grid[mid - 1:mid + 2, mid + 2:-2] = 7
    grid[mid, mid + 1] = 7
    grid[2:mid - 1, mid - 1:mid + 2] = 8
    grid[mid - 1, mid] = 8
    grid[mid + 2:-2, mid - 1:mid + 2] = 9
    grid[mid + 1, mid] = 9
    return grid


class FourWayBatch:
    """N independent official-task copies with one batched NumPy step."""

    def __init__(self, num_envs: int, seed: int = 0, reward_profile: str = "symmetric"):
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        if reward_profile not in {"symmetric", "native"}:
            raise ValueError(reward_profile)
        self.num_envs = num_envs
        self.reward_profile = reward_profile
        self.rng = np.random.default_rng(seed)
        self.grid = np.repeat(native_grid()[None], num_envs, axis=0)
        self.pos = np.zeros((num_envs, 2), np.float64)
        self.vel = np.zeros_like(self.pos)
        self.steps = np.zeros(num_envs, np.int32)
        self.reset()

    def reset(self, indices: np.ndarray | None = None) -> np.ndarray:
        ids = np.arange(self.num_envs) if indices is None else np.asarray(indices, dtype=int)
        if ids.size:
            self.pos[ids] = self.rng.uniform(-0.001, 0.001, (len(ids), 2))
            self.vel[ids] = self.rng.uniform(-0.001, 0.001, (len(ids), 2))
            self.steps[ids] = 0
            # The author code does not rebuild grid_map in _reset.  Hint
            # cells cleared on visitation therefore remain cleared for this
            # environment replica across episodes.
        return self.observation()[ids]

    def observation(self) -> np.ndarray:
        return np.concatenate((self.pos, self.vel), axis=-1).astype(np.float32)

    @staticmethod
    def grid_index(pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ij = np.rint(pos / GRID_SIZE).astype(np.int64)
        return 13 - ij[:, 1], 13 + ij[:, 0]

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, np.float64)
        if actions.shape != (self.num_envs, 2):
            raise ValueError(f"expected {(self.num_envs, 2)} actions, got {actions.shape}")
        force = np.clip(actions, -1.0, 1.0) * 250.0
        wall_hit = np.zeros(self.num_envs, bool)
        moving = np.ones(self.num_envs, bool)
        for _ in range(5):
            next_pos = self.pos + .002 * self.vel
            bi, bj = self.grid_index(self.pos)
            ai, aj = self.grid_index(next_pos)
            outside = (ai < 0) | (ai > 26) | (aj < 0) | (aj > 26)
            inside = ~outside
            collision = outside.copy()
            rows = np.flatnonzero(inside)
            collision[rows] = self.grid[rows, ai[rows], aj[rows]] == 1
            collided = moving & collision
            wall_hit |= collided
            x_hit = collided & (np.abs(aj - bj) == 1)
            y_hit = collided & (np.abs(ai - bi) == 1)
            self.vel[x_hit, 0] *= -.01
            self.vel[y_hit, 1] *= -.01
            advance = moving & ~collision
            self.pos[advance] = next_pos[advance]
            self.vel[advance] = np.clip(self.vel[advance] + .002 * force[advance] / 10., -4., 4.)
            moving = advance
        self.steps += 1
        i, j = self.grid_index(self.pos)
        cells = self.grid[np.arange(self.num_envs), i, j]
        hint = (cells >= 6) & (cells <= 9)
        goal = (cells >= 2) & (cells <= 5)
        reward = np.full(self.num_envs, -1., np.float64)
        if self.reward_profile == "native":
            reward[hint] += 50. * ((cells[hint] - 5) / 4.) ** 3
            reward[goal] += 500. * ((cells[goal] - 1) / 4.) ** 3
        else:
            # The paper's original rewards privilege south over west by 64×.
            # Equal terminal reward tests behavioral diversity rather than a
            # preference encoded in the task.  Hints are disabled here.
            reward[goal] += 500.
        reward[wall_hit] -= 10.
        if self.reward_profile == "native":
            self.grid[np.flatnonzero(hint), i[hint], j[hint]] = 0
        terminated = goal
        truncated = self.steps >= MAX_STEPS
        info = {
            "goal_id": np.where(goal, cells - 2, -1),
            "hint_id": np.where(hint, cells - 6, -1),
            "wall_hit": wall_hit.copy(),
            "success": goal.copy(),
        }
        return self.observation(), reward.astype(np.float32), terminated, truncated, info


class FourWayEnv(gym.Env):
    metadata = {}

    def __init__(self, seed: int = 0, reward_profile: str = "symmetric"):
        self.batch = FourWayBatch(1, seed, reward_profile)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (4,), np.float32)
        self.action_space = gym.spaces.Box(-1., 1., (2,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.batch.rng = np.random.default_rng(seed)
        return self.batch.reset()[0], {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.batch.step(np.asarray(action)[None])
        return obs[0], float(reward[0]), bool(terminated[0]), bool(truncated[0]), {
            key: val[0].item() for key, val in info.items()
        }
