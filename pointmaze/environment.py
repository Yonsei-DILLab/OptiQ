"""Official Farama PointMaze dynamics with a disclosed four-goal task wrapper.

The installed ``gymnasium_robotics`` package creates the MuJoCo point and
walls.  The custom cross map, fixed central reset cell and any-goal reward are
benchmark choices; they are not a stock PointMaze environment ID.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


MAZE_MAP = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, "g", 1, 1, 1],
    [1, 1, 1, 0, 1, 1, 1],
    [1, "g", 0, "r", 0, "g", 1],
    [1, 1, 1, 0, 1, 1, 1],
    [1, 1, 1, "g", 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
]
GOAL_CELLS = ((3, 5), (3, 1), (1, 3), (5, 3))  # east, west, north, south


class FourGoalPointMaze(gym.Env):
    """Fixed reset cell, hidden sampled target, reward for the nearest arm."""

    metadata = {}

    def __init__(self, reward_type: str = "dense", max_episode_steps: int = 300):
        from gymnasium_robotics.envs.maze.point_maze import PointMazeEnv

        if reward_type not in {"dense", "sparse"}:
            raise ValueError(reward_type)
        self.reward_type = reward_type
        self.max_episode_steps = int(max_episode_steps)
        # MuJoCo dynamics are the official package implementation.  Its own
        # randomly sampled desired_goal is hidden from the policy and ignored
        # by this explicitly any-goal reward wrapper.
        self.native = PointMazeEnv(
            maze_map=MAZE_MAP, reward_type=reward_type,
            continuing_task=False, reset_target=False,
        )
        self.goal_positions = np.asarray(
            [self.native.maze.cell_rowcol_to_xy(np.asarray(cell)) for cell in GOAL_CELLS],
            dtype=np.float64,
        )
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (4,), np.float64)
        self.action_space = self.native.action_space
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs, _native_info = self.native.reset(seed=seed, options={"reset_cell": np.asarray((3, 3))})
        self.steps = 0
        # Never expose the native single-goal task's sampled target.
        return obs["observation"].copy(), {}

    def step(self, action):
        obs, _native_reward, _native_terminated, _native_truncated, _native_info = self.native.step(action)
        xy = obs["achieved_goal"]
        distances = np.linalg.norm(self.goal_positions - xy[None], axis=1)
        goal_id = int(np.argmin(distances))
        success = bool(distances[goal_id] <= .45)
        # Same dense exp(-distance) and sparse 0/1 formula as PointMaze,
        # extended from its one sampled goal to the nearest of four goals.
        reward = float(np.exp(-distances[goal_id])) if self.reward_type == "dense" else float(success)
        self.steps += 1
        truncated = self.steps >= self.max_episode_steps
        return obs["observation"].copy(), reward, success, truncated, {
            "success": success, "goal_id": goal_id if success else -1,
            "nearest_goal_id": goal_id,
            "nearest_goal_distance": float(distances[goal_id]),
        }

    def close(self):
        self.native.close()
