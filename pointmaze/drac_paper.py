"""Thin adapter for the original DrAC multi-goal PointMaze environments."""

from __future__ import annotations

from pathlib import Path
import sys

import gymnasium as gym
import numpy as np


UPSTREAM = Path(__file__).resolve().parent / "drac_upstream"
MAP_NAMES = ("simple", "medium", "hard")
HORIZONS = {"simple": 150, "medium": 300, "hard": 600}
GOAL_COUNTS = {"simple": 4, "medium": 4, "hard": 8}


def upstream_modules():
    # The original source uses `from envs.mgmaze...` absolute imports.
    # Put its untouched package on sys.path rather than rewriting those files.
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    from envs.mgmaze.maps import get_map
    from envs.mgmaze.point_maze import MultiGoalPointMaze
    return get_map, MultiGoalPointMaze


class PaperPointMaze(gym.Env):
    """Expose the original physics/reward with zero-based goal IDs for reports."""

    metadata = {}

    def __init__(self, maze: str, obstacle: bool = False):
        if maze not in MAP_NAMES:
            raise ValueError(maze)
        get_map, original = upstream_modules()
        self.maze_name = maze
        self.maze_map = get_map(maze)
        self.obstacle = bool(obstacle)
        self.native = original(maze_map=maze, reward_type="sparse",
                               maze_eval_mode=self.obstacle)
        self.goal_positions = np.asarray(self.native.maze.unique_goal_locations, np.float32)
        self.horizon = HORIZONS[maze]
        if len(self.goal_positions) != GOAL_COUNTS[maze] or self.native.max_steps != self.horizon:
            raise ValueError("upstream goal count or horizon changed")
        self.observation_space = self.native.observation_space
        self.action_space = self.native.action_space

    def reset(self, *, seed=None, options=None):
        return self.native.reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.native.step(action)
        info = dict(info)
        target = int(info["target"])
        info["goal_id"] = target - 1 if terminated and target > 0 else -1
        return obs, float(reward), bool(terminated), bool(truncated), info

    def close(self):
        self.native.close()
