"""Smoke and task-invariant checks against installed official PointMaze."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np


SPEC = importlib.util.spec_from_file_location("four_goal_pointmaze", Path(__file__).with_name("environment.py"))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PointMazeTests(unittest.TestCase):
    def test_four_symmetric_goals_and_hidden_native_target(self):
        env = MODULE.FourGoalPointMaze(reward_type="dense")
        try:
            goal_positions = env.goal_positions
            self.assertEqual(goal_positions.shape, (4, 2))
            np.testing.assert_allclose(np.linalg.norm(goal_positions, axis=1), 2.)
            obs, info = env.reset(seed=123)
            self.assertEqual(obs.shape, (4,))
            self.assertNotIn("native_desired_goal", info)
            for _ in range(16):
                obs, reward, done, truncated, info = env.step(np.zeros(2))
                self.assertTrue(np.isfinite(obs).all())
                self.assertTrue(0 <= reward <= 1)
                self.assertNotIn("is_success", info)
                self.assertIn("goal_id", info)
                if done or truncated:
                    break
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
