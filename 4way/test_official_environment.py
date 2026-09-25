"""Task-level checks for the official-source vector physics and goals."""

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np


SPEC = importlib.util.spec_from_file_location("official_fourway", Path(__file__).with_name("official_environment.py"))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FourWayTests(unittest.TestCase):
    def test_vector_port_matches_author_dynamics_and_native_reward(self):
        # The author's old Gym file imports Keras solely for a cross-policy
        # novelty option. Stub that unused import, leaving the source untouched.
        keras = types.ModuleType("keras")
        models = types.ModuleType("keras.models")
        models.load_model = lambda *args, **kwargs: None
        with patch.dict(sys.modules, {"keras": keras, "keras.models": models}):
            original_path = Path(__file__).parent / "upstream/simpler_path_finding.py"
            spec = importlib.util.spec_from_file_location("fourway_author", original_path)
            original_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(original_module)
        reference = original_module.SimplerPathFinding()
        batch = MODULE.FourWayBatch(1, seed=8, reward_profile="native")
        np.testing.assert_array_equal(reference.grid_map, batch.grid[0])
        rng = np.random.default_rng(83)
        starts = ((0., 0.), (-2.5, 0.), (2.5, 0.), (0., 2.5), (0., -2.5))
        for x, y in starts:
            reference.point_pos = np.array([x, y], dtype=np.float64)
            reference.point_vel = rng.uniform(-1., 1., 2)
            batch.pos[0] = reference.point_pos.copy()
            batch.vel[0] = reference.point_vel.copy()
            for _ in range(80):
                action = rng.uniform(-1., 1., 2)
                expected_obs, reward_parts, expected_done, _ = reference._step(action)
                actual_obs, actual_reward, actual_done, _truncated, _ = batch.step(action[None])
                np.testing.assert_allclose(actual_obs[0], expected_obs, rtol=2e-6, atol=2e-6)
                self.assertAlmostEqual(float(actual_reward[0]), float(reward_parts[0]), places=5)
                self.assertEqual(bool(actual_done[0]), bool(expected_done))
                np.testing.assert_array_equal(reference.grid_map, batch.grid[0])
                if expected_done:
                    break

    def test_goal_symmetry_and_native_asymmetry(self):
        goals = ((0, -12, 0), (1, 12, 0), (2, 0, 12), (3, 0, -12))
        for profile in ("symmetric", "native"):
            batch = MODULE.FourWayBatch(1, seed=0, reward_profile=profile)
            rewards = []
            for _id, x, y in goals:
                batch.pos[0] = np.array([x, y]) * .25
                batch.vel[0] = 0
                _obs, reward, done, _truncated, info = batch.step(np.zeros((1, 2)))
                self.assertTrue(done[0]); self.assertEqual(info["goal_id"][0], _id)
                rewards.append(float(reward[0]))
            if profile == "symmetric":
                self.assertEqual(rewards, [499.] * 4)
            else:
                self.assertEqual(rewards, [7.8125-1, 62.5-1, 210.9375-1, 500.-1])

    def test_vector_is_independent_and_finite(self):
        batch = MODULE.FourWayBatch(32, seed=123)
        random = np.random.default_rng(5)
        for _ in range(100):
            obs, reward, done, truncated, info = batch.step(random.uniform(-1, 1, (32, 2)))
            self.assertEqual(obs.shape, (32, 4))
            self.assertTrue(np.isfinite(obs).all() and np.isfinite(reward).all())
            self.assertEqual(done.shape, (32,))
            self.assertEqual(truncated.shape, (32,))
            self.assertEqual(info["goal_id"].shape, (32,))
            batch.reset(np.flatnonzero(done | truncated))


if __name__ == "__main__":
    unittest.main()
