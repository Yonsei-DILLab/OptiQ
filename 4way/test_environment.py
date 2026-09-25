"""Task-level checks for the official-source vector physics and goals."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np


SPEC = importlib.util.spec_from_file_location("official_fourway", Path(__file__).with_name("environment.py"))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FourWayTests(unittest.TestCase):
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
