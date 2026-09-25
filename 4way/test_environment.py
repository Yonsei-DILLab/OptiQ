"""The main batched four-goal task must equal the already-used scalar task."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MainFourWayTests(unittest.TestCase):
    def test_vector_matches_existing_environment(self):
        scalar_module = load(HERE / "environment.py", "main_fourway_scalar")
        batch_module = load(HERE / "vector_environment.py", "main_fourway_vector")
        scalar = [scalar_module.FourWayEnv() for _ in range(8)]
        vector = batch_module.FourWayBatch(8, seed=9)
        for env in scalar:
            env.reset()
        rng = np.random.default_rng(19)
        for _ in range(90):
            actions = rng.uniform(-1.3, 1.3, (8, 2)).astype(np.float32)
            actual = vector.step(actions)
            for i, env in enumerate(scalar):
                expected_obs, expected_reward, done, truncated, info = env.step(actions[i])
                np.testing.assert_array_equal(actual[0][i], expected_obs)
                self.assertAlmostEqual(float(actual[1][i]), expected_reward, places=4)
                self.assertEqual(bool(actual[2][i]), done)
                self.assertEqual(bool(actual[3][i]), truncated)
                self.assertEqual(int(actual[4][i]), info["goal_id"])
                if done or truncated:
                    env.reset()
                    vector.reset([i])


if __name__ == "__main__":
    unittest.main()
