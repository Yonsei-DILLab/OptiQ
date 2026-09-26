"""Evaluation draws a new latent, omits conditional noise, and preserves training."""
from types import SimpleNamespace
import tempfile
import unittest
from pathlib import Path
import jax
import numpy as np
from .agents import OptiQ
from .run import evaluate


class SamplingTest(unittest.TestCase):
    def test_random_latent_mu_only_and_training_noise(self):
        calls = []
        class Policy:
            actor_state = None
            key = jax.random.PRNGKey(13)
            def reset_noise(self):
                self.key, self.noise_key = jax.random.split(self.key)
            def sample_action(self, state, obs, key, **kwargs):
                calls.append(kwargs)
                return jax.random.normal(key, obs.shape) * .1
        actor = OptiQ.__new__(OptiQ)
        actor.model = SimpleNamespace(policy=Policy())
        obs = np.zeros((8, 2), np.float32)
        a, b = actor.act(obs), actor.act(obs)
        self.assertFalse(np.array_equal(a, b))
        self.assertTrue(all(c == dict(deterministic=False, sample_conditional_noise=False) for c in calls))
        actor.act(obs, mode="train")
        self.assertTrue(calls[-1]["sample_conditional_noise"])
        self.assertFalse(calls[-1]["deterministic"])

    def test_rollout_rejects_sigma_for_optiq(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "mu_only"):
                evaluate(SimpleNamespace(method="optiq"), "4way", 0, 5, "policy", Path(folder)/"x.npz")


if __name__ == "__main__":
    unittest.main()
