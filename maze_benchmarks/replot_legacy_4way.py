"""Probe a preserved wall-free OptiQ 4-Way policy without retraining it."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
from pathlib import Path

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

from .visualize_4way import probe_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=3.)
    parser.add_argument("--source-root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    legacy_path = args.source_root / "experiments/4way/train.py"
    spec = importlib.util.spec_from_file_location("legacy_fourway_train", legacy_path)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)

    trainer = legacy.load_trainer()
    cfg = legacy.load_policy_config(trainer, args.probe.parent, 20_000, 1_000, args.temperature)
    model = trainer.runner.OptiQDIME(
        "MlpPolicy", env=legacy.SpaceOnlyEnv(), cfg=cfg,
        model_save_path=None, save_every_n_steps=20_000)
    policy = model.policy
    template = dict(actor=policy.actor_state, critic=policy.qf_state,
                    target_actor=policy.target_actor_state,
                    target_critic_params=policy.qf_state.target_params)
    restored = flax.serialization.from_bytes(template, args.checkpoint.read_bytes())
    policy.actor_state = restored["actor"]
    policy.qf_state = restored["critic"]
    policy.target_actor_state = restored["target_actor"]

    class FrozenPolicy:
        method = "optiq"
        critic_label = "mean of two live OptiQ critics"

        @contextmanager
        def evaluation_rng(self, seed):
            key, noise_key = policy.key, policy.noise_key
            policy.key = jax.random.PRNGKey(seed)
            policy.noise_key = jax.random.PRNGKey(seed + 1)
            try:
                yield
            finally:
                policy.key, policy.noise_key = key, noise_key

        def act(self, obs, mode="mu_only"):
            policy.reset_noise()
            actions = policy.sample_action(
                policy.actor_state, jnp.asarray(obs, jnp.float32), policy.noise_key,
                deterministic=False, sample_conditional_noise=mode != "mu_only")
            return np.clip(np.asarray(actions), -1., 1.)

        def q(self, obs, actions):
            state = policy.qf_state
            values = state.apply_fn(
                {"params": state.params, "batch_stats": state.batch_stats},
                jnp.asarray(obs, jnp.float32), jnp.asarray(actions, jnp.float32),
                rngs={"dropout": jax.random.PRNGKey(31415)}, train=False)
            return np.asarray(values).mean(axis=0).reshape(-1)

    probe_policy(FrozenPolicy(), args.probe)


if __name__ == "__main__":
    main()
