"""Exercise collection, entropy control, replay and learner isolation on CPU."""
import copy
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
from hydra import compose, initialize_config_dir
import jax
import numpy as np
from omegaconf import OmegaConf
import pytest
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure

from optiq_dime.algorithm import OptiQDIME
from optiq_dime.collection_exploration import (
    AlphaController, CollectionExplorationOptiQDIME, ExplorationSettings, gmm_entropy_proxy,
)
from run_optiq_dime import validate_config

ROOT = Path(__file__).resolve().parents[1]


def config(name="mujoco_v3_exploration", overrides=()):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(config_name=name, overrides=["benchmark=ant", *overrides])


@pytest.mark.parametrize("mode,target", [("fixed", 0.), ("adaptive", -.9), ("adaptive", -.5), ("adaptive", 0.)])
def test_only_collection_configuration_changes(mode, target):
    base = OmegaConf.to_container(config("mujoco_v3"), resolve=True)
    cfg = config(overrides=[f"alg.collection_exploration.mode={mode}",
                            f"alg.collection_exploration.target_entropy_per_dim={target}"])
    assert validate_config(cfg)
    actual = OmegaConf.to_container(cfg, resolve=True)
    actual["alg"].pop("collection_exploration")
    actual["wandb"] = base["wandb"]
    assert actual == base


@pytest.mark.parametrize("changes", [{"mode": "bad"}, {"initial_alpha": 0}, {"alpha_lr": float("nan")},
    {"target_entropy_per_dim": float("inf")}, {"update_interval": 1.5}, {"entropy_samples": 2}])
def test_invalid_settings(changes):
    with pytest.raises(ValueError):
        ExplorationSettings(**changes)


def test_alpha_feedback_direction_and_zero_error():
    low, high, equal = [AlphaController(1.5, .03) for _ in range(3)]
    low.update(-8., -4.)
    high.update(0., -4.)
    equal.update(-4., -4.)
    assert low.alpha > 1.5 > high.alpha
    assert equal.alpha == 1.5
    assert low.alpha == pytest.approx(1.5 * np.exp(.03))


def test_entropy_is_conditional_on_state_and_increases_with_spread():
    rng = np.random.default_rng(7)
    small = rng.normal(0, .08, (4, 500, 2))
    large = small * 2
    shifts = np.array([-5., -1., 1., 5.])[:, None, None]
    h_small, _ = gmm_entropy_proxy(small, components=1)
    h_large, _ = gmm_entropy_proxy(large, components=1)
    h_shifted, _ = gmm_entropy_proxy(small + shifts, components=1)
    np.testing.assert_allclose(h_shifted, h_small, atol=1e-10)
    np.testing.assert_allclose(h_large - h_small, 2*np.log(2), atol=1e-3)
    h_clipped, convergence = gmm_entropy_proxy(np.clip(large, -.1, .1), components=3)
    assert np.isfinite(h_clipped).all() and 0 <= convergence <= 1


class RecordingEnv(gym.Env):
    observation_space = gym.spaces.Box(-100, 100, shape=(2,), dtype=np.float32)
    action_space = gym.spaces.Box(-2, 4, shape=(2,), dtype=np.float32)

    def __init__(self):
        self.actions = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros(2, np.float32), {}

    def step(self, action):
        self.actions.append(action.copy())
        self.steps += 1
        return np.full(2, self.steps, np.float32), float(action.sum()), self.steps == 4, False, {}


def make_model(mode="adaptive"):
    cfg = config(overrides=[f"alg.collection_exploration.mode={mode}", "alg.learning_starts=2",
        "alg.actor.learning_starts=2", "alg.buffer_size=64", "alg.batch_size=4",
        "alg.collection_exploration.update_interval=2", "alg.collection_exploration.entropy_states=4",
        "alg.collection_exploration.entropy_samples=32", "alg.collection_exploration.recent_capacity=8",
        "alg.actor.hidden_dims=[8,8]", "alg.critic.hs=[8,8]"])
    env = RecordingEnv()
    model = CollectionExplorationOptiQDIME("MlpPolicy", env, None, 1, cfg)
    model.gradient_steps = 0
    model.set_logger(configure(None, []))
    return model, env


@pytest.mark.parametrize("mode", ["fixed", "adaptive"])
def test_replay_matches_executed_actions_and_evaluation_bypasses_noise(mode):
    model, env = make_model(mode)
    try:
        model.learn(total_timesteps=12)
        np.testing.assert_allclose(model.policy.unscale_action(model.replay_buffer.actions[:12, 0]),
                                   np.asarray(env.actions), atol=1e-6)
        np.testing.assert_allclose(model.replay_buffer.rewards[:12, 0], np.asarray(env.actions).sum(axis=1))
        assert model.exploration_action_count == 10  # Warmup uses the original path.
        assert model.recent_count == 8
        before = copy.deepcopy(model.exploration_noise_rng.bit_generator.state)
        with patch.object(model, "_sample_action", side_effect=AssertionError("Evaluation entered collector")):
            evaluate_policy(model, RecordingEnv(), n_eval_episodes=1, deterministic=False, warn=False)
        assert before == model.exploration_noise_rng.bit_generator.state
    finally:
        model.get_env().close()


def test_entropy_probe_does_not_touch_actor_optimizer_or_training_rollout_rngs():
    model, _ = make_model()
    try:
        model.learn(total_timesteps=12)
        model._n_updates = 2
        actor = [np.asarray(x).copy() for x in jax.tree_util.tree_leaves(model.policy.actor_state)]
        rngs = [np.asarray(x).copy() for x in (model.key, model.policy.key, model.policy.noise_key)]
        np_rng = np.random.get_state()
        noise_rng = copy.deepcopy(model.exploration_noise_rng.bit_generator.state)
        model._update_exploration()
        assert model.exploration_alpha.updates == 1
        assert model.last_exploration_update == 2
        for before, after in zip(actor, jax.tree_util.tree_leaves(model.policy.actor_state)):
            np.testing.assert_array_equal(before, after)
        for before, after in zip(rngs, (model.key, model.policy.key, model.policy.noise_key)):
            np.testing.assert_array_equal(before, after)
        np.testing.assert_array_equal(np_rng[1], np.random.get_state()[1])
        assert noise_rng == model.exploration_noise_rng.bit_generator.state
        model._update_exploration()
        assert model.exploration_alpha.updates == 1
        assert CollectionExplorationOptiQDIME.update_actor is OptiQDIME.update_actor
        assert CollectionExplorationOptiQDIME.update_critic is OptiQDIME.update_critic
        assert CollectionExplorationOptiQDIME._train.__func__ is OptiQDIME._train.__func__
    finally:
        model.get_env().close()


def test_real_training_and_exploration_checkpoint(tmp_path):
    model, _ = make_model()
    model.gradient_steps = 1
    model.model_save_path = str(tmp_path)
    model.save_every_n_steps = 10
    try:
        model.learn(total_timesteps=12)
        model._save_model()
        assert model._n_updates == int(model.policy.actor_state.step) == 10
        assert model.exploration_alpha.updates > 0
        assert model.ent_coef_state.apply_fn({"params": model.ent_coef_state.params}, 0) == 0
        assert (tmp_path / "exploration_state_12.json").exists()
        assert (tmp_path / "actor_state_12.msgpack").exists()
    finally:
        model.get_env().close()
