"""Collection-only uniform exploration: action/replay consistency and isolation."""

import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gymnasium as gym
from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import pytest
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

from optiq_dime import OptiQDIME
from run_optiq_dime import validate_config

ROOT = Path(__file__).resolve().parents[1]


def config(name="mujoco_humanoid_behavior010", overrides=()):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(config_name=name, overrides=list(overrides))


@pytest.mark.parametrize("seed", range(4))
def test_only_behavior_probability_changes_latest_remote_training_config(seed):
    base = OmegaConf.to_container(config("mujoco_setting", [f"seed={seed}"]), resolve=True)
    cfg = config(overrides=[f"seed={seed}"])
    assert validate_config(cfg)
    actual = OmegaConf.to_container(cfg, resolve=True)
    expected = copy.deepcopy(base["alg"])
    expected["behavior_uniform_probability"] = .1
    assert actual["alg"] == expected
    for key in base:
        if key not in {"alg", "wandb", "run_name", "output_root"}:
            assert actual[key] == base[key]
    assert actual["env_name"] == "Humanoid-v4"
    assert actual["alg"]["actor"]["include_anchor"] is True
    assert actual["alg"]["actor"]["temperature"] == .25
    assert actual["alg"]["optimizer"]["ac_grad_norm"] == 2


@pytest.mark.parametrize("probability", [-.1, 1.1, float("nan"), float("inf")])
def test_invalid_probability_is_rejected(probability):
    cfg = config()
    cfg.alg.behavior_uniform_probability = probability
    with pytest.raises(ValueError, match="behavior_uniform_probability"):
        validate_config(cfg)


def collection_stub(probability=.1, seed=0, timestep=5000):
    model = object.__new__(OptiQDIME)
    model.behavior_uniform_probability = probability
    model.behavior_rng = np.random.default_rng(np.random.SeedSequence([seed, 510010]))
    model.behavior_uniform_count = model.behavior_action_count = 0
    model.num_timesteps = timestep
    model.policy = SimpleNamespace(unscale_action=lambda action: 2.0 + 3.0 * action)
    model.set_logger(configure(None, []))
    return model


@pytest.mark.parametrize("probability,timestep", [(0., 5000), (.1, 4999)])
def test_disabled_and_warmup_keep_original_actions_without_rng_or_count_changes(probability, timestep):
    model = collection_stub(probability, timestep=timestep)
    action, replay = np.ones((1, 3)), np.zeros((1, 3))
    before = copy.deepcopy(model.behavior_rng.bit_generator.state)
    with patch.object(OffPolicyAlgorithm, "_sample_action", return_value=(action, replay)) as parent:
        output, stored = model._sample_action(5000)
    assert output is action and stored is replay
    parent.assert_called_once_with(5000, None, 1)
    assert model.behavior_rng.bit_generator.state == before
    assert model.behavior_action_count == model.behavior_uniform_count == 0


def test_one_decision_per_vector_and_non_unit_action_bounds():
    model = collection_stub()
    uniform = np.array([[-.8, -.2, .4], [.2, .4, .6], [.9, -.9, .1]], dtype=np.float64)
    model.behavior_rng = SimpleNamespace(random=lambda n: np.array([.02, .8, .05]),
                                        uniform=lambda low, high, size: uniform)
    original = np.full((3, 3), .25, dtype=np.float32)
    with patch.object(OffPolicyAlgorithm, "_sample_action", return_value=(2+3*original, original)):
        action, replay = model._sample_action(5000, n_envs=3)
    expected = original.copy()
    expected[[0, 2]] = uniform[[0, 2]]
    np.testing.assert_array_equal(replay, expected)
    np.testing.assert_allclose(action, 2+3*expected)
    assert replay.dtype == np.float32
    assert model.behavior_uniform_count == 2 and model.behavior_action_count == 3


def test_random_stream_reproducibility_independence_and_ten_percent_rate():
    first, second, other = collection_stub(), collection_stub(), collection_stub(seed=1)
    original = np.full((20000, 3), .314, dtype=np.float32)
    global_state = np.random.get_state()
    with patch.object(OffPolicyAlgorithm, "_sample_action", return_value=(original, original)):
        a = first._sample_action(5000, n_envs=20000)[1]
        b = second._sample_action(5000, n_envs=20000)[1]
        c = other._sample_action(5000, n_envs=20000)[1]
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    after = np.random.get_state()
    assert global_state[0] == after[0]
    np.testing.assert_array_equal(global_state[1], after[1])
    assert global_state[2:] == after[2:]
    replaced = np.any(a != original, axis=-1)
    assert replaced.sum() == first.behavior_uniform_count
    assert .09 < replaced.mean() < .11
    assert np.abs(a).max() <= 1
    assert np.max(np.abs(a[replaced].mean(axis=0))) < .05


class RecordingEnv(gym.Env):
    observation_space = gym.spaces.Box(-100., 100., shape=(2,), dtype=np.float32)
    action_space = gym.spaces.Box(np.array([-2., 0., 10.], dtype=np.float32),
                                  np.array([3., 4., 20.], dtype=np.float32))

    def __init__(self):
        self.actions = []
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        self.actions.append(action.copy())
        self.steps += 1
        return np.full(2, self.steps, dtype=np.float32), float(action.sum()), self.steps == 4, False, {}


def test_real_collector_stores_executed_actions_and_evaluation_bypasses_exploration():
    cfg = config(overrides=["alg.behavior_uniform_probability=1.0", "alg.learning_starts=2",
                            "alg.actor.learning_starts=2", "alg.buffer_size=32", "alg.batch_size=4"])
    env = RecordingEnv()
    model = OptiQDIME("MlpPolicy", env, None, 1, cfg)
    model.gradient_steps = 0  # Isolate collection; GPU smoke separately exercises training.
    model.set_logger(configure(None, []))
    try:
        model.learn(total_timesteps=8)
        executed = np.asarray(env.actions)
        replay = model.replay_buffer.actions[:8, 0]
        np.testing.assert_allclose(model.policy.unscale_action(replay), executed, atol=1e-6)
        np.testing.assert_allclose(model.replay_buffer.rewards[:8, 0], executed.sum(axis=1))
        assert model.behavior_action_count == model.behavior_uniform_count == 6
        assert int(model.policy.actor_state.step) == 0
        state = copy.deepcopy(model.behavior_rng.bit_generator.state)
        evaluation = RecordingEnv()
        with patch.object(model, "_sample_action", side_effect=AssertionError("evaluation used collector")):
            evaluate_policy(model, evaluation, n_eval_episodes=1, deterministic=False, warn=False)
        assert model.behavior_rng.bit_generator.state == state
        assert model.behavior_action_count == model.behavior_uniform_count == 6
    finally:
        model.get_env().close()
        model.logger.close()
