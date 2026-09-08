import inspect
import math
from pathlib import Path

import gymnasium as gym
from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import pytest
from stable_baselines3.common.logger import configure

from optiq_dime import OptiQDIME
from optiq_dime.schedules import target_temperature
from run_optiq_dime import validate_config

ROOT = Path(__file__).resolve().parents[1]


def config(name="optiq_dime_reach_target_anneal", overrides=()):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(config_name=name, overrides=list(overrides))


def test_target_comparison_matches_baseline_except_schedule_and_labels():
    baseline = OmegaConf.to_container(
        config("optiq_dime_no_anchor", ["benchmark=reach_hard"]), resolve=True)
    cfg = config()
    assert validate_config(cfg)
    actual = OmegaConf.to_container(cfg, resolve=True)
    assert actual["alg"]["actor"].pop("temperature_schedule") == {
        "initial_temperature": 10.0, "end_steps": 400000}
    assert "proposal_schedule" not in actual["alg"]["actor"]
    for key in ("output_root", "run_name"):
        actual.pop(key)
        baseline.pop(key)
    actual["wandb"].pop("group")
    baseline["wandb"].pop("group")
    assert actual == baseline


@pytest.mark.parametrize("step,expected", [
    (0, 10.0), (200000, math.sqrt(2.5)), (400000, 0.25), (1000000, 0.25),
])
def test_exponential_schedule_endpoints_and_geometric_midpoint(step, expected):
    assert target_temperature(config().alg.actor, step) == pytest.approx(expected)


def test_clock_includes_random_action_warmup_and_has_no_hold():
    actor = config().alg.actor
    assert target_temperature(actor, 5001) == pytest.approx(10 * .025 ** (5001 / 400000))
    temperatures = [target_temperature(actor, t) for t in range(0, 400001, 80000)]
    ratios = np.array(temperatures[1:]) / temperatures[:-1]
    np.testing.assert_allclose(ratios, .025 ** .2)


@pytest.mark.parametrize("override", [
    "initial_temperature=0", "initial_temperature=0.1", "initial_temperature=nan",
    "end_steps=0", "end_steps=1.5", "end_steps=true", "end_steps=1000001",
])
def test_invalid_target_schedule_rejected(override):
    with pytest.raises(ValueError, match="temperature_schedule"):
        validate_config(config(overrides=["alg.actor.temperature_schedule." + override]))


def test_no_schedule_preserves_configured_temperature_at_all_steps():
    actor = config("optiq_dime_no_anchor").alg.actor
    for step in (0, 5001, 250000, 1000000):
        assert target_temperature(actor, step) == 0.25


@pytest.mark.parametrize("benchmark,env_id", [("ant", "Ant-v4"), ("half_cheetah", "HalfCheetah-v4")])
def test_actual_training_anneals_q_temperature_with_fixed_kde_td_and_beta(benchmark, env_id):
    cfg = config(overrides=[
        f"benchmark={benchmark}", "alg.critic.hs=[32,32]", "alg.actor.hidden_dims=[32,32]",
        "alg.buffer_size=32", "alg.batch_size=4", "alg.learning_starts=2",
        "alg.actor.learning_starts=2", "alg.actor.temperature_schedule.end_steps=6",
    ])
    model = OptiQDIME("MlpPolicy", gym.make(env_id), None, 1, cfg)
    model.set_logger(configure(None, []))
    captured = []
    original = model._train
    signature = inspect.signature(original)

    def tracked(*args, **kwargs):
        bound = signature.bind(*args, **kwargs).arguments
        names = ("temperature", "proposal_std", "proposal_clip", "density_beta",
                 "td_noise_std", "td_noise_clip", "sinkhorn_epsilon")
        captured.append({k: bound[k] for k in names})
        return original(*args, **kwargs)

    model._train = tracked
    try:
        model.learn(total_timesteps=8)
        assert len(captured) == 6
        for step, record in zip(range(3, 9), captured):
            expected_temperature = 10 * .025 ** min(step / 6, 1)
            assert record["temperature"] == pytest.approx(expected_temperature)
            assert {k: v for k, v in record.items() if k != "temperature"} == {
                "proposal_std": .2, "proposal_clip": .5, "density_beta": .1,
                "td_noise_std": .2, "td_noise_clip": .5, "sinkhorn_epsilon": .05,
            }
        assert model._n_updates == 12
        assert model.logger.name_to_value["train/temperature"] == .25
        assert model.logger.name_to_value["train/proposal_temperature"] == 1
        assert np.isfinite(model.logger.name_to_value["train/actor_loss"])
    finally:
        model.get_env().close()


@pytest.mark.parametrize("benchmark", ["ant", "half_cheetah"])
def test_mujoco_target_preserves_reach_protocol_except_environment_support_and_output(benchmark):
    reach = OmegaConf.to_container(config(), resolve=True)
    cfg = config("optiq_dime_mujoco_target_anneal", [f"benchmark={benchmark}"])
    assert validate_config(cfg)
    actual = OmegaConf.to_container(cfg, resolve=True)
    assert actual["alg"]["critic"]["v_min"] == -1600
    assert actual["alg"]["critic"]["v_max"] == 1600
    assert actual["seed"] == 0
    assert actual["successful_steps"] is None
    for value in (reach, actual):
        for key in ("env_name", "task", "successful_steps", "output_root", "run_name"):
            value.pop(key)
        value["wandb"].pop("group")
        for key in ("v_min", "v_max"):
            value["alg"]["critic"].pop(key)
    assert actual == reach
