from pathlib import Path
import subprocess
import types

from flax.training.train_state import TrainState
import gymnasium as gym
from hydra import compose, initialize_config_dir
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import pytest
import optax
from stable_baselines3.common.logger import KVWriter, configure

from common.type_aliases import RLTrainState
from optiq_dime import OptiQDIME
from optiq_dime.critic_utils import critic_expectation, scalar_td_loss
from run_optiq_dime import validate_config

ROOT = Path(__file__).resolve().parents[1]


def config(name="optiq_scalar_mujoco_target_anneal", overrides=()):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(config_name=name, overrides=list(overrides))


@pytest.mark.parametrize("benchmark", ["ant", "half_cheetah"])
def test_scalar_matches_mujoco_setting_critic_and_retains_temperature_experiment(benchmark):
    cfg = config(overrides=[f"benchmark={benchmark}"])
    assert validate_config(cfg)
    actual = OmegaConf.to_container(cfg, resolve=True)
    baseline = OmegaConf.to_container(config("optiq_dime_mujoco_target_anneal",
                                            [f"benchmark={benchmark}"]), resolve=True)
    assert actual["alg"]["critic"].pop("type") == "scalar"
    assert actual["alg"]["critic"].pop("crossq_style") is False
    for key, expected in {"hs": [256, 256, 256], "activation": "gelu"}.items():
        assert actual["alg"]["critic"].pop(key) == expected
        baseline["alg"]["critic"].pop(key)
    for key, expected in {"n_atoms": 1, "v_min": None, "v_max": None, "entr_coeff": 0.0}.items():
        assert actual["alg"]["critic"].pop(key) == expected
        baseline["alg"]["critic"].pop(key)
    for key, expected in {"tau": .005, "utd": 1}.items():
        assert actual["alg"].pop(key) == expected
        baseline["alg"].pop(key)
    for key, expected in {"bn": False, "critic_b1": .9}.items():
        assert actual["alg"]["optimizer"].pop(key) == expected
        baseline["alg"]["optimizer"].pop(key)
    for value in (actual, baseline):
        value.pop("run_name")
        value.pop("output_root")
        value["wandb"].pop("group")
    assert actual == baseline
    assert cfg.alg.actor.density_beta == .1
    assert cfg.alg.actor.temperature_schedule.initial_temperature == 10
    assert cfg.alg.actor.temperature_schedule.end_steps == 400000


@pytest.mark.parametrize("override", [
    "alg.critic.n_atoms=101", "alg.critic.entr_coeff=0.005",
    "alg.critic.v_min=-1600", "alg.critic.type=unknown",
    "alg.critic.n_atoms=true",
])
def test_incompatible_scalar_settings_fail_before_training(override):
    with pytest.raises(ValueError):
        validate_config(config(overrides=[override]))


def test_atom_count_alone_does_not_silently_switch_critic_type():
    with pytest.raises(ValueError, match="critic.type=scalar"):
        validate_config(config("optiq_dime_mujoco_target_anneal", ["alg.critic.n_atoms=1"]))


def test_scalar_q_is_unbounded_and_not_multiplied_by_categorical_support():
    values = jnp.array([[[-5000.0], [8000.0]], [[-4000.0], [9000.0]]])
    np.testing.assert_array_equal(critic_expectation(values, jnp.array([-1600.])), values[..., 0])
    probabilities = jnp.array([[0.25, 0.75]])
    np.testing.assert_allclose(critic_expectation(probabilities, jnp.array([-2., 2.])), [1.])


def test_scalar_td_uses_min_bootstrap_terminal_mask_no_clipping_and_stopped_target():
    current = jnp.array([[0., 3., 4500.], [6., 1., 4498.]])
    next_q = jnp.array([[2., 1000., 4000.], [6., -1000., 6000.]])
    rewards, dones = jnp.array([1., 3., 2000.]), jnp.array([0., 1., 0.])
    loss, target = scalar_td_loss(current, next_q, rewards, dones, .5)
    np.testing.assert_array_equal(target, [2., 3., 4000.])
    expected = np.square(np.array(current) - np.array(target)[None, :]).mean(axis=1).sum()
    assert float(loss) == pytest.approx(expected)
    grad = jax.grad(lambda q: scalar_td_loss(current, q, rewards, dones, .5)[0])(next_q)
    np.testing.assert_array_equal(grad, jnp.zeros_like(next_q))


class CaptureMetrics(KVWriter):
    def __init__(self):
        self.records = []

    def write(self, key_values, key_excluded, step=0):
        self.records.append((step, dict(key_values)))

    def close(self):
        pass


@pytest.mark.parametrize("benchmark,env_id", [("ant", "Ant-v4"), ("half_cheetah", "HalfCheetah-v4")])
def test_scalar_shared_training_updates_actor_and_critic_without_bn(benchmark, env_id, tmp_path):
    cfg = config(overrides=[f"benchmark={benchmark}", "alg.buffer_size=32",
                           "alg.batch_size=4", "alg.learning_starts=2",
                           "alg.actor.learning_starts=2", "diagnostic_interval=4",
                           "alg.actor.temperature_schedule.end_steps=6"])
    assert validate_config(cfg)
    model = OptiQDIME("MlpPolicy", gym.make(env_id), str(tmp_path), 4, cfg)
    model.set_logger(configure(None, []))
    capture = CaptureMetrics()
    model.logger.output_formats.append(capture)
    actor_before = jax.tree_util.tree_leaves(model.policy.actor_state.params)
    critic_before = jax.tree_util.tree_leaves(model.policy.qf_state.params)
    try:
        assert model.crossq_style is False
        assert model.tau == .005
        assert list(model.policy.qf.net_arch) == [256, 256, 256]
        model.learn(total_timesteps=8)
        assert model._n_updates == 6
        assert not model.policy.qf_state.batch_stats
        assert not model.policy.qf_state.target_batch_stats
        assert any(not np.array_equal(a, b) for a, b in zip(
            jax.tree_util.tree_leaves(model.policy.qf_state.params),
            jax.tree_util.tree_leaves(model.policy.qf_state.target_params)))
        for before, state in [(actor_before, model.policy.actor_state.params),
                              (critic_before, model.policy.qf_state.params)]:
            assert any(not np.array_equal(a, b) for a, b in zip(before, jax.tree_util.tree_leaves(state)))
        q = model.policy.predict_critic(jnp.zeros((4, model.observation_space.shape[0])),
                                        jnp.zeros((4, model.action_space.shape[0])))
        assert q.shape == (2, 4, 1)
        assert np.isfinite(q).all()
        step, metrics = capture.records[-1]
        assert step == 8
        assert np.isfinite(metrics["train/critic_loss"])
        assert metrics["train/critic_td_rmse"] ** 2 * 2 == pytest.approx(metrics["train/critic_loss"], rel=1e-5)
        assert "train/entrQ_1" not in metrics and "train/entrQ_2" not in metrics
        assert metrics["train/temperature"] == .25
        assert metrics["train/proposal_temperature"] == 1
        assert 1 <= metrics["train/source_ess_absolute"] <= 64.001
        assert 1 <= metrics["train/counterfactual_ess_T0p25"] * 64 <= 64.001
        assert (tmp_path / "critic_state_8.msgpack").stat().st_size > 0
        assert (tmp_path / "actor_state_8.msgpack").stat().st_size > 0
    finally:
        model.get_env().close()


def constant_critic(variables, observations, actions, rngs=None, mutable=False, train=False):
    values = jnp.broadcast_to(variables["params"]["q"][:, None, None],
                             (2, observations.shape[0], 1))
    return (values, {}) if mutable else values


def zero_actor(variables, observations, latents):
    return jnp.zeros_like(latents)


def test_scalar_target_network_backup_gradients_and_polyak_match_mujoco_branch():
    actor = TrainState.create(apply_fn=zero_actor,
        params={"Dense_0": {"bias": jnp.zeros(1)}}, tx=optax.sgd(.01))
    critic = RLTrainState.create(apply_fn=constant_critic,
        params={"q": jnp.array([2., 6.])}, batch_stats={},
        target_params={"q": jnp.array([10., 14.])}, target_batch_stats={}, tx=optax.sgd(.01))
    obs, actions = jnp.zeros((2, 3)), jnp.zeros((2, 1))
    args = (False, False, .9, actor, critic, obs, actions, obs,
            jnp.array([1., 10000.]), jnp.array([0., 1.]),
            1, jnp.ones((1,)), None, None, 0., .2, .5, jax.random.PRNGKey(5))
    updated, metrics, key = OptiQDIME.update_critic(*args)
    target = np.array([10., 10000.])
    current = np.array([2., 6.])
    expected_loss = np.square(current[:, None] - target[None, :]).mean(axis=1).sum()
    np.testing.assert_allclose(metrics["critic_loss"], expected_loss, rtol=1e-6)
    expected_params = current - .01 * 2 * (current - target.mean())
    np.testing.assert_allclose(updated.params["q"], expected_params, rtol=1e-6)
    soft = OptiQDIME.soft_update(.005, updated)
    np.testing.assert_allclose(soft.target_params["q"],
        .995 * np.array([10., 14.]) + .005 * expected_params, rtol=1e-6)
    reference = reference_algorithm("7e2da67")
    expected, ref_metrics, ref_key = reference.OptiQDIME.update_critic(*args)
    for a, b in zip(jax.tree_util.tree_leaves(expected), jax.tree_util.tree_leaves(updated)):
        np.testing.assert_array_equal(a, b)
    for name in ("critic_loss", "current_q_values", "next_q_values"):
        np.testing.assert_array_equal(metrics[name], ref_metrics[name])
    np.testing.assert_array_equal(key, ref_key)


def reference_algorithm(commit):
    reference = types.ModuleType("optiq_dime.reference_algorithm")
    reference.__package__ = "optiq_dime"
    source = subprocess.check_output(["git", "show", f"{commit}:optiq_dime/algorithm.py"], cwd=ROOT, text=True)
    exec(compile(source, "reference_algorithm.py", "exec"), reference.__dict__)
    return reference


def test_distributional_critic_update_matches_pre_scalar_implementation():
    reference = reference_algorithm("575677d")
    cfg = config("optiq_dime_mujoco_target_anneal", ["alg.critic.hs=[32,32]",
                 "alg.actor.hidden_dims=[32,32]", "alg.buffer_size=32", "alg.batch_size=4"])
    model = OptiQDIME("MlpPolicy", gym.make("Ant-v4"), None, 1, cfg)
    try:
        args = (True, False, .99, model.policy.target_actor_state, model.policy.qf_state,
                jnp.zeros((4, 27)), jnp.zeros((4, 8)), jnp.ones((4, 27)),
                jnp.array([1., -1., 2., 3.]), jnp.array([0., 1., 0., 1.]),
                101, jnp.linspace(-1600, 1600, 101), -1600, 1600, .005, .2, .5,
                jax.random.PRNGKey(17))
        expected = reference.OptiQDIME.update_critic(*args)
        actual = OptiQDIME.update_critic(*args)
        for a, b in zip(jax.tree_util.tree_leaves(expected), jax.tree_util.tree_leaves(actual)):
            np.testing.assert_array_equal(a, b)
    finally:
        model.get_env().close()
