"""Plain-TD arithmetic, absence of policy entropy, and end-to-end v3 updates."""
import csv
from pathlib import Path

from flax import serialization
import gymnasium as gym
from hydra import compose, initialize_config_dir
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import pytest
from stable_baselines3.common.logger import configure

from optiq_dime import OptiQDIME
import optiq_dime.algorithm as algorithm
from optiq_dime.policy import OptiQPolicy
from run_optiq_dime import validate_config
from test_semi_implicit import actor_state, critic_state

ROOT = Path(__file__).resolve().parents[1]


def config(name="mujoco_v3", overrides=()):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        return compose(config_name=name, overrides=list(overrides))


@pytest.mark.parametrize("name", ["mujoco_v3", "v3/final"])
def test_v3_has_only_intended_algorithm_changes(name):
    cfg = config(name)
    assert validate_config(cfg)
    expected = OmegaConf.to_container(config("v2/final").alg, resolve=True)
    expected["ent_coef"]["init"] = 0.0
    expected["critic"]["backup_mode"] = "td"
    expected["actor"]["entropy_samples"] = 0
    expected["actor"]["entropy_diagnostics"] = False
    expected["actor"]["soft_guard"] = {"enabled": False}
    assert OmegaConf.to_container(cfg.alg, resolve=True) == expected
    assert cfg.wandb.project == config("v2/final").wandb.project
    assert "v3" in cfg.output_root and "v3" in cfg.wandb.group


@pytest.mark.parametrize("override", [
    "alg.ent_coef.init=0.1", "alg.ent_coef.type=auto",
    "alg.actor.soft_guard.enabled=true", "alg.critic.backup_mode=unknown",
    "alg.critic.backup_mode=soft_td", "alg.actor.temperature=0.0",
])
def test_v3_rejects_conflicting_backup_settings(override):
    with pytest.raises(ValueError):
        validate_config(config(overrides=[override]))


def forbidden_entropy(*args, **kwargs):
    raise AssertionError("v3 must not evaluate policy entropy or the soft guard")


def action_dependent_critic(variables, observations, actions, rngs=None, mutable=False, train=False):
    q = variables["params"]["q"][:, None, None]
    values = q + (actions[:, 0] + 2.0 * actions[:, 1])[None, :, None]
    return (values, {"batch_stats": {}}) if mutable else values


def test_td_target_sampled_action_terminal_temperature_and_stopped_gradient(monkeypatch):
    monkeypatch.setattr(algorithm, "idac_action_and_log_density", forbidden_entropy)
    actor = actor_state()
    critic = critic_state().replace(apply_fn=action_dependent_critic)
    obs = jnp.ones((2, 3))
    actions = jnp.zeros((2, 2))
    rewards, dones = jnp.array([1., 3.]), jnp.array([0., 1.])
    key = jax.random.PRNGKey(123)
    OptiQDIME.update_critic.clear_cache()

    def update(a=actor, temperature=.1, samples=0):
        return OptiQDIME.update_critic(
            False, False, .9, a, critic, obs, actions, obs, rewards, dones,
            1, jnp.array([-3600.]), -3600., 3600., 0., 0., 0., key,
            True, samples, temperature, "td",
        )

    next_actions = OptiQPolicy.sample_action(actor, obs, jax.random.split(key, 6)[1])
    expected_q = 10. + np.asarray(next_actions[:, 0] + 2 * next_actions[:, 1])
    target = np.asarray(rewards) + .9 * (1 - np.asarray(dones)) * expected_q
    expected_loss = ((np.array([2., 6.])[:, None] - target) ** 2).mean(axis=1).sum()
    updated, metrics, next_key = update()
    np.testing.assert_allclose(metrics["critic_loss"], expected_loss, rtol=2e-6)
    np.testing.assert_allclose(metrics["next_q_values"], target.mean(), rtol=2e-6)
    assert target[1] == 3.
    for name in ("ent_coef", "backup_entropy_term", "backup_discounted_entropy_term"):
        assert float(metrics[name]) == 0.
    assert "backup_entropy_lower" not in metrics
    # Neither teacher temperature nor the unused entropy sample count enters TD.
    other, other_metrics, other_key = update(temperature=17., samples=91)
    for a, b in zip(jax.tree_util.tree_leaves(updated.params), jax.tree_util.tree_leaves(other.params)):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(metrics["critic_loss"], other_metrics["critic_loss"])
    np.testing.assert_array_equal(next_key, other_key)
    gradients = jax.grad(lambda p: update(actor.replace(params=p))[1]["critic_loss"])(actor.params)
    assert all(np.count_nonzero(x) == 0 for x in jax.tree_util.tree_leaves(gradients))


def test_actor_update_retains_full_nll_when_entropy_diagnostics_are_disabled():
    actor, critic = actor_state(), critic_state()
    args = (actor, critic, jnp.ones((2, 3)), jax.random.PRNGKey(57),
            jnp.array([-3600.]), 16, 4, "exact", .05, .5, False, True, 1.,
            False, 16., 257, .1, .25, 100, "mean", "argmax", True, False,
            "conditional_ot_nll", "conditional_mixture")
    with_entropy = OptiQDIME.update_actor(*args, entropy_diagnostics=True)
    without_entropy = OptiQDIME.update_actor(*args, entropy_diagnostics=False)
    for a, b in zip(jax.tree_util.tree_leaves(with_entropy[0]),
                    jax.tree_util.tree_leaves(without_entropy[0])):
        np.testing.assert_allclose(a, b, atol=2e-7, rtol=2e-6)
    np.testing.assert_allclose(with_entropy[3]["actor_loss"], without_entropy[3]["actor_loss"], rtol=2e-6)
    assert "policy_entropy_lower" not in without_entropy[3]
    assert int(without_entropy[0].step) == 1


@pytest.mark.parametrize("benchmark,env_id", [("humanoid", "Humanoid-v4"), ("ant", "Ant-v4")])
def test_v3_common_loop_has_no_entropy_or_guard_and_restores_checkpoint(tmp_path, monkeypatch, benchmark, env_id):
    # Retain 256x3 networks and 16x64 OT; this is an integration check, not a run.
    cfg = config(overrides=[f"benchmark={benchmark}", "alg.batch_size=4", "alg.buffer_size=32",
        "alg.learning_starts=2", "alg.actor.learning_starts=2", "diagnostic_interval=1"])
    assert validate_config(cfg)
    monkeypatch.setattr(algorithm, "idac_action_and_log_density", forbidden_entropy)
    monkeypatch.setattr(algorithm, "conditional_mixture_log_prob", forbidden_entropy)
    monkeypatch.setattr(algorithm, "sampled_soft_update", forbidden_entropy)
    for fn in (OptiQDIME._train, OptiQDIME.update_critic, OptiQDIME.update_actor):
        fn.clear_cache()
    env = gym.make(env_id, max_episode_steps=2)
    model = OptiQDIME("MlpPolicy", env, str(tmp_path), 4, cfg)
    model.set_logger(configure(str(tmp_path / "logs"), ["csv"]))
    try:
        model.learn(total_timesteps=8)
        assert model.backup_mode == "td"
        assert model._n_updates == int(model.policy.actor_state.step) == 6
        assert model.soft_guard_attempts == model.soft_guard_accepts == 0
        assert float(model.ent_coef_state.apply_fn({"params": model.ent_coef_state.params}, 8)) == 0.
        assert model.replay_buffer.timeouts[:8].sum() > 0
        assert not np.asarray(model.replay_buffer._get_samples(np.arange(8)).dones).any()
        for state, prefix in ((model.policy.actor_state, "actor"), (model.policy.qf_state, "critic")):
            restored = serialization.from_bytes(state, (tmp_path / f"{prefix}_state_8.msgpack").read_bytes())
            for a, b in zip(jax.tree_util.tree_leaves(state), jax.tree_util.tree_leaves(restored)):
                np.testing.assert_array_equal(a, b)
        obs, key = jnp.zeros((1, model.observation_space.shape[0])), jax.random.PRNGKey(12)
        restored = serialization.from_bytes(model.policy.actor_state, (tmp_path / "actor_state_8.msgpack").read_bytes())
        np.testing.assert_array_equal(OptiQPolicy.sample_action(restored, obs, key),
                                      OptiQPolicy.sample_action(model.policy.actor_state, obs, key))
        if model.logger.name_to_value:
            model.logger.dump(model.num_timesteps)
        with (tmp_path / "logs/progress.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert rows and "train/policy_entropy_lower" not in rows[-1]
        assert "train/soft_guard_gap_mean" not in rows[-1]
        training_rows = [row for row in rows if row.get("train/actor_updates")]
        assert float(training_rows[-1]["train/actor_updates"]) == 6
        assert all(float(row["train/backup_entropy_term"]) == 0. for row in training_rows)
        assert all(float(row["train/ent_coef"]) == 0. for row in training_rows)
    finally:
        model.get_env().close()
        model.logger.close()


def test_v3_launcher_verifies_the_actual_overrides():
    from scripts.verify_v3 import verify
    assert verify(["benchmark=ant", "seed=3"]).env_name == "Ant-v4"
    with pytest.raises(ValueError, match="no policy entropy evaluation"):
        verify(["alg.actor.entropy_diagnostics=true"])
    with pytest.raises(ValueError, match="full OT NLL"):
        verify(["alg.actor.distillation_loss=pointwise_mse"])
