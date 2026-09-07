from pathlib import Path

import gymnasium as gym
from hydra import compose, initialize_config_dir
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from stable_baselines3.common.buffers import ReplayBuffer

from optiq_dime import OptiQDIME
from optiq_dime.evaluation import MujocoEvalCallback
from optiq_dime.transport import (
    sample_truncated_gaussian, sample_truncated_gaussian_mixture,
    truncated_mixture_log_density,
)
from run_optiq_dime import validate_config
from scripts.mujoco_beta_sweep import tasks, command, BETAS

ROOT = Path(__file__).resolve().parents[1]


def config(overrides=()):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(config_name="optiq_dime_mujoco", overrides=list(overrides))


def test_task_table_and_gym_configs():
    table = tasks([1, 2, 3, 4, 5])
    assert len(table) == len(set(table)) == 100
    for task in table:
        cfg = config(command(task, [])[3:])
        assert validate_config(cfg)
        assert cfg.env_name == {"ant": "Ant-v4", "humanoid": "Humanoid-v4"}[task.environment]
        assert cfg.total_steps == {"ant": 3000000, "humanoid": 5000000}[task.environment]
        assert cfg.alg.actor.density_beta == task.beta
        assert cfg.alg.critic.v_min == -1600 and cfg.alg.critic.v_max == 1600
        assert cfg.alg.actor.proposal_std == 0.1 and cfg.alg.actor.proposal_clip == 0.15


@pytest.mark.parametrize("anchor", [False, True])
def test_exact_mixture_component_allocation_and_support(anchor):
    centers = jnp.broadcast_to(jnp.array([[[-0.75], [-0.25], [0.25], [0.75]]]), (256, 4, 1))
    key = jax.random.PRNGKey(321)
    samples, indices = sample_truncated_gaussian_mixture(
        key, centers, 5, 0.1, 0.15, anchor, return_component_indices=True)
    bare = sample_truncated_gaussian_mixture(key, centers, 5, 0.1, 0.15, anchor)
    np.testing.assert_array_equal(samples, bare)
    generating_centers = np.take_along_axis(np.asarray(centers[..., 0]), np.asarray(indices).reshape(256, -1), axis=1)
    assert np.max(np.abs(np.asarray(samples).reshape(256, -1) - generating_centers)) <= 0.150001
    random_indices = np.asarray(indices[..., int(anchor):])
    fractions = np.bincount(random_indices.ravel(), minlength=4) / random_indices.size
    np.testing.assert_allclose(fractions, 0.25, atol=0.025)
    # IID component allocation is not a fixed repeat count per component.
    counts = (random_indices.reshape(256, -1) == 0).sum(axis=1)
    assert counts.std() > 0.8
    if anchor:
        np.testing.assert_array_equal(samples[:, :, 0], centers)
        np.testing.assert_array_equal(indices[0, :, 0], np.arange(4))
    logq = truncated_mixture_log_density(samples.reshape(256, -1, 1), centers, 0.1, 0.15)
    assert np.isfinite(logq).all()


@pytest.mark.parametrize("sampler", [sample_truncated_gaussian, sample_truncated_gaussian_mixture])
def test_sampling_and_density_describe_same_measure(sampler):
    centers = jnp.array([[[-0.5], [0.5]]])
    samples = sampler(jax.random.PRNGKey(19), centers, 20000, 0.8, 1.5).reshape(1, -1, 1)
    logq = truncated_mixture_log_density(samples, centers, 0.8, 1.5)
    # Inverse-density weighting should recover a uniform target on [-1, 1].
    weights = np.exp(-np.asarray(logq)[0])
    y = np.asarray(samples)[0, :, 0]
    assert abs(np.average(y, weights=weights)) < 0.025
    assert abs(np.average(y**2, weights=weights) - 1/3) < 0.025
    grid = jnp.linspace(-1, 1, 4001).reshape(1, -1, 1)
    density = np.exp(np.asarray(truncated_mixture_log_density(grid, centers, 0.8, 1.5))[0])
    assert abs(np.trapz(density, np.asarray(grid).ravel()) - 1) < 0.002


@pytest.mark.parametrize("env_id,obs_dim,act_dim,limit", [
    ("Ant-v4", 27, 8, 1.0), ("Humanoid-v4", 376, 17, 0.4),
])
def test_environment_action_bounds_and_timeout_bootstrap(env_id, obs_dim, act_dim, limit):
    env = gym.make(env_id, max_episode_steps=1)
    try:
        obs, _ = env.reset(seed=10)
        assert obs.shape == (obs_dim,)
        assert env.action_space.shape == (act_dim,)
        np.testing.assert_allclose(env.action_space.high, limit)
        next_obs, reward, terminated, truncated, _ = env.step(np.zeros(act_dim))
        assert not terminated and truncated
        replay = ReplayBuffer(4, env.observation_space, env.action_space, device="cpu")
        replay.add(obs[None], next_obs[None], np.zeros((1, act_dim)), np.array([reward]),
                   np.array([True]), [{"TimeLimit.truncated": True}])
        assert replay.sample(1).dones.item() == 0  # continue Bellman bootstrap at timeout
    finally:
        env.close()


@pytest.fixture(scope="module")
def model():
    from stable_baselines3.common.logger import configure
    cfg = config(["alg.critic.hs=[32,32]", "alg.actor.hidden_dims=[32,32]",
                  "alg.buffer_size=32", "alg.batch_size=4"])
    env = gym.make("Ant-v4")
    model = OptiQDIME("MlpPolicy", env, None, 1, cfg)
    model.set_logger(configure(None, []))
    yield model
    model.get_env().close()


@pytest.mark.parametrize("mode", ["stratified", "exact"])
@pytest.mark.parametrize("beta", BETAS)
def test_fixed_beta_actor_update_is_finite(model, mode, beta):
    a = model.cfg.alg.actor
    before = jax.tree_util.tree_leaves(model.policy.actor_state.params)
    state, loss, _, metrics = OptiQDIME.update_actor(
        model.policy.actor_state, model.policy.qf_state, jnp.zeros((4, 27)),
        jax.random.PRNGKey(7), jnp.linspace(-1600, 1600, 101),
        a.num_policy_samples, a.proposals_per_policy_sample, mode, a.proposal_std,
        a.proposal_clip, a.include_anchor, True, beta, False, 16.0, 257,
        a.temperature, a.sinkhorn_epsilon, a.sinkhorn_iterations, a.source_q_eval, a.transport_target_mode,
    )
    assert np.isfinite(loss)
    assert all(np.isfinite(v).all() for v in metrics.values())
    assert float(metrics["density_beta_mean"]) == pytest.approx(beta)
    assert 1 <= float(metrics["source_ess_absolute"]) <= 80.001
    assert any(not np.array_equal(x, y) for x, y in zip(before, jax.tree_util.tree_leaves(state.params)))


def test_evaluation_preserves_training_rng_and_saves_results(model, tmp_path):
    from stable_baselines3.common.env_util import make_vec_env
    cfg = config(["num_eval_episodes=1"])
    env = make_vec_env(lambda: gym.make("Ant-v4", max_episode_steps=2))
    callback = MujocoEvalCallback(env, cfg, tmp_path)
    callback.init_callback(model)
    callback.num_timesteps = 7
    key_before = jax.random.key_data(model.policy.key)
    noise_before = jax.random.key_data(model.policy.noise_key)
    try:
        callback.evaluate()
        np.testing.assert_array_equal(jax.random.key_data(model.policy.key), key_before)
        np.testing.assert_array_equal(jax.random.key_data(model.policy.noise_key), noise_before)
        results = np.load(tmp_path / "evaluations.npz")
        np.testing.assert_array_equal(results["timesteps"], [7])
        assert results["results"].shape == (1, 1)
        assert results["ep_lengths"][0, 0] == 2
    finally:
        env.close()
