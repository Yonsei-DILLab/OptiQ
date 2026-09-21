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
from scripts.mujoco_beta_sweep import tasks, command, BETAS, DEFAULT_SEEDS

ROOT = Path(__file__).resolve().parents[1]


def config(overrides=()):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(config_name="optiq_dime_mujoco", overrides=list(overrides))


def test_task_table_and_gym_configs():
    table = tasks()
    assert DEFAULT_SEEDS == (1, 2, 3, 4)
    assert len(table) == len(set(table)) == 80
    for environment in ("ant", "humanoid"):
        for sampling in ("stratified", "exact"):
            for beta in BETAS:
                assert [task.seed for task in table if
                        (task.environment, task.sampling, task.beta) ==
                        (environment, sampling, beta)] == list(DEFAULT_SEEDS)
    for task in table:
        cfg = config(command(task, [])[3:])
        assert validate_config(cfg)
        assert cfg.env_name == {"ant": "Ant-v4", "humanoid": "Humanoid-v4"}[task.environment]
        assert cfg.total_steps == 1000000
        assert cfg.seed == task.seed
        assert cfg.alg.actor.proposal_sampling_mode == task.sampling
        assert cfg.alg.actor.density_beta == task.beta
        assert cfg.alg.critic.v_min == -1600 and cfg.alg.critic.v_max == 1600
        assert cfg.alg.actor.proposal_std == 0.1 and cfg.alg.actor.proposal_clip == 0.15


@pytest.mark.parametrize("environment", ["ant", "humanoid"])
def test_all_reference_hyperparameters_match_except_approved_critic_support(environment):
    import json
    from omegaconf import OmegaConf
    reference = json.loads((ROOT / "tests/data/supplied_dog_reference.json").read_text())
    current = OmegaConf.to_container(config([f"mujoco_env={environment}"]), resolve=True)

    def compare(expected, actual, prefix=""):
        for key, value in expected.items():
            path = f"{prefix}.{key}" if prefix else key
            if path in {"alg.critic.v_min", "alg.critic.v_max"}:
                continue
            if isinstance(value, dict):
                compare(value, actual[key], path)
            else:
                assert actual[key] == value, f"Reference mismatch: {path}"

    compare(reference, current)
    assert current['log_interval'] == 1
    assert current['progress_bar'] is True
    assert current['alg']['actor']['proposal_sampling_mode'] == 'stratified'
    assert current['alg']['actor']['density_beta'] == 1
    assert current['alg']['actor']['adaptive_density_beta'] is False


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


def test_evaluation_matches_reference_rng_schedule_and_returns(model, tmp_path):
    from stable_baselines3.common.env_util import make_vec_env
    from models.actor_critic_evaluation_callback import EvalCallback
    cfg = config(["num_eval_episodes=1", "eval_interval=4"])
    envs = [make_vec_env(lambda: gym.make("Ant-v4", max_episode_steps=2)) for _ in range(2)]
    original = EvalCallback(envs[0], jax_random_key_for_seeds=cfg.seed,
                            n_eval_episodes=1, eval_freq=4, deterministic=False,
                            log_path=str(tmp_path / "reference"))
    actual = MujocoEvalCallback(envs[1], cfg, tmp_path / "actual")
    try:
        np.testing.assert_array_equal(original.seed_list, actual.seed_list)
        original.init_callback(model)
        actual.init_callback(model)
        initial_key, initial_noise = model.policy.key, model.policy.noise_key
        end_keys = []
        for callback in [original, actual]:
            model.policy.key, model.policy.noise_key = initial_key, initial_noise
            for step in range(1, 7):
                callback.n_calls = callback.num_timesteps = step
                callback._on_step()
            callback._on_training_end()
            end_keys.append((jax.random.key_data(model.policy.key),
                             jax.random.key_data(model.policy.noise_key)))
        np.testing.assert_array_equal(end_keys[0], end_keys[1])
        assert not np.array_equal(end_keys[1][0], jax.random.key_data(initial_key))
        np.testing.assert_array_equal(original.evaluations_results, actual.evaluations_results)
        assert actual.evaluations_timesteps == [1, 4]  # no extra off-schedule final evaluation
        results = np.load(tmp_path / "actual" / "evaluations.npz")
        np.testing.assert_array_equal(results["timesteps"], [1, 4])
        np.testing.assert_array_equal(results["results"], original.evaluations_results)
    finally:
        for env in envs:
            env.close()


@pytest.mark.parametrize("mode", ["stratified", "exact"])
def test_actor_update_matches_critic_dime_reference(model, mode):
    import subprocess
    import types
    reference = types.ModuleType("optiq_dime._reference_algorithm")
    reference.__package__ = "optiq_dime"
    source = subprocess.check_output(
        ["git", "show", "8b8fee13b2cbc4d90183d7b803fce033eadf0a19:optiq_dime/algorithm.py"], text=True)
    exec(compile(source, "critic-dime/algorithm.py", "exec"), reference.__dict__)
    a = model.cfg.alg.actor
    args = (model.policy.actor_state, model.policy.qf_state, jnp.zeros((4, 27)),
            jax.random.PRNGKey(123), jnp.linspace(-1600, 1600, 101),
            a.num_policy_samples, a.proposals_per_policy_sample, mode, a.proposal_std,
            a.proposal_clip, a.include_anchor, True, 0.5, False, 16.0, 257,
            a.temperature, a.sinkhorn_epsilon, a.sinkhorn_iterations, a.source_q_eval,
            a.transport_target_mode)
    old_state, old_loss, old_key, _ = reference.OptiQDIME.update_actor(*args)
    new_state, new_loss, new_key, _ = OptiQDIME.update_actor(*args)
    np.testing.assert_array_equal(old_key, new_key)
    np.testing.assert_allclose(old_loss, new_loss, rtol=1e-6, atol=1e-7)
    for old, new in zip(jax.tree_util.tree_leaves(old_state), jax.tree_util.tree_leaves(new_state)):
        np.testing.assert_allclose(old, new, rtol=1e-6, atol=1e-7)
