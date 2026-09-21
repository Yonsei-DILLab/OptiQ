"""Actual policy, density, OT/backup, rollback and checkpoint consistency."""
from pathlib import Path
from flax import serialization
from hydra import compose, initialize_config_dir
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from scipy.special import logsumexp
from scipy.stats import norm
from stable_baselines3.common.logger import configure

from optiq_dime import OptiQDIME
from optiq_dime.latent import FiniteMixtureTrainState, finite_latent_codes, sample_latents
from optiq_dime.policy import OptiQPolicy
from optiq_dime.semi_implicit import finite_policy_action_and_log_density
from optiq_dime.soft_improvement import entropy_bracket_sample, candidate_gap_samples
from run_optiq_dime import validate_config
from test_semi_implicit import critic_state


def cfg(overrides=()):
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1]/"configs"), version_base=None):
        return compose(config_name="mujoco_v2_finite", overrides=list(overrides))


def analytic_actor(variables, observations, z):
    p = variables["params"]
    return z+p["mu"]["bias"]+observations[:, :1], jnp.full_like(z, p["ls"])


def analytic_state():
    return FiniteMixtureTrainState.create(apply_fn=analytic_actor, tx=optax.adam(.001),
        params={"mu": {"bias": jnp.zeros(2)}, "ls": jnp.array(np.log(.2))},
        latent_components=16, latent_codebook_seed=20260911)


def test_real_policy_density_and_rollout_match_all_components_scipy():
    actor = analytic_state()
    obs = jnp.linspace(-.1, .1, 128)[:, None]
    key = jax.random.PRNGKey(42)
    a, u, lp = finite_policy_action_and_log_density(actor, obs, key, 16)
    codes = np.asarray(finite_latent_codes(actor, 2))
    centers = codes[None, :, :]+np.asarray(obs)[:, None, :]
    normal_lp = norm.logpdf(np.asarray(u)[:, None, :], centers, .2).sum(axis=-1)
    jac = (2*(np.log(2)-np.asarray(u)-np.logaddexp(0, -2*np.asarray(u)))).sum(axis=-1)
    expected = logsumexp(normal_lp, axis=-1)-np.log(16)-jac
    np.testing.assert_allclose(lp, expected, atol=1.e-5)
    np.testing.assert_allclose(a, OptiQPolicy.sample_action(actor, obs, key), atol=2.e-6)
    _, _, lower, upper = entropy_bracket_sample(actor, obs, key, 16)
    np.testing.assert_array_equal(lower, upper)
    np.testing.assert_allclose(lower, -expected, atol=1.e-5)
    # No entropy-bracket penalty for an unchanged policy, even when separated.
    gap = candidate_gap_samples(actor, actor, critic_state(), obs, key, .1, jnp.zeros(1), 16, 8)
    np.testing.assert_array_equal(gap, np.zeros(gap.shape))


def test_uniform_latent_prior_and_serialization_preserve_actual_policy():
    actor = analytic_state()
    codes = np.asarray(finite_latent_codes(actor, 2))
    draws = np.asarray(sample_latents(actor, jax.random.PRNGKey(3), (32768, 2)))
    distances = ((draws[:, None, :]-codes[None, :, :])**2).sum(axis=-1)
    assert np.max(distances.min(axis=-1)) == 0.
    counts = np.bincount(distances.argmin(axis=-1), minlength=16)
    assert np.max(np.abs(counts-2048)) < 200
    restored = serialization.from_bytes(actor, serialization.to_bytes(actor))
    assert restored.latent_components == 16 and restored.latent_codebook_seed == 20260911
    obs, key = jnp.zeros((64, 1)), jax.random.PRNGKey(8)
    np.testing.assert_array_equal(OptiQPolicy.sample_action(actor, obs, key),
                                  OptiQPolicy.sample_action(restored, obs, key))


@pytest.mark.parametrize("override", ["alg.actor.entropy_samples=8", "alg.actor.num_policy_samples=8",
    "alg.actor.soft_guard.components=8", "alg.actor.latent_components=1",
    "alg.actor.latent_codebook_seed=-1", "alg.actor.mean_latent_skip_scale=-1"])
def test_partial_actual_mixture_or_invalid_prior_is_rejected(override):
    with pytest.raises(ValueError):
        validate_config(cfg([override]))


def test_finite_ot_gradient_boundaries_and_all_student_components():
    config = cfg(["alg.actor.hidden_dims=[16,16]", "alg.critic.hs=[16,16]"])
    policy = OptiQPolicy(gym.spaces.Box(-1., 1., (3,), dtype=np.float32),
                        gym.spaces.Box(-1., 1., (2,), dtype=np.float32), config)
    policy.build(jax.random.PRNGKey(0), lambda _: .0003, .0003)
    actor, critic = policy.actor_state, critic_state()
    observations = jnp.zeros((4, 3))

    def update(q):
        return OptiQDIME.update_actor(actor, q, observations, jax.random.PRNGKey(9),
            jnp.zeros(1), 16, 4, "exact", .05, .5, False, True, 1., False, 16., 257,
            .1, .25, 100, "mean", "argmax", True, False, "conditional_ot_nll", "conditional_mixture")

    new, loss, _, metrics = update(critic)
    assert isinstance(new, FiniteMixtureTrainState) and np.isfinite(loss)
    assert metrics["policy_density_exact"] == 1.
    assert 1. <= metrics["source_ess_absolute"] <= 64.001
    assert metrics["actor_latent_mean_variance_fraction"] > .8
    assert int(new.step) == 1
    q_grad = jax.grad(lambda p: update(critic.replace(params=p))[1])(critic.params)
    assert all(np.count_nonzero(x) == 0 for x in jax.tree_util.tree_leaves(q_grad))


def test_finite_common_humanoid_loop_backup_guard_timeout_checkpoint(tmp_path):
    config = cfg(["alg.batch_size=4", "alg.buffer_size=32", "alg.learning_starts=2",
        "alg.actor.learning_starts=2", "alg.actor.hidden_dims=[32,32]", "alg.critic.hs=[32,32]",
        "alg.actor.soft_guard.batch_size=4", "alg.actor.soft_guard.draws=4", "diagnostic_interval=1"])
    assert validate_config(config)
    model = OptiQDIME("MlpPolicy", gym.make("Humanoid-v4", max_episode_steps=2), str(tmp_path), 4, config)
    model.set_logger(configure(str(tmp_path/"logs"), ["csv"]))
    try:
        model.learn(total_timesteps=8)
        assert model._n_updates == 6 and model.soft_guard_attempts == 6
        assert int(model.policy.actor_state.step) == model.soft_guard_accepts
        assert model.replay_buffer.timeouts[:8].sum() > 0
        assert np.count_nonzero(model.replay_buffer._get_samples(np.arange(8)).dones.numpy()) == 0
        checkpoint = tmp_path/"actor_state_8.msgpack"
        restored = serialization.from_bytes(model.policy.actor_state, checkpoint.read_bytes())
        assert isinstance(restored, FiniteMixtureTrainState)
        obs, key = jnp.zeros((1, 376)), jax.random.PRNGKey(5)
        np.testing.assert_array_equal(OptiQPolicy.sample_action(restored, obs, key),
                                      OptiQPolicy.sample_action(model.policy.actor_state, obs, key))
        for actual, target in zip(jax.tree_util.tree_leaves(model.policy.actor_state.params),
                                 jax.tree_util.tree_leaves(model.policy.target_actor_state.params)):
            np.testing.assert_array_equal(actual, target)
    finally:
        model.get_env().close()
        model.logger.close()
