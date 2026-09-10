"""Scalar output, TD target/gradient and real MuJoCo training regressions."""
from pathlib import Path

from flax import serialization
from flax.training.train_state import TrainState
import gymnasium as gym
from hydra import compose, initialize_config_dir
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import optax
import pytest
from stable_baselines3.common.logger import configure

from common.type_aliases import RLTrainState
from optiq_dime import OptiQDIME
from optiq_dime.critic_utils import critic_expectation
from run_optiq_dime import validate_config

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ['hopper', 'walker2d', 'halfcheetah', 'ant', 'humanoid']


def config(name='mujoco_setting', overrides=()):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / 'configs')):
        return compose(config_name=name, overrides=list(overrides))


@pytest.mark.parametrize('benchmark', BENCHMARKS)
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_only_requested_scalar_and_anchor_defaults_change(benchmark, seed):
    old = config('optiq_dime_no_anchor', [f'benchmark={benchmark}', f'seed={seed}'])
    new = config(overrides=[f'benchmark={benchmark}', f'seed={seed}'])
    assert validate_config(new)
    a, b = [OmegaConf.to_container(c.alg, resolve=True) for c in (old, new)]
    assert b['critic']['type'] == 'scalar'
    assert b['critic']['activation'] == 'gelu'
    assert b['critic']['crossq_style'] is False
    assert b['critic']['hs'] == [256, 256, 256]
    assert b['critic']['n_atoms'] == 1
    assert b['critic']['entr_coeff'] == 0
    del b['critic']['type']
    del b['critic']['crossq_style']
    for key in ['hs', 'n_atoms', 'entr_coeff', 'activation']:
        b['critic'][key] = a['critic'][key]
    assert b['optimizer']['bn'] is False and b['utd'] == 1
    assert b['optimizer']['critic_b1'] == b['optimizer']['actor_b1'] == 0.9
    assert b['tau'] == 0.005
    b['optimizer']['bn'] = a['optimizer']['bn']
    b['optimizer']['critic_b1'] = a['optimizer']['critic_b1']
    b['utd'] = a['utd']
    b['tau'] = a['tau']
    assert b['actor']['density_correction_beta'] == b['actor']['density_beta'] == 1.0
    for key in ['density_correction_beta', 'density_beta']:
        b['actor'][key] = a['actor'][key]
    assert b['actor']['include_anchor'] is True
    assert b['actor']['num_policy_samples'] == 16
    assert b['actor']['proposals_per_policy_sample'] == 5
    for key in ['include_anchor', 'proposals_per_policy_sample']:
        b['actor'][key] = a['actor'][key]
    assert a == b  # All other actor and optimizer settings are unchanged.
    for key in ['env_name', 'seed', 'total_steps', 'eval_interval', 'num_eval_episodes',
                'eval_at_start', 'stochastic_eval', 'diagnostic_interval', 'checkpoint_interval']:
        assert old[key] == new[key]
    assert new.alg.actor.include_anchor is True
    assert new.alg.actor.density_beta == 1.0
    assert not new.alg.actor.adaptive_density_beta


def test_scalar_alias_and_validation():
    cfg = config(overrides=['alg.actor.density_correction_beta=0.001'])
    assert cfg.alg.actor.density_beta == 0.001
    assert cfg.env_name == 'Humanoid-v4'
    for override, match in [('alg.critic.n_atoms=101', 'Scalar critics require'),
                            ('alg.critic.entr_coeff=0.005', 'entr_coeff=0'),
                            ('alg.critic.n_atoms=0', 'positive')]:
        with pytest.raises(ValueError, match=match):
            validate_config(config(overrides=[override]))


def test_scalar_q_is_not_multiplied_by_support_endpoint():
    raw = jnp.array([[[-2.], [4.]], [[3.], [-7.]]])
    np.testing.assert_array_equal(critic_expectation(raw, jnp.array([-3600.])), raw[..., 0])
    np.testing.assert_allclose(critic_expectation(jnp.array([[0.2, 0.3, 0.5]]),
                                                jnp.array([-2., 0., 2.])), [0.6])


def constant_critic(variables, observations, actions, rngs=None, mutable=False, train=False):
    values = jnp.broadcast_to(variables['params']['q'][:, None, None],
                              (2, observations.shape[0], 1))
    if mutable:
        return values, {'batch_stats': variables['batch_stats']}
    return values


def zero_actor(variables, observations, latents):
    return jnp.zeros_like(latents)


@pytest.mark.parametrize('crossq', [True, False])
@pytest.mark.parametrize('terminal_reward', [2., 10000.])
def test_scalar_td_min_backup_terminal_mask_stop_gradient_and_no_clipping(crossq, terminal_reward):
    actor = TrainState.create(apply_fn=zero_actor,
        params={'Dense_0': {'bias': jnp.zeros(1)}}, tx=optax.sgd(0.01))
    critic = RLTrainState.create(apply_fn=constant_critic,
        params={'q': jnp.array([2., 6.])}, batch_stats={},
        target_params={'q': jnp.array([10., 14.])}, target_batch_stats={}, tx=optax.sgd(0.01))
    obs, acts = jnp.zeros((2, 3)), jnp.zeros((2, 1))
    new, metrics, _ = OptiQDIME.update_critic(crossq, False, 0.9, actor, critic,
        obs, acts, obs, jnp.array([1., terminal_reward]), jnp.array([0., 1.]),
        1, jnp.array([-1.]), -1., 1., 0., 0.2, 0.5, jax.random.PRNGKey(5))
    target = np.array([1. + .9 * (2. if crossq else 10.), terminal_reward])
    current = np.array([2., 6.])
    expected_loss = np.square(current[:, None] - target[None, :]).mean(axis=1).sum()
    np.testing.assert_allclose(metrics['critic_loss'], expected_loss, rtol=1e-6)
    np.testing.assert_allclose(metrics['next_q_values'], target.mean(), rtol=1e-6)
    expected_params = current - .01 * 2 * (current - target.mean())
    np.testing.assert_allclose(new.params['q'], expected_params, rtol=1e-6)
    assert metrics['entrQ_1'] == metrics['entrQ_2'] == 0
    updated_target = OptiQDIME.soft_update(0.005, new)
    np.testing.assert_allclose(updated_target.target_params['q'],
        .995 * np.array([10., 14.]) + .005 * expected_params, rtol=1e-6)


@pytest.mark.parametrize('benchmark', BENCHMARKS)
def test_real_environment_scalar_actor_critic_training_and_checkpoint(benchmark):
    cfg = config(overrides=[f'benchmark={benchmark}', 'alg.batch_size=4',
                           'alg.buffer_size=32', 'alg.learning_starts=2',
                           'alg.actor.learning_starts=2', 'diagnostic_interval=1'])
    env = gym.make(cfg.env_name)
    model = OptiQDIME('MlpPolicy', env, None, 1, cfg)
    model.set_logger(configure(None, []))
    actor_before = jax.tree_util.tree_leaves(model.policy.actor_state.params)
    critic_before = jax.tree_util.tree_leaves(model.policy.qf_state.params)
    try:
        assert not model.crossq_style
        assert model.tau == 0.005
        assert list(model.policy.qf.net_arch) == [256, 256, 256]
        model.learn(total_timesteps=8)
        assert model._n_updates == 6  # (8 - 2) x one update.
        assert not model.policy.qf_state.batch_stats
        assert not model.policy.qf_state.target_batch_stats
        assert any(not np.array_equal(a, b) for a, b in zip(
            jax.tree_util.tree_leaves(model.policy.qf_state.params),
            jax.tree_util.tree_leaves(model.policy.qf_state.target_params)))
        for before, state in [(actor_before, model.policy.actor_state),
                              (critic_before, model.policy.qf_state)]:
            after = jax.tree_util.tree_leaves(state.params)
            assert all(np.isfinite(a).all() for a in after)
            assert any(not np.array_equal(a, b) for a, b in zip(before, after))
        obs = jnp.zeros((2, *env.observation_space.shape))
        actions = jnp.zeros((2, *env.action_space.shape))
        q = model.policy.predict_critic(obs, actions)
        assert q.shape == (2, 2, 1)
        assert np.isfinite(q).all()
        a = cfg.alg.actor
        _, loss, _, metrics = OptiQDIME.update_actor(model.policy.actor_state,
            model.policy.qf_state, obs, jax.random.PRNGKey(9), jnp.array([-3600.]),
            a.num_policy_samples, a.proposals_per_policy_sample, a.proposal_sampling_mode,
            a.proposal_std, a.proposal_clip, a.include_anchor, a.density_correction,
            a.density_beta, a.adaptive_density_beta, a.minimum_source_ess, a.density_beta_grid_size,
            a.temperature, a.sinkhorn_epsilon, a.sinkhorn_iterations,
            a.source_q_eval, a.transport_target_mode)
        assert np.isfinite(loss) and all(np.isfinite(v).all() for v in metrics.values())
        assert float(metrics['density_beta_mean']) == 1.0
        assert 1 <= metrics['source_ess_absolute'] <= 80.001
        # Raw Q diagnostics must not silently multiply scalar outputs by v_min.
        _, loss2, _, metrics2 = OptiQDIME.update_actor(model.policy.actor_state,
            model.policy.qf_state, obs, jax.random.PRNGKey(9), jnp.array([1.]),
            a.num_policy_samples, a.proposals_per_policy_sample, a.proposal_sampling_mode,
            a.proposal_std, a.proposal_clip, a.include_anchor, a.density_correction,
            a.density_beta, a.adaptive_density_beta, a.minimum_source_ess, a.density_beta_grid_size,
            a.temperature, a.sinkhorn_epsilon, a.sinkhorn_iterations,
            a.source_q_eval, a.transport_target_mode)
        np.testing.assert_array_equal(loss, loss2)
        np.testing.assert_array_equal(metrics['source_q_mean'], metrics2['source_q_mean'])
        restored = serialization.from_bytes(model.policy.qf_state,
                                            serialization.to_bytes(model.policy.qf_state))
        for x, y in zip(jax.tree_util.tree_leaves(restored.params),
                        jax.tree_util.tree_leaves(model.policy.qf_state.params)):
            np.testing.assert_array_equal(x, y)
    finally:
        model.get_env().close()
