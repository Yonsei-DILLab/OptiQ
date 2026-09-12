"""Dual evaluation semantics and actual short Hopper training integration."""
import copy
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from omegaconf import OmegaConf
from stable_baselines3.common.logger import configure

from scripts.verify_v4 import verify
from run_optiq_dime import create_algorithm
from optiq_dime.policy import OptiQPolicy
from test_semi_implicit import actor_state


def test_defaults_and_two_actions():
    cfg = verify()
    assert list(cfg.alg.actor.hidden_dims) == list(cfg.alg.critic.hs) == [256, 256]
    actor = actor_state()
    obs, key = jnp.ones((3, 3)), jax.random.PRNGKey(37)
    # Match fused/JIT arithmetic on both sides instead of relaxing precision.
    @jax.jit
    def expected(state, obs, key):
        latent_key, _ = jax.random.split(key)
        z = jax.random.normal(latent_key, (3, 2))
        mu, _ = state.apply_fn({"params": state.params}, obs, z)
        return jnp.tanh(mu)
    random_mu = OptiQPolicy.sample_action(actor, obs, key, False, False)
    np.testing.assert_allclose(random_mu, expected(actor, obs, key), atol=1e-7)
    zero = OptiQPolicy.sample_action(actor, obs, key, True, False)
    other = OptiQPolicy.sample_action(actor, obs, jax.random.PRNGKey(38), True, False)
    np.testing.assert_array_equal(zero, other)
    assert not np.array_equal(zero, random_mu)
    # Conditional std affects collection but neither mu-only evaluation.
    params = dict(actor.params)
    params['log_std'] = dict(params['log_std'])
    params['log_std']['bias'] = params['log_std']['bias'] + 1
    changed = actor.replace(params=params)
    for deterministic in (False, True):
        np.testing.assert_array_equal(
            OptiQPolicy.sample_action(actor, obs, key, deterministic, False),
            OptiQPolicy.sample_action(changed, obs, key, deterministic, False))
    assert not np.array_equal(OptiQPolicy.sample_action(actor, obs, key),
                              OptiQPolicy.sample_action(changed, obs, key))


@pytest.mark.parametrize("temperature", [.1, .25])
@pytest.mark.parametrize("seed", range(4))
def test_behavior010_preserves_v4_algorithm_and_evaluation_defaults(temperature, seed):
    overrides = ['benchmark=ant', f'seed={seed}', f'alg.actor.temperature={temperature}']
    baseline = OmegaConf.to_container(verify(overrides), resolve=True)
    configured = OmegaConf.to_container(verify(overrides, 'mujoco_v4_behavior010'), resolve=True)
    alias = OmegaConf.to_container(verify(overrides, 'v4/behavior010'), resolve=True)
    expected = copy.deepcopy(baseline)
    expected['alg']['behavior_uniform_probability'] = .1
    for key in ('wandb', 'run_name', 'output_root'):
        expected[key] = configured[key]
    assert configured == expected == alias
    assert baseline['alg']['behavior_uniform_probability'] == 0.
    assert configured['wandb']['project'] == 'v4-test'
    assert 'behavior010' in configured['wandb']['group']
    assert 'behavior010' in configured['run_name']


@pytest.mark.parametrize("config_name", ['mujoco_v4', 'mujoco_v4_behavior010'])
def test_real_hopper_dual_eval_training_and_failure_restore(tmp_path, monkeypatch, config_name):
    # Force replacement in the opt-in case to exercise real env.step/replay
    # behavior in a short integration test; rate=.1 is covered separately.
    probability = 1.0 if config_name == 'mujoco_v4_behavior010' else 0.0
    cfg = verify(['benchmark=hopper', f'output_root={tmp_path}',
        'alg.batch_size=4', 'alg.buffer_size=32', 'alg.learning_starts=2',
        'alg.actor.learning_starts=2', 'num_eval_episodes=2', 'eval_interval=4',
        'diagnostic_interval=4', 'checkpoint_interval=4',
        f'alg.behavior_uniform_probability={probability}'], config_name)
    model, callbacks = create_algorithm(cfg)
    model.set_logger(configure(str(tmp_path/'test_logs'), ['csv']))
    cb = callbacks.callbacks[0]
    # Short validation episodes, not a benchmark score.
    cb.eval_env.envs[0].env._max_episode_steps = 3
    executed = []
    training_env = model.get_env().envs[0]
    original_step = training_env.step
    def record_step(action):
        executed.append(np.array(action, copy=True))
        return original_step(action)
    monkeypatch.setattr(training_env, 'step', record_step)
    try:
        model.learn(total_timesteps=8, callback=callbacks)
        assert model._n_updates == int(model.policy.actor_state.step) == 6
        assert model.behavior_uniform_count == model.behavior_action_count == int(6 * probability)
        replay = model.replay_buffer.actions[:8, 0]
        np.testing.assert_allclose(model.policy.unscale_action(replay), executed, atol=1e-6)
        assert cb.evaluations_timesteps == [1, 4, 8]
        for mode in cb.MODES:
            saved = np.load(cb.directory/f'evaluations_{mode}.npz')
            assert saved['results'].shape == (3, 2)
            np.testing.assert_array_equal(saved['timesteps'], [1, 4, 8])
        np.testing.assert_array_equal(cb.histories['zero_z']['env_seeds'],
                                      cb.histories['stochastic_z']['env_seeds'])
        key, noise = model.policy.key, model.policy.noise_key
        behavior_rng = copy.deepcopy(model.behavior_rng.bit_generator.state)
        # Both v4 evaluation modes must bypass the collector entirely.
        def collection_fail(*args, **kwargs):
            raise AssertionError('Evaluation called the collection exploration hook')
        monkeypatch.setattr(model, '_sample_action', collection_fail)
        cb.n_calls = 12
        cb.num_timesteps = 12
        cb._on_step()
        np.testing.assert_array_equal(model.policy.key, key)
        np.testing.assert_array_equal(model.policy.noise_key, noise)
        assert not model.policy.evaluation_mu_only
        assert model.behavior_rng.bit_generator.state == behavior_rng
        def fail(*args, **kwargs):
            raise RuntimeError('intentional evaluation failure')
        monkeypatch.setattr('optiq_dime.dual_evaluation.evaluate_policy', fail)
        with pytest.raises(RuntimeError, match='intentional'):
            cb._on_step()
        np.testing.assert_array_equal(model.policy.key, key)
        np.testing.assert_array_equal(model.policy.noise_key, noise)
        assert not model.policy.evaluation_mu_only
        assert model.behavior_rng.bit_generator.state == behavior_rng
        content = (tmp_path/'test_logs/progress.csv').read_text()
        assert 'eval/zero_z/mean_reward' in content
        assert 'eval/stochastic_z/mean_reward' in content
    finally:
        cb.eval_env.close()
        model.get_env().close()
        model.logger.close()
