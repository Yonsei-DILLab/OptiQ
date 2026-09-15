"""Behavior selection, warmup preservation, and actual training/evaluation isolation."""
import copy
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import optax
import pytest
from stable_baselines3.common.logger import configure
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

from common.type_aliases import RLTrainState
from optiq_dime.algorithm import OptiQDIME
from optiq_dime.behavior import select_best_of_k
from optiq_dime.policy import OptiQPolicy
from run_optiq_dime import create_algorithm, validate_config
from scripts.verify_v5 import verify
from test_semi_implicit import actor_state


def directional_critic(variables, observations, actions, rngs=None, train=False):
    value = variables['params']['sign'] * observations[:, 0] * actions[:, 0]
    return jnp.stack((value, 10.0 - 3.0 * value))[:, :, None]


def directional_state():
    return RLTrainState.create(apply_fn=directional_critic, params={'sign': jnp.array(1.)},
        batch_stats={}, target_params={'sign': jnp.array(-1.)}, target_batch_stats={},
        tx=optax.sgd(.01))


def test_eight_gaussian_candidates_ranked_by_live_twin_min_per_state():
    actor, critic = actor_state(), directional_state()
    obs = jnp.array([[1., 0., 0.], [-1., 0., 0.]])
    first = jnp.array([[-1., .5], [1., -.5]])
    key = jax.random.PRNGKey(37)
    selected, metrics = select_best_of_k(actor, critic, obs, first, key, 8)
    sample_key, _ = jax.random.split(key)
    extra = OptiQPolicy.sample_action(actor, jnp.repeat(obs, 7, axis=0), sample_key).reshape(2, 7, 2)
    candidates = np.concatenate((np.asarray(first)[:, None], np.asarray(extra)), axis=1)
    scores = np.asarray(obs[:, :1]) * candidates[:, :, 0]
    indices = scores.argmax(axis=1)
    np.testing.assert_allclose(selected, candidates[np.arange(2), indices], atol=1e-7)
    assert np.all(indices != 0)
    np.testing.assert_allclose(metrics['best_of_k_q_gain'], (scores.max(axis=1) - scores[:, 0]).mean(), atol=1e-7)
    # Mean-Q and target-min both prefer the opposite extreme for this critic.
    assert np.all(indices != scores.argmin(axis=1))
    assert np.all(np.abs(selected) <= 1)
    params = copy.deepcopy(actor.params)
    params['log_std']['bias'] += 1.
    noisier, _ = select_best_of_k(actor.replace(params=params), critic, obs, first, key, 8)
    assert not np.array_equal(noisier, selected), 'Collection must retain Gaussian epsilon, not just mu'


@pytest.mark.parametrize('k,timestep', [(1, 5000), (8, 4999)])
def test_k1_and_warmup_preserve_original_actions_and_rng(k, timestep):
    model = object.__new__(OptiQDIME)
    model.behavior_uniform_probability = 0.
    model.behavior_best_of_k = k
    model.behavior_best_of_k_key = jax.random.PRNGKey(4)
    model.num_timesteps = timestep
    model.policy = SimpleNamespace()
    original_key = model.behavior_best_of_k_key
    actions, replay = np.ones((1, 2)), np.zeros((1, 2))
    with patch.object(OffPolicyAlgorithm, '_sample_action', return_value=(actions, replay)), \
         patch('optiq_dime.algorithm.select_best_of_k', side_effect=AssertionError('unexpected selection')):
        result, stored = model._sample_action(5000)
    assert result is actions and stored is replay
    np.testing.assert_array_equal(model.behavior_best_of_k_key, original_key)


@pytest.mark.parametrize('value', [0, -1, 1.5, True, float('nan'), float('inf')])
def test_invalid_k_rejected(value):
    cfg = verify(['benchmark=hopper'], 'mujoco_v5_bestof8')
    cfg.alg.behavior_best_of_k = value
    with pytest.raises(ValueError, match='behavior_best_of_k'):
        validate_config(cfg)


def test_winner_profile_changes_only_teacher_and_collection_settings():
    baseline = OmegaConf.to_container(verify(['benchmark=ant']), resolve=True)
    changed = OmegaConf.to_container(verify(['benchmark=ant'], 'mujoco_v5_bestof8'), resolve=True)
    alias = OmegaConf.to_container(verify(['benchmark=ant'], 'v5/bestof8'), resolve=True)
    expected = copy.deepcopy(baseline)
    expected['alg']['behavior_best_of_k'] = 8
    expected['alg']['actor'].update(
        teacher_distribution='best_of_k_winners', teacher_best_of_k=8,
        source_q_eval='min', source_reference='winner_distribution',
        density_correction=False, density_beta=0., density_correction_beta=0.,
        temperature=None, teacher_std_floor=0., proposal_std=0., proposal_std_pretanh=0.)
    for key in ('wandb', 'output_root', 'run_name'):
        expected[key] = changed[key]
    assert changed == expected == alias
    with pytest.raises(ValueError, match='best-of-k collection matches profile'):
        verify(['+alg.behavior_best_of_k=8'])


@pytest.mark.parametrize('task', ['hopper', 'ant'])
@pytest.mark.parametrize('profile', ['mujoco_v5_bestof8', 'mujoco_v5_bestk_combined'])
def test_real_training_replay_and_unchanged_dual_evaluation(tmp_path, monkeypatch, task, profile):
    cfg = verify([f'benchmark={task}', 'seed=4', f'output_root={tmp_path}',
        'alg.batch_size=4', 'alg.buffer_size=32', 'alg.learning_starts=2',
        'alg.actor.learning_starts=2', 'num_eval_episodes=2', 'eval_interval=4',
        'diagnostic_interval=4', 'checkpoint_interval=4'], profile)
    model, callbacks = create_algorithm(cfg)
    model.set_logger(configure(str(tmp_path / 'test_logs'), ['csv']))
    callback = callbacks.callbacks[0]
    callback.eval_env.envs[0].env._max_episode_steps = 2
    model.get_env().envs[0].env._max_episode_steps = 2
    executed = []
    env = model.get_env().envs[0]
    original_step = env.step
    def record_step(action):
        executed.append(np.array(action, copy=True))
        return original_step(action)
    monkeypatch.setattr(env, 'step', record_step)
    # The collector selection helper must never be called by TD or evaluation.
    # Winner OT uses its own independent teacher groups inside the actor update.
    import optiq_dime.algorithm as algorithm
    original_select = algorithm.select_best_of_k
    calls = []
    in_collection = False
    def record_select(*args, **kwargs):
        assert in_collection, 'Selection escaped the collector into learning/evaluation'
        calls.append(args[-1])
        return original_select(*args, **kwargs)
    monkeypatch.setattr(algorithm, 'select_best_of_k', record_select)
    original_collect = model._sample_action
    def collect(*args, **kwargs):
        nonlocal in_collection
        in_collection = True
        try:
            return original_collect(*args, **kwargs)
        finally:
            in_collection = False
    monkeypatch.setattr(model, '_sample_action', collect)
    try:
        model.learn(total_timesteps=8, callback=callbacks)
        assert calls == [8] * 6 and model.behavior_best_of_k_count == 6
        assert model._n_updates == int(model.policy.actor_state.step) == 6
        assert model.backup_mode == 'td' and model.behavior_uniform_count == 0
        if profile == 'mujoco_v5_bestk_combined':
            import csv
            with (tmp_path / 'test_logs' / 'progress.csv').open() as handle:
                rows = [r for r in csv.DictReader(handle) if r.get('train/proposal_best_of_k')]
            assert rows
            for row in rows:
                assert float(row['train/proposal_best_of_k']) == 8
                assert float(row['train/proposal_pilot_selects_best']) == 1
                assert float(row['train/density_beta_mean']) == 1
                assert float(row['train/temperature']) == cfg.alg.actor.temperature
                assert float(row['train/backup_entropy_term']) == 0
        np.testing.assert_allclose(model.policy.unscale_action(model.replay_buffer.actions[:8, 0]), executed, atol=1e-6)
        key = model.behavior_best_of_k_key.copy()
        policy_key, noise_key = model.policy.key.copy(), model.policy.noise_key.copy()
        callback.n_calls = callback.num_timesteps = 12
        callback._on_step()
        np.testing.assert_array_equal(key, model.behavior_best_of_k_key)
        np.testing.assert_array_equal(policy_key, model.policy.key)
        np.testing.assert_array_equal(noise_key, model.policy.noise_key)
        assert len(calls) == 6
        assert callback.evaluations_timesteps == [1, 4, 8, 12]
        for mode in callback.MODES:
            assert np.isfinite(callback.histories[mode]['results']).all()
        np.testing.assert_array_equal(callback.histories['zero_z']['env_seeds'], callback.histories['stochastic_z']['env_seeds'])
        np.testing.assert_array_equal(callback.histories['zero_z']['policy_seeds'], callback.histories['stochastic_z']['policy_seeds'])
    finally:
        callback.eval_env.close()
        model.get_env().close()
        model.logger.close()
