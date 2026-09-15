"""Exact finite winner law and preservation of the OptiQ training components."""
import copy
import csv
import itertools
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import pytest
from stable_baselines3.common.logger import configure

from optiq_dime.algorithm import OptiQDIME
from optiq_dime.bestk_teacher import best_of_k_mass
from optiq_dime.behavior import select_best_of_k
from run_optiq_dime import create_algorithm, validate_config
from scripts.verify_v5 import verify
from test_best_of_k import directional_state
from test_semi_implicit import actor_state, critic_state


def enumerate_winners(weights, scores, k):
    draws = np.asarray(list(itertools.product(range(len(weights)), repeat=k)))
    winners = draws[np.arange(len(draws)), np.asarray(scores)[draws].argmax(axis=1)]
    return np.bincount(winners, weights=np.asarray(weights)[draws].prod(axis=1), minlength=len(weights))


@pytest.mark.parametrize('k', [1, 2, 8])
@pytest.mark.parametrize('scores', [[3., 1., 2.], [2., 1., 2.], [1., 1., 1.]])
def test_mass_matches_exhaustive_iid_draws_including_ties(k, scores):
    weights = np.array([.2, .3, .5])
    expected = enumerate_winners(weights, scores, k)
    actual = best_of_k_mass(jnp.asarray(weights), jnp.asarray(scores), k)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)


def test_permutation_zero_weights_and_float32_tiny_top_mass():
    weights = jnp.array([[.2, 0, .8], [1., 0, 1e-30]])
    scores = jnp.array([[1., 3., 2.], [1., 0, 2.]])
    actual = best_of_k_mass(weights, scores, 8)
    order = np.array([2, 0, 1])
    np.testing.assert_allclose(best_of_k_mass(weights[:, order], scores[:, order], 8), actual[:, order], rtol=2e-6)
    np.testing.assert_allclose(actual.sum(-1), 1, atol=1e-7)
    assert actual[0, 1] == 0 and actual[1, 1] == 0
    np.testing.assert_allclose(actual[1, 2], 8e-30, rtol=2e-6, atol=0)


def test_density_and_temperature_enter_before_winner_selection():
    scores = jnp.array([.2, .4, .7])
    density = jnp.array([.1, 2., .7])
    weights = jax.nn.softmax(scores / .25 - jnp.log(density))
    actual = best_of_k_mass(weights, scores, 8)
    expected = enumerate_winners(np.asarray(weights, dtype=np.float64), scores, 8)
    np.testing.assert_allclose(actual, expected / expected.sum(), atol=2e-7)
    for alternative in (jax.nn.softmax(scores / .25), jax.nn.softmax(scores / 1. - jnp.log(density))):
        assert not np.allclose(actual, best_of_k_mass(alternative, scores, 8), atol=1e-3)
    assert (actual * scores).sum() >= (weights * scores).sum()


def update(actor, critic, key, k=None):
    optional = {} if k is None else {'teacher_boltzmann_best_of_k': k}
    return OptiQDIME.update_actor(actor, critic, jnp.ones((4, 3)), key, jnp.array([-3600.]),
        16, 4, 'exact', .05, .5, False, True, 1., False, 16., 257,
        .25, .1, 100, 'mean', 'argmax', True, False,
        'conditional_ot_nll', 'conditional_mixture', 0., False, 'mean', **optional)


def test_k1_is_original_update_and_equal_q_does_not_bias_ties():
    actor, critic, key = actor_state(), critic_state(), jax.random.PRNGKey(17)
    base, identity, tied = update(actor, critic, key), update(actor, critic, key, 1), update(actor, critic, key, 8)
    for a, b in zip(jax.tree.leaves(base), jax.tree.leaves(identity)):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(jax.tree.leaves(base[:3]), jax.tree.leaves(tied[:3])):
        np.testing.assert_allclose(a, b, rtol=2e-6, atol=2e-7)
    assert abs(float(tied[3]['teacher_bestk_q_gain'])) < 1e-6


def test_gradient_boundaries_heads_and_original_sampling_rng():
    actor, critic, key = actor_state(), directional_state(), jax.random.PRNGKey(23)
    changed, loss, newkey, metrics = update(actor, critic, key, 8)
    assert np.isfinite(loss)
    assert metrics['temperature'] == .25 and metrics['density_beta_mean'] == 1
    assert metrics['teacher_boltzmann_best_of_k'] == 8 and metrics['teacher_bestk_q_gain'] > 0
    assert 'proposal_pilot_count' not in metrics
    np.testing.assert_array_equal(newkey, update(actor, critic, key)[2])
    for head in ('mu', 'log_std'):
        assert any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(actor.params[head]), jax.tree.leaves(changed.params[head])))
    qgrad = jax.grad(lambda params: update(actor, critic.replace(params=params), key, 8)[1])(critic.params)
    assert all(np.count_nonzero(g) == 0 for g in jax.tree.leaves(qgrad))


def test_profile_has_only_two_algorithm_changes_from_original_optiq():
    base = OmegaConf.to_container(verify([], 'mujoco_v5'), resolve=True)
    cfg = OmegaConf.to_container(verify([], 'mujoco_v5_bestk_boltzmann'), resolve=True)
    base['alg']['behavior_best_of_k'] = 8
    base['alg']['actor']['teacher_boltzmann_best_of_k'] = 8
    for key in ('run_name', 'wandb', 'output_root'):
        base[key] = cfg[key]
    assert cfg == base == OmegaConf.to_container(verify([], 'v5/bestk_boltzmann'), resolve=True)


@pytest.mark.parametrize('field,value', [
    ('teacher_boltzmann_best_of_k', 0), ('teacher_boltzmann_best_of_k', True),
    ('teacher_boltzmann_best_of_k', 1.5), ('density_correction', False),
    ('density_correction_beta', 0.), ('source_q_eval', 'min'),
    ('normalize_ot_cost', True), ('teacher_distribution', 'best_of_k_winners'),
])
def test_invalid_or_conflicting_teacher_config_rejected(field, value):
    cfg = verify([], 'mujoco_v5_bestk_boltzmann')
    OmegaConf.update(cfg, 'alg.actor.' + field, value, force_add=True)
    with pytest.raises(ValueError):
        validate_config(cfg)


@pytest.mark.parametrize('task', ['ant', 'hopper'])
def test_real_training_uses_original_proposal_plain_td_and_unchanged_evaluation(tmp_path, task):
    temperature = .25 if task == 'ant' else .01
    cfg = verify([f'benchmark={task}', 'seed=0', f'output_root={tmp_path}',
        'alg.batch_size=256', 'alg.buffer_size=32', 'alg.learning_starts=2',
        'alg.actor.learning_starts=2', 'num_eval_episodes=2', 'eval_interval=4',
        'diagnostic_interval=1', 'checkpoint_interval=4', f'alg.actor.temperature={temperature}'],
        'mujoco_v5_bestk_boltzmann')
    model, callbacks = create_algorithm(cfg)
    model.set_logger(configure(str(tmp_path/'logs'), ['csv']))
    callback = callbacks.callbacks[0]
    callback.eval_env.envs[0].env._max_episode_steps = 2
    model.get_env().envs[0].env._max_episode_steps = 2
    mixture_rng = copy.deepcopy(model.behavior_best_of_k_rng.bit_generator.state)
    selected = []
    def collect(*args, **kwargs):
        result = select_best_of_k(*args, **kwargs)
        selected.append(np.asarray(result[0]).copy())
        return result
    try:
        with patch('optiq_dime.algorithm.make_bestk_proposal', side_effect=AssertionError('guided teacher used')), \
             patch('optiq_dime.algorithm.update_winner_actor', side_effect=AssertionError('actor-winner teacher used')), \
             patch('optiq_dime.algorithm.select_best_of_k', side_effect=collect) as selector:
            model.learn(total_timesteps=6, callback=callbacks)
            assert selector.call_count == 4  # Only post-warmup collection, never TD/evaluation.
        model.logger.dump(step=6)
        with (tmp_path/'logs'/'progress.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        logged = [r for r in rows if r.get('train/teacher_boltzmann_best_of_k')]
        assert logged and all(float(r['train/teacher_boltzmann_best_of_k']) == 8 for r in logged)
        assert all(float(r['train/temperature']) == pytest.approx(temperature) for r in logged)
        assert all(float(r['train/density_beta_mean']) == 1 for r in logged)
        assert not any(r.get('train/proposal_pilot_count') for r in rows)
        assert model._n_updates == 4 and model.backup_mode == 'td'
        assert model.behavior_best_of_k_rng.bit_generator.state == mixture_rng
        np.testing.assert_allclose(model.replay_buffer.actions[2:6], np.asarray(selected))
        assert set(callback.histories) == {'zero_z', 'stochastic_z'}
        assert int(model.policy.actor_state.step) == 4
        for leaf in jax.tree.leaves((model.policy.actor_state.params, model.policy.qf_state.params)):
            assert np.isfinite(leaf).all()
    finally:
        callback.eval_env.close()
        model.get_env().close()
        model.logger.close()
