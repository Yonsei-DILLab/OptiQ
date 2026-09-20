"""Actor-only gradient clipping preserves v7 targets and checkpoint identities."""
import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from gmm40.v7 import GMM40V7
from gmm40.validate_v7 import make_agent, tree_error


def small_agent(limit=None):
    return GMM40V7(seed=3, batch=2, num_students=16, proposal_components=8,
                   teacher_resample_count=4, actor_samples=4, iterations=5,
                   hidden_dims=(8, 8), dual_hidden_dims=(8, 8),
                   actor_max_grad_norm=limit)


@pytest.mark.parametrize('limit', [0., -1., float('inf'), float('-inf'), float('nan')])
def test_invalid_clip_limit_rejected(limit):
    with pytest.raises(ValueError, match='actor_max_grad_norm'):
        small_agent(limit)


def test_default_is_plain_adam_with_historical_signature():
    agent = small_agent()
    assert 'actor_max_grad_norm' not in json.loads(agent.settings_signature)
    gradients = jax.tree_util.tree_map(lambda value: jnp.full_like(value, 1000.), agent.state.params)
    plain_adam = optax.adam(3e-4)
    actual = agent.state.tx.update(gradients, agent.state.opt_state, agent.state.params)
    expected = plain_adam.update(gradients, plain_adam.init(agent.state.params), agent.state.params)
    assert tree_error(actual, expected) == 0


@pytest.mark.parametrize('limit', [2., 10.])
def test_large_gradient_is_clipped_before_adam_moments(limit):
    agent = small_agent(limit)
    gradients = jax.tree_util.tree_map(lambda value: jnp.full_like(value, 1000.), agent.state.params)
    raw_norm = optax.global_norm(gradients)
    assert float(raw_norm) > limit
    _, updated_optimizer = jax.jit(agent.state.tx.update)(
        gradients, agent.state.opt_state, agent.state.params)
    # Recover the gradient entering Adam from its first moment, not the final
    # adaptive parameter step: clipping bounds gradients, not parameter deltas.
    first_moment = updated_optimizer[1][0].mu
    recovered = jax.tree_util.tree_map(lambda value: value / .1, first_moment)
    expected = jax.tree_util.tree_map(lambda value: value * limit / raw_norm, gradients)
    assert tree_error(recovered, expected) < 2e-7
    np.testing.assert_allclose(optax.global_norm(recovered), limit, rtol=2e-6)
    np.testing.assert_allclose(updated_optimizer[1][0].nu['mu']['bias'],
                               .001 * expected['mu']['bias'] ** 2, rtol=3e-6)


def test_clipping_preserves_teacher_correction_source_weights_and_dual_update():
    plain, clipped = small_agent(), small_agent(2.)
    assert plain.settings == clipped.settings
    assert tree_error(plain.state.params, clipped.state.params) == 0
    assert tree_error(plain.dual_state, clipped.dual_state) == 0
    before, after = plain.prepare(), clipped.prepare()
    for name in ('teacher_u', 'teacher_log_w', 'teacher_log_q', 'teacher_indices',
                 'source_importance', 'source_log_importance', 'source_indices',
                 'actor_z', 'actor_noise', 'cost', 'ot'):
        assert tree_error(before[name], after[name]) == 0, name
    plain_metrics, clipped_metrics = plain.advance(1), clipped.advance(1)
    # First-step loss, raw gradients, sampled weights and simultaneous dual
    # update remain identical; only the actor optimizer receives clipped grads.
    for name in ('loss', 'actor_gradient_norm', 'actor_source_importance_max',
                 'actor_source_importance_mean'):
        np.testing.assert_allclose(plain_metrics[name], clipped_metrics[name], rtol=2e-6)
    assert tree_error(plain.dual_state, clipped.dual_state) == 0
    assert tree_error(plain.key, clipped.key) == 0
    assert clipped_metrics['actor_gradient_norm_clipped'] <= 2.000001
    assert clipped_metrics['actor_gradient_clip_fraction'] == float(
        clipped_metrics['actor_gradient_norm'] > 2.)
    assert clipped_metrics['actor_gradient_clip_scale'] <= 1.
    assert clipped_metrics['actor_parameter_delta_norm'] > 0
    assert clipped_metrics['actor_gradient_block_max'] == clipped_metrics['actor_gradient_norm']
    assert 'actor_gradient_clip_fraction' not in plain_metrics


def test_clip_checkpoint_resume_and_mismatched_limit_rejection(tmp_path):
    trained = small_agent(2.)
    trained.advance(1)
    checkpoint = tmp_path / 'clip2.bin'
    trained.save(checkpoint)
    restored = small_agent(2.)
    restored.restore(checkpoint)
    assert tree_error(trained.checkpoint(), restored.checkpoint()) == 0
    trained.advance(1)
    restored.advance(1)
    assert tree_error(trained.checkpoint(), restored.checkpoint()) == 0
    for wrong in (small_agent(), small_agent(10.)):
        with pytest.raises(ValueError, match='settings/seed mismatch'):
            wrong.restore(checkpoint)
        assert wrong.updates == int(wrong.state.step) == int(wrong.dual_state.step) == 0
    old_checkpoint = tmp_path / 'plain.bin'
    plain = small_agent()
    plain.save(old_checkpoint)
    same_plain = small_agent(None)
    same_plain.restore(old_checkpoint)
    assert tree_error(plain.checkpoint(), same_plain.checkpoint()) == 0
    with pytest.raises(ValueError, match='settings/seed mismatch'):
        small_agent(2.).restore(old_checkpoint)


def test_make_agent_passes_clip_without_changing_algorithm_settings():
    reference = small_agent()
    config = dict(reference.settings, seed=3, batch=2, hidden_dims=(8, 8),
                  potential_solver='persistent_dual', dual_hidden_dims=(8, 8),
                  actor_max_grad_norm=10.)
    configured = make_agent(config)
    assert configured.actor_max_grad_norm == 10.
    assert configured.settings == reference.settings
    assert json.loads(configured.settings_signature)['actor_max_grad_norm'] == 10.


def test_clip_block_metrics_keep_peaks_and_count_actual_clipped_updates():
    agent = small_agent(2.)
    metrics = dict(actor_gradient_norm=jnp.array([1., 9.]),
                   actor_gradient_clip_fraction=jnp.array([0., 1.]),
                   actor_source_importance_max=jnp.array([2., 500.]),
                   actor_parameter_delta_norm=jnp.array([.01, .03]))
    reduced = agent._aggregate_metrics(metrics)
    assert float(reduced['actor_gradient_norm']) == 5.
    assert float(reduced['actor_gradient_clip_fraction']) == .5
    assert float(reduced['actor_gradient_block_max']) == 9.
    assert float(reduced['actor_source_importance_block_max']) == 500.
    np.testing.assert_allclose(reduced['actor_parameter_step_block_max'], .03)
