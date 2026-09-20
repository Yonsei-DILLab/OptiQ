"""GMM adapter preserves complete state and stops at the first rejected update."""
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gmm40 import v8 as adapter
from gmm40.v7 import GMM40V7
from gmm40.v8 import GMM40V8, OTConvergenceError
from gmm40.validate_v7 import checkpoint_audit, tree_error


def assert_same_values(left, right):
    """JIT may materialize the initial Python step=0 as a JAX int32 scalar."""
    left_leaves, right_leaves = jax.tree.leaves(left), jax.tree.leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for before, after in zip(left_leaves, right_leaves):
        np.testing.assert_array_equal(before, after)


def small_agent(**overrides):
    settings = dict(seed=3, batch=2, num_students=16, proposal_components=8,
                    teacher_resample_count=4, actor_samples=4, latent_seed=3,
                    hidden_dims=(8, 8), max_iterations=100, min_iterations=5,
                    relative_tolerance=1e-3)
    settings.update(overrides)
    return GMM40V8(**settings)


@pytest.mark.parametrize('clip', [None, 2.])
def test_full_actor_adam_rng_checkpoint_resume_is_exact(tmp_path, clip):
    original = small_agent(actor_max_grad_norm=clip)
    metrics = original.advance(3)
    assert metrics['actor_updates_attempted'] == metrics['actor_updates_accepted'] == 3.
    assert original.dual_state is None
    checkpoint = tmp_path / 'v8.bin'
    original.save(checkpoint)
    audit = checkpoint_audit(checkpoint, 3)
    assert audit['optimizer_counts'] == [3]
    assert audit['dual_updates'] is None
    restored = small_agent(actor_max_grad_norm=clip)
    restored.restore(checkpoint)
    assert tree_error(original.checkpoint(), restored.checkpoint()) == 0
    first, second = original.advance(2), restored.advance(2)
    assert first == second
    assert tree_error(original.checkpoint(), restored.checkpoint()) == 0
    key = np.asarray(original.key).copy()
    original.evaluate_samples(64, 900003)
    np.testing.assert_array_equal(key, original.key)


def test_checkpoint_rejects_v7_or_different_v8_settings(tmp_path):
    current = small_agent()
    checkpoint = tmp_path / 'v8.bin'
    current.save(checkpoint)
    assert json.loads(current.settings_signature)['version'] == 8
    for override in ({'latent_seed': 4}, {'actor_max_grad_norm': 2.},
                     {'relative_tolerance': 2e-3}, {'max_iterations': 101}):
        incompatible = small_agent(**override)
        with pytest.raises(ValueError, match='settings/seed mismatch'):
            incompatible.restore(checkpoint)
        assert incompatible.updates == int(incompatible.state.step) == 0
    historical = GMM40V7(seed=3, batch=2, num_students=16, proposal_components=8,
        teacher_resample_count=4, actor_samples=4, latent_seed=3, hidden_dims=(8, 8),
        potential_solver='fresh_sinkhorn', source_importance_correction=False)
    old_checkpoint = tmp_path / 'v7.bin'
    historical.save(old_checkpoint)
    with pytest.raises(ValueError, match='settings/seed mismatch'):
        current.restore(old_checkpoint)


@pytest.mark.parametrize('accepted_prefix', [0, 2])
def test_first_rejection_stops_chunk_and_preserves_last_accepted_state(
        monkeypatch, tmp_path, accepted_prefix):
    calls = []

    def controlled_update(state, observations, key, q_fn, **settings):
        del observations, q_fn, settings
        # Count executed updates, not Python traces of the JIT/while body.
        jax.debug.callback(lambda step: calls.append(int(step)), state.step, ordered=True)
        accept = state.step < accepted_prefix
        gradients = jax.tree.map(lambda value: jnp.full_like(value, .2), state.params)
        proposed = state.apply_gradients(grads=gradients)
        updated = jax.lax.cond(accept, lambda _: proposed, lambda _: state, operand=None)
        next_key = jnp.where(accept, jax.random.split(key)[0], key)
        scalar = jnp.asarray(state.step + 1, dtype=jnp.float32)
        metrics = dict(actor_update_accepted=accept.astype(jnp.float32),
            ot_converged_fraction=accept.astype(jnp.float32),
            ot_row_relative_error=.1 * scalar, ot_col_relative_error=.01 * scalar,
            ot_iterations_max=scalar, actor_gradient_norm=2. * scalar,
            actor_loss=scalar)
        return updated, scalar, next_key, metrics

    monkeypatch.setattr(adapter.conditional_sac_v8, 'update_actor', controlled_update)
    agent = small_agent()
    expected_state, expected_key = agent.state, agent.key
    for _ in range(accepted_prefix):
        gradients = jax.tree.map(lambda value: jnp.full_like(value, .2), expected_state.params)
        expected_state = jax.jit(lambda state, grads: state.apply_gradients(grads=grads))(
            expected_state, gradients)
        expected_key = jax.random.split(expected_key)[0]
    with pytest.raises(OTConvergenceError, match='rejected update'):
        agent.advance(100)
    jax.effects_barrier()
    assert calls == list(range(accepted_prefix + 1))
    assert agent.updates == int(agent.state.step) == accepted_prefix
    for actual, expected in zip(jax.tree.leaves(agent.state), jax.tree.leaves(expected_state)):
        # Independent Adam compilation can fuse float32 operations differently.
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-9)
    np.testing.assert_array_equal(agent.key, expected_key)
    metrics = agent.last_metrics
    assert metrics['actor_updates_attempted'] == accepted_prefix + 1
    assert metrics['actor_updates_accepted'] == accepted_prefix
    assert metrics['actor_update_accepted'] == metrics['ot_converged_fraction'] == 0.
    np.testing.assert_allclose(metrics['actor_loss'], (accepted_prefix + 2) / 2)
    np.testing.assert_allclose(metrics['ot_row_relative_error'], .1 * (accepted_prefix + 1))
    assert metrics['actor_gradient_norm'] == 2. * (accepted_prefix + 1)
    checkpoint = tmp_path / 'failure.bin'
    agent.save(checkpoint)
    checkpoint_audit(checkpoint, accepted_prefix)
    restored = small_agent()
    restored.restore(checkpoint)
    assert tree_error(agent.checkpoint(), restored.checkpoint()) == 0
    unchanged = agent.checkpoint()
    with pytest.raises(OTConvergenceError):
        agent.advance(100)
    jax.effects_barrier()
    assert calls == [*range(accepted_prefix + 1), accepted_prefix]
    assert_same_values(unchanged, agent.checkpoint())
    assert agent.last_metrics['actor_updates_attempted'] == 1.


def test_actual_nonconvergent_solve_preserves_actor_adam_and_rng():
    agent = small_agent(min_iterations=1, max_iterations=1, relative_tolerance=1e-9)
    previous = agent.checkpoint()
    with pytest.raises(OTConvergenceError):
        agent.advance(20)
    assert agent.last_metrics['ot_converged_fraction'] < 1.
    assert agent.last_metrics['actor_updates_attempted'] == 1.
    assert agent.last_metrics['actor_updates_accepted'] == 0.
    assert_same_values(previous, agent.checkpoint())
