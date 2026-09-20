"""Isolate actor source correction from teacher density correction and OT."""

import math

from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from optiq_dime.conditional_sac import actor_objective, prepare_batch, update_actor
from optiq_dime.persistent_transport import create_dual_state, update_actor_persistent


def _actor():
    def apply(variables, observations, latents):
        parameters = variables['params']
        mu = latents @ parameters['mu']['kernel'] + parameters['mu']['bias']
        mu = mu + 0.1 * observations[:, :1]
        return mu, jnp.broadcast_to(parameters['log_std']['bias'], latents.shape)

    return TrainState.create(
        apply_fn=apply,
        params={'mu': {'kernel': jnp.array([[0.3, -0.1], [0.1, 0.2]]),
                       'bias': jnp.array([0.2, -0.15])},
                'log_std': {'bias': jnp.log(jnp.array([0.3, 0.4]))}},
        tx=optax.sgd(1e-3))


def _q(observations, actions):
    del observations
    return -jnp.sum((actions - 0.4) ** 2, axis=-1)


def _settings():
    return dict(num_students=8, proposal_components=16, proposals_per_component=1,
                teacher_resample_count=4, actor_samples=4, latent_seed=7,
                temperature=0.3, epsilon=0.7, iterations=3, action_scale=2.)


def _equal_trees(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('persistent', [False, True])
def test_switch_changes_only_applied_weights_preserving_teacher_and_transport(persistent):
    actor, obs, key = _actor(), jnp.zeros((2, 1)), jax.random.PRNGKey(31)
    settings = _settings()
    if persistent:
        settings['source_potential'] = jnp.linspace(-0.3, 0.4, 8)[None]
    default = prepare_batch(actor, obs, key, _q, **settings)
    corrected = prepare_batch(actor, obs, key, _q, source_importance_correction=True, **settings)
    uncorrected = prepare_batch(actor, obs, key, _q, source_importance_correction=False, **settings)
    _equal_trees(default, corrected)
    assert corrected.keys() == uncorrected.keys()
    for name in corrected:
        if name not in ('source_importance', 'source_importance_correction'):
            _equal_trees(corrected[name], uncorrected[name])
    np.testing.assert_array_equal(uncorrected['source_importance'], 1.)
    assert not bool(uncorrected['source_importance_correction'])
    assert float(jnp.max(jnp.abs(corrected['source_importance'] - 1.))) > 0.01
    # The Boltzmann teacher correction remains Q/T - log q in both versions.
    expected_log_w = jax.nn.log_softmax(
        corrected['teacher_q'] / settings['temperature'] - corrected['teacher_log_q'], axis=-1)
    np.testing.assert_array_equal(uncorrected['teacher_log_w'], expected_log_w)
    expected_log_ratio = (-math.log(settings['num_students'])
                          - jnp.take_along_axis(uncorrected['source_log_mass'],
                                                uncorrected['source_indices'], axis=1))
    np.testing.assert_array_equal(uncorrected['source_log_importance'], expected_log_ratio)


@pytest.mark.parametrize('correction', [False, True])
def test_conditional_loss_and_gradient_match_manual_applied_weight(correction):
    actor, obs, settings = _actor(), jnp.zeros((2, 1)), _settings()
    data = prepare_batch(actor, obs, jax.random.PRNGKey(32), _q,
                         source_importance_correction=correction,
                         source_potential=jnp.zeros((1, 8)), **settings)

    def manual(params):
        z = data['actor_z']
        mu = z @ params['mu']['kernel'] + params['mu']['bias']
        ls = params['log_std']['bias']
        u = mu + jnp.exp(ls) * data['actor_noise']
        normal_log_density = (-0.5 * ((u - mu) * jnp.exp(-ls)) ** 2
                              - ls - 0.5 * math.log(2 * math.pi)).sum(-1)
        jacobian = (2 * (math.log(2) - u - jax.nn.softplus(-2 * u))).sum(-1)
        log_pi = normal_log_density - jacobian - 2 * math.log(settings['action_scale'])
        distance = ((u[:, :, None] - data['anchors'][:, None]) ** 2).sum(-1)
        log_assignment = jax.nn.log_softmax(
            (data['ot']['source_potential'][:, None] - distance) / settings['epsilon'], axis=-1)
        selected = jnp.take_along_axis(
            log_assignment, data['source_indices'][..., None], axis=-1)[..., 0]
        local_loss = (settings['temperature'] * (log_pi - selected)
                      + ((jnp.tanh(u) - 0.4) ** 2).sum(-1))
        weight = jnp.exp(data['source_log_importance']) if correction else jnp.ones_like(local_loss)
        return jnp.mean(weight * local_loss)

    def implemented(params):
        return actor_objective(params, actor, obs, data, _q,
                               temperature=settings['temperature'], epsilon=settings['epsilon'],
                               action_scale=settings['action_scale'])

    (loss, metrics), gradient = jax.value_and_grad(implemented, has_aux=True)(actor.params)
    expected_loss, expected_gradient = jax.value_and_grad(manual)(actor.params)
    np.testing.assert_allclose(loss, expected_loss, rtol=3e-6, atol=3e-6)
    for actual, expected in zip(jax.tree.leaves(gradient), jax.tree.leaves(expected_gradient)):
        np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)
    assert float(metrics['actor_source_importance_used']) == float(correction)
    np.testing.assert_array_equal(metrics['actor_source_raw_log_importance_max'],
                                  data['source_log_importance'].max())
    if not correction:
        assert float(metrics['actor_source_importance_mean']) == 1.
        assert float(metrics['actor_source_importance_max']) == 1.
        assert float(metrics['actor_source_importance_ess_fraction']) == 1.
        assert float(metrics['actor_source_log_importance_max']) == 0.


def test_persistent_dual_update_is_identical_without_actor_source_correction():
    actor, obs, key, settings = _actor(), jnp.ones((2, 1)), jax.random.PRNGKey(33), _settings()
    dual = create_dual_state(jax.random.PRNGKey(34), obs[:1], num_sources=8,
                             hidden_dims=(8, 8), learning_rate=1e-3)
    corrected = update_actor_persistent(actor, dual, obs, key, _q,
                                        dual_observations=obs[:1], **settings)
    uncorrected = update_actor_persistent(actor, dual, obs, key, _q,
                                          dual_observations=obs[:1],
                                          source_importance_correction=False, **settings)
    _equal_trees(corrected[1], uncorrected[1])
    _equal_trees(corrected[3], uncorrected[3])
    for name in ('ot_dual_gradient_norm', 'ot_dual_loss', 'ot_dual_potential_gradient_rms',
                 'ot_row_marginal_error', 'ot_source_mass_tv', 'source_ess_absolute',
                 'teacher_log_density_mean', 'ot_source_log_importance_second_moment'):
        np.testing.assert_array_equal(corrected[4][name], uncorrected[4][name])
    assert any(not np.array_equal(a, b) for a, b in zip(
        jax.tree.leaves(corrected[0].params), jax.tree.leaves(uncorrected[0].params)))
    assert int(corrected[1].step) == int(uncorrected[1].step) == 1


def test_jit_update_default_is_corrected_and_explicit_disabled_is_unit_weight():
    actor, obs, key, settings = _actor(), jnp.zeros((2, 1)), jax.random.PRNGKey(35), _settings()
    run = jax.jit(lambda correction: update_actor(
        actor, obs, key, _q, source_importance_correction=correction,
        source_potential=jnp.zeros((1, 8)), **settings))
    corrected, disabled = run(True), run(False)
    default = jax.jit(lambda: update_actor(
        actor, obs, key, _q, source_potential=jnp.zeros((1, 8)), **settings))()
    _equal_trees(corrected, default)
    assert int(disabled[0].step) == 1
    assert float(disabled[3]['actor_source_importance_used']) == 0.
    assert float(disabled[3]['actor_source_importance_mean']) == 1.
    assert all(np.isfinite(value).all() for value in disabled[3].values())
