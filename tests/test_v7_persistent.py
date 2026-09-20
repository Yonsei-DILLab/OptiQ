"""Persistent semi-dual descent and its frozen-map actor update contract."""

import math

from flax import serialization
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from optiq_dime import conditional_sac
from optiq_dime.conditional_sac import actor_objective, prepare_batch
from optiq_dime.latent_transport import latent_assignment_log_probs
from optiq_dime.persistent_transport import (
    create_dual_state,
    semidual_objective,
    update_actor_persistent,
)


def _actor(call_shapes=None):
    def apply(variables, observations, latents):
        if call_shapes is not None:
            call_shapes.append(tuple(latents.shape))
        parameters = variables["params"]
        mu = latents @ parameters["mu"]["kernel"] + parameters["mu"]["bias"]
        mu = mu + 0.05 * observations[:, :1]
        return mu, jnp.broadcast_to(parameters["log_std"]["bias"], latents.shape)

    return TrainState.create(
        apply_fn=apply,
        params={"mu": {"kernel": jnp.array([[0.2, -0.1], [0.1, 0.3]]),
                       "bias": jnp.array([0.2, -0.1])},
                "log_std": {"bias": jnp.log(jnp.array([0.3, 0.5]))}},
        tx=optax.sgd(1e-3),
    )


def _q(observations, actions):
    del observations
    return -jnp.sum((actions - 0.4) ** 2, axis=-1)


def _settings():
    return dict(num_students=4, proposal_components=8, proposals_per_component=1,
                teacher_resample_count=2, actor_samples=2, latent_seed=3,
                temperature=0.3, epsilon=0.8, iterations=7)


def _tree_allclose(actual, expected, atol=3e-6):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for x, y in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(x, y, rtol=3e-6, atol=atol)


def _potential(state, observations):
    return state.apply_fn({"params": state.params}, observations)


@pytest.mark.parametrize("epsilon", [0.07, 0.8])
def test_semidual_gradient_is_teacher_weighted_mass_minus_uniform_prior(epsilon):
    cost = jnp.array([[[0.2, 1.0, 0.6], [0.5, 0.1, 0.9], [0.4, 0.8, 0.2]],
                      [[0.7, 0.3, 0.1], [0.2, 0.6, 0.8], [0.9, 0.1, 0.4]]])
    weights = jnp.array([[0.7, 0.2, 0.1], [0.15, 0.3, 0.55]])
    potential = jnp.array([[0.1, -0.2, 0.1], [-0.05, 0.15, -0.1]])
    assignment = jax.nn.softmax((potential[:, :, None] - cost) / epsilon, axis=1)
    mass = jnp.sum(weights[:, None, :] * assignment, axis=-1)
    gradient = jax.grad(lambda f: semidual_objective(cost, weights, f, epsilon))(potential)
    np.testing.assert_allclose(gradient, (mass - 1 / 3) / 2, atol=3e-6)
    np.testing.assert_allclose(gradient.sum(axis=-1), 0, atol=3e-7)
    np.testing.assert_allclose(
        semidual_objective(cost, weights, potential + 1.25, epsilon),
        semidual_objective(cost, weights, potential, epsilon), atol=3e-6)
    # A descent step must improve the objective; catches the opposite dual sign.
    assert float(semidual_objective(cost, weights, potential - 0.02 * gradient, epsilon)) < float(
        semidual_objective(cost, weights, potential, epsilon))


def test_shared_gmm_forward_and_repeated_state_forward_have_equal_dual_gradient():
    one_state = jnp.ones((1, 3))
    dual = create_dual_state(jax.random.PRNGKey(8), one_state,
                             num_sources=4, hidden_dims=(8, 8), learning_rate=1e-3)
    repeated_state = jnp.repeat(one_state, 3, axis=0)
    costs = jax.random.uniform(jax.random.PRNGKey(9), (3, 4, 2))
    weights = jnp.array([[0.7, 0.3], [0.2, 0.8], [0.55, 0.45]])

    def objective(params, observations):
        f = dual.apply_fn({"params": params}, observations)
        return semidual_objective(costs, weights, jnp.broadcast_to(f, (3, 4)), 0.5)

    single = jax.value_and_grad(objective)(dual.params, one_state)
    repeated = jax.value_and_grad(objective)(dual.params, repeated_state)
    _tree_allclose(single, repeated)
    assert any(float(jnp.linalg.norm(g)) > 0 for g in jax.tree.leaves(single[1]))


@pytest.mark.parametrize("shared_gmm_state", [False, True])
def test_both_optimizers_use_the_same_preupdate_potential_and_keep_adam_state(shared_gmm_state):
    actor = _actor()
    obs = jnp.array([[0.3, -0.2, 0.7], [-0.4, 0.8, 0.2]])
    dual_obs = jnp.ones((1, 3)) if shared_gmm_state else obs
    dual = create_dual_state(jax.random.PRNGKey(11), dual_obs,
                             num_sources=4, hidden_dims=(8, 8), learning_rate=0.02)
    key, settings = jax.random.PRNGKey(12), _settings()
    for step in range(2):
        preupdate_potential = _potential(dual, dual_obs)
        data = prepare_batch(actor, obs, key, _q, source_potential=preupdate_potential, **settings)
        expected_loss, actor_gradient = jax.value_and_grad(lambda p: actor_objective(
            p, actor, obs, data, _q, temperature=settings["temperature"],
            epsilon=settings["epsilon"])[0])(actor.params)

        def dual_loss(params):
            f = dual.apply_fn({"params": params}, dual_obs)
            return semidual_objective(data["cost"], data["ot_weights"],
                                      jnp.broadcast_to(f, (2, 4)), settings["epsilon"])

        expected_dual_gradient = jax.grad(dual_loss)(dual.params)
        expected_actor = actor.apply_gradients(grads=actor_gradient)
        expected_dual = dual.apply_gradients(grads=expected_dual_gradient)
        new_actor, new_dual, loss, new_key, metrics = update_actor_persistent(
            actor, dual, obs, key, _q, dual_observations=dual_obs, **settings)
        _tree_allclose(new_actor.params, expected_actor.params)
        _tree_allclose(new_actor.opt_state, expected_actor.opt_state)
        _tree_allclose(new_dual.params, expected_dual.params)
        _tree_allclose(new_dual.opt_state, expected_dual.opt_state)
        np.testing.assert_allclose(loss, expected_loss, atol=2e-6)
        np.testing.assert_array_equal(new_key, data["key"])
        assert int(new_actor.step) == int(new_dual.step) == step + 1
        assert all(np.isfinite(value).all() for value in metrics.values())
        assert any(not np.array_equal(x, y) for x, y in zip(
            jax.tree.leaves(dual.params), jax.tree.leaves(new_dual.params)))
        actor, dual, key = new_actor, new_dual, new_key

    # A checkpoint must retain both optimizer moments and counts, not just f.
    restored = serialization.from_bytes(dual, serialization.to_bytes(dual))
    _tree_allclose(restored.params, dual.params)
    _tree_allclose(restored.opt_state, dual.opt_state)
    assert int(restored.step) == 2


def test_rl_potential_depends_on_state_after_one_semidual_update():
    obs = jnp.array([[0.3, -0.2, 0.7], [-0.4, 0.8, 0.2]])
    dual = create_dual_state(jax.random.PRNGKey(13), obs,
                             num_sources=4, hidden_dims=(8, 8), learning_rate=0.02)
    np.testing.assert_allclose(_potential(dual, obs), 0, atol=0)
    _, updated, _, _, _ = update_actor_persistent(
        _actor(), dual, obs, jax.random.PRNGKey(14), _q, **_settings())
    values = np.asarray(_potential(updated, obs))
    assert np.isfinite(values).all()
    assert np.max(np.abs(values[0] - values[1])) > 1e-5
    np.testing.assert_allclose(values.mean(axis=-1), 0, atol=1e-7)


def test_supplied_potential_preserves_default_teacher_256_to_16_and_skips_sinkhorn(monkeypatch):
    calls = []
    actor, obs, key = _actor(calls), jnp.zeros((1, 3)), jax.random.PRNGKey(15)
    # Keep actual production H/P/M/K/A defaults; only shorten the fresh control.
    control = prepare_batch(actor, obs, key, _q, iterations=1, epsilon=0.8)
    calls.clear()

    def forbidden_solver(*args, **kwargs):
        raise AssertionError("Persistent potential must bypass fresh Sinkhorn")

    monkeypatch.setattr(conditional_sac, "sinkhorn_with_source_potential", forbidden_solver)
    data = prepare_batch(actor, obs, key, _q, epsilon=0.8,
                         source_potential=jnp.zeros((1, 4096)))
    assert calls == [(256, 2)]
    assert data["cost"].shape == (1, 4096, 16)
    assert data["actor_z"].shape == (1, 16, 2)
    for name in ("anchors", "teacher_u", "teacher_log_q", "teacher_q", "teacher_log_w",
                 "teacher_component_indices", "teacher_indices", "ot_u", "ot_weights",
                 "pair_teacher_indices", "pair_u", "actor_noise", "key"):
        np.testing.assert_array_equal(data[name], control[name], err_msg=name)
    np.testing.assert_array_equal(data["teacher_component_indices"], jnp.arange(256)[None])
    np.testing.assert_array_equal(data["ot_weights"], jnp.full((1, 16), 1 / 16))
    np.testing.assert_allclose(data["ot"]["teacher_mass"], 1 / 16, atol=1e-7)


def test_supplied_potential_and_source_importance_are_frozen_but_action_gradient_is_live():
    actor, obs, key = _actor(), jnp.zeros((1, 3)), jax.random.PRNGKey(16)
    settings = _settings()
    potential = jnp.array([[0.2, -0.3, 0.1, 0.0]])

    def loss_from_potential(f):
        data = prepare_batch(actor, obs, key, _q, source_potential=f, **settings)
        return actor_objective(actor.params, actor, obs, data, _q,
                               temperature=0.3, epsilon=0.8)[0]

    np.testing.assert_array_equal(jax.grad(loss_from_potential)(potential), jnp.zeros_like(potential))
    data = prepare_batch(actor, obs, key, _q, source_potential=potential, **settings)
    query = jnp.array([[[0.25, -0.2]]])
    selected = 1
    def log_assignment(u):
        return latent_assignment_log_probs(u, data["anchors"],
                                            data["ot"]["source_potential"], 0.8)[0, 0, selected]
    probabilities = jnp.exp(latent_assignment_log_probs(
        query, data["anchors"], data["ot"]["source_potential"], 0.8))[0, 0]
    expected = 2 / 0.8 * (data["anchors"][0, selected]
                           - jnp.sum(probabilities[:, None] * data["anchors"][0], axis=0))
    np.testing.assert_allclose(jax.grad(log_assignment)(query)[0, 0], expected, atol=2e-6)
    assert float(jnp.linalg.norm(expected)) > 0.1



def test_arbitrary_persistent_map_has_no_compressed_teacher_importance_bound():
    # Identical anchors isolate the potential. Unlike a completed Sinkhorn
    # cycle, an arbitrary persistent f need not give each row at least h_i/K.
    anchors = jnp.zeros((1, 2, 1))
    teacher = jnp.array([[[-0.2], [0.4]]])
    f = jnp.array([[0.0, -20.0]])
    log_r = latent_assignment_log_probs(teacher, anchors, f, epsilon=1.0)
    log_mass = jax.scipy.special.logsumexp(log_r - math.log(2), axis=1)
    log_importance = -math.log(2) - log_mass
    log_second_moment = jax.scipy.special.logsumexp(-2 * math.log(2) - log_mass, axis=-1)
    assert np.isfinite(log_mass).all()
    assert float(log_importance.max()) > math.log(2) + 10
    assert float(log_second_moment[0]) > math.log(2) + 10
    # The expectation identity still holds without clipping/self-normalizing.
    np.testing.assert_allclose(jnp.exp(log_mass + log_importance).sum(axis=-1), 1, atol=1e-6)
