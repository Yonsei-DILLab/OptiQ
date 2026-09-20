"""Marginal correctness and actor-gradient routing for latent OT in v7."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from optiq_dime.latent_transport import (
    latent_assignment_log_probs,
    latent_transport_cost,
    sinkhorn_with_source_potential,
)
from optiq_dime.transport import sinkhorn


def fixture_points():
    anchors = jnp.array([[[-0.8, 0.2], [0.1, -0.3], [0.9, 0.4]]])
    teacher = jnp.array([[[-0.7, 0.1], [-0.2, 0.4], [0.4, -0.1], [0.8, 0.3]]])
    weights = jnp.array([[0.1, 0.2, 0.5, 0.2]])
    return anchors, teacher, weights


def test_matches_original_solver_and_both_marginals():
    anchors, teacher, weights = fixture_points()
    costs = latent_transport_cost(anchors, teacher)
    result = jax.jit(sinkhorn_with_source_potential, static_argnames="iterations")(
        costs, weights, jnp.asarray(0.3), 300
    )
    np.testing.assert_allclose(result["plan"], sinkhorn(costs, weights, 0.3, 300), atol=8e-8, rtol=2e-6)
    np.testing.assert_allclose(result["source_mass"], 1 / 3, atol=2e-6)
    np.testing.assert_allclose(result["teacher_mass"], weights, atol=1e-7)
    assert float(result["row_error"][0]) < 2e-6
    assert float(result["column_error"][0]) < 1e-7
    assert float(result["source_tv"][0]) < 2e-6


def test_teacher_assignment_reconstructs_joint_without_double_weighting():
    anchors, teacher, weights = fixture_points()
    result = sinkhorn_with_source_potential(latent_transport_cost(anchors, teacher), weights, 0.2, 300)
    log_r = latent_assignment_log_probs(teacher, anchors, result["source_potential"], 0.2)
    reconstructed = jnp.swapaxes(jnp.exp(log_r) * weights[..., None], 1, 2)
    np.testing.assert_allclose(reconstructed, result["plan"], atol=1e-7, rtol=2e-6)
    np.testing.assert_allclose(jnp.exp(log_r).sum(-1), 1.0, atol=1e-7)
    # Applying W a second time changes the teacher marginal. It is not part of r.
    assert not np.allclose((reconstructed * weights[:, None]).sum(1), weights)


def test_assignment_gradient_matches_analytic_and_finite_difference():
    anchors, _, _ = fixture_points()
    query = jnp.array([[[0.13, -0.17], [0.5, 0.23]]])  # K != N is intentional.
    potential = jnp.array([[0.2, -0.3, 0.1]])
    epsilon = 0.4
    def selected_log_probability(q, z, f):
        return latent_assignment_log_probs(q, z, f, epsilon)[0, 0, 1]
    gradient_q, gradient_z, gradient_f = jax.grad(selected_log_probability, argnums=(0, 1, 2))(
        query, anchors, potential
    )
    probabilities = jnp.exp(latent_assignment_log_probs(query, anchors, potential, epsilon))[0, 0]
    expected = 2 / epsilon * (anchors[0, 1] - jnp.sum(probabilities[:, None] * anchors[0], axis=0))
    np.testing.assert_allclose(gradient_q[0, 0], expected, atol=2e-6, rtol=2e-6)
    assert np.linalg.norm(gradient_q[0, 0]) > 0.1
    np.testing.assert_array_equal(gradient_q[0, 1], 0)
    np.testing.assert_array_equal(gradient_z, 0)
    np.testing.assert_array_equal(gradient_f, 0)
    step = 1e-3
    for coordinate in range(2):
        delta = jnp.zeros_like(query).at[0, 0, coordinate].set(step)
        difference = (selected_log_probability(query + delta, anchors, potential)
                      - selected_log_probability(query - delta, anchors, potential)) / (2 * step)
        np.testing.assert_allclose(gradient_q[0, 0, coordinate], difference, atol=2e-4, rtol=3e-4)


def test_transport_labels_stop_all_solver_gradients():
    anchors, teacher, weights = fixture_points()
    costs = latent_transport_cost(anchors, teacher)
    def labels_sum(c, w):
        labels = sinkhorn_with_source_potential(c, w, 0.2, 100)
        return jnp.square(labels["plan"]).sum() + jnp.square(labels["source_potential"]).sum()
    cost_grad, weight_grad = jax.grad(labels_sum, argnums=(0, 1))(costs, weights)
    np.testing.assert_array_equal(cost_grad, 0)
    np.testing.assert_array_equal(weight_grad, 0)


def test_assignment_is_potential_gauge_invariant():
    anchors, teacher, _ = fixture_points()
    potential = jnp.array([[0.2, -0.1, 0.5]])
    original = latent_assignment_log_probs(teacher, anchors, potential, 0.3)
    shifted = latent_assignment_log_probs(teacher, anchors, potential + 4.0, 0.3)
    np.testing.assert_allclose(original, shifted, atol=2e-6, rtol=1e-6)


def test_finite_iteration_error_is_reported_not_hidden_by_row_normalization():
    anchors = jnp.array([[[-2.0], [2.0]]])
    teacher = jnp.array([[[-2.0], [2.0]]])
    weights = jnp.array([[0.9, 0.1]])
    costs = latent_transport_cost(anchors, teacher)
    unfinished = sinkhorn_with_source_potential(costs, weights, 0.1, 1)
    np.testing.assert_allclose(unfinished["teacher_mass"], weights, atol=1e-7)
    assert float(unfinished["source_tv"][0]) > 0.39
    assert float(unfinished["row_error"][0]) > 0.39
    converged = sinkhorn_with_source_potential(costs, weights, 0.1, 1000)
    np.testing.assert_allclose(converged["source_mass"], 0.5, atol=3e-6)
    np.testing.assert_allclose(converged["teacher_mass"], weights, atol=1e-7)


def test_tiny_weights_and_sharp_costs_stay_finite_with_column_projection():
    anchors = jnp.array([[[-3.0], [0.0], [3.0]]])
    teacher = jnp.array([[[-3.0], [2.9], [3.0]]])
    weights = jnp.array([[0.0, 0.3, 0.7]])
    normalized = jnp.clip(weights, 1e-20, 1)
    normalized /= normalized.sum(-1, keepdims=True)
    result = sinkhorn_with_source_potential(latent_transport_cost(anchors, teacher), weights, 0.001, 100)
    assert all(np.isfinite(value).all() for value in result.values())
    np.testing.assert_allclose(result["teacher_mass"], normalized, atol=1e-7, rtol=2e-6)
    log_r = latent_assignment_log_probs(teacher, anchors, result["source_potential"], 0.001)
    assert np.isfinite(log_r).all()
    reconstructed = jnp.swapaxes(jnp.exp(log_r) * normalized[..., None], 1, 2)
    np.testing.assert_allclose(reconstructed, result["plan"], atol=1e-7, rtol=2e-6)


def test_cost_is_coordinate_distance_without_batch_or_dimension_normalization():
    anchors = jnp.array([[[10000.0, 10000.0], [10001.0, 9999.0]]])
    teacher = jnp.array([[[10000.25, 9999.75]]])
    # Norm/dot-product identities in float32 can cancel this short distance.
    np.testing.assert_array_equal(latent_transport_cost(anchors, teacher), [[[0.125], [1.125]]])


@pytest.mark.parametrize("epsilon", [0.0, -0.1, float("inf"), float("nan")])
def test_rejects_invalid_numeric_epsilon(epsilon):
    anchors, teacher, weights = fixture_points()
    with pytest.raises(ValueError, match="epsilon"):
        sinkhorn_with_source_potential(latent_transport_cost(anchors, teacher), weights, epsilon, 100)
    with pytest.raises(ValueError, match="epsilon"):
        latent_assignment_log_probs(teacher, anchors, jnp.zeros(anchors.shape[:2]), epsilon)


def test_rejects_incompatible_coordinate_shapes():
    with pytest.raises(ValueError, match="coordinate dimension"):
        latent_transport_cost(jnp.zeros((1, 16, 2)), jnp.zeros((1, 64, 3)))
