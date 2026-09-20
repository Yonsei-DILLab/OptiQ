"""The shared v7 actor update: teacher correction, joint sampling and gradients."""

import itertools
import math

from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from optiq_dime.conditional_sac import (
    actor_objective,
    components,
    conditional_action_log_prob,
    prepare_batch,
    stratified_resample,
    update_actor,
)
from optiq_dime.latent_transport import (
    latent_assignment_log_probs, latent_transport_cost, sinkhorn_with_source_potential,
)
from optiq_dime.policy import SemiImplicitActor
from optiq_dime.semi_implicit import ConditionalGaussianProposal


def small_actor():
    model = SemiImplicitActor(2, (16, 16), -5.0, 1.0, math.log(0.5))
    params = model.init(jax.random.PRNGKey(3), jnp.zeros((1, 3)), jnp.zeros((1, 2)))["params"]
    return TrainState.create(apply_fn=model.apply, params=params, tx=optax.sgd(1e-3))


def bias_actor(mu=0.2, log_std=math.log(0.5)):
    def apply(variables, observations, latents):
        parameters = variables["params"]
        return (jnp.broadcast_to(parameters["mu"]["bias"], latents.shape),
                jnp.broadcast_to(parameters["log_std"]["bias"], latents.shape))
    return TrainState.create(
        apply_fn=apply, params={"mu": {"bias": jnp.array([mu])},
                               "log_std": {"bias": jnp.array([log_std])}},
        tx=optax.sgd(1e-3),
    )


def distinct_component_actor(call_shapes):
    """Small affine MLP fixture whose proposal components are distinguishable."""
    def apply(variables, observations, latents):
        del observations
        call_shapes.append(tuple(latents.shape))
        parameters = variables["params"]
        mu = latents @ parameters["mu"]["kernel"] + parameters["mu"]["bias"]
        log_std = jnp.broadcast_to(parameters["log_std"]["bias"], latents.shape)
        return mu, log_std
    return TrainState.create(
        apply_fn=apply,
        params={"mu": {"kernel": jnp.array([[0.8, 0.1], [-0.2, 0.6]]),
                       "bias": jnp.array([0.1, -0.2])},
                "log_std": {"bias": jnp.log(jnp.array([0.015, 0.12]))}},
        tx=optax.sgd(1e-4),
    )


def quadratic_q(observations, actions):
    del observations
    return -jnp.sum((actions - 0.4) ** 2, axis=-1)


def small_settings():
    return dict(num_students=8, proposal_components=4, proposals_per_component=4,
                teacher_resample_count=4, actor_samples=4, latent_seed=7,
                temperature=0.3, epsilon=0.4, iterations=80)


def manual_data(anchors, potential, source_indices, importance=None):
    if importance is None:
        importance = jnp.ones(source_indices.shape)
    actor_z = jnp.take_along_axis(anchors, source_indices[..., None], axis=1)
    return dict(anchors=anchors, actor_z=actor_z, source_indices=source_indices,
                actor_noise=jnp.zeros_like(actor_z), ot={"source_potential": potential},
                source_importance=jax.lax.stop_gradient(importance),
                source_log_importance=jax.lax.stop_gradient(jnp.log(importance)))


def test_fixed_quadrature_but_fresh_teacher_and_gaussian_noise():
    actor, obs = small_actor(), jnp.zeros((2, 3))
    first = prepare_batch(actor, obs, jax.random.PRNGKey(21), quadratic_q, **small_settings())
    second = prepare_batch(actor, obs, jax.random.PRNGKey(22), quadratic_q, **small_settings())
    np.testing.assert_array_equal(first["anchors"], second["anchors"])
    np.testing.assert_array_equal(first["anchors"][0], first["anchors"][1])
    expected_bank = jax.random.normal(jax.random.PRNGKey(7), (8, 2))
    np.testing.assert_array_equal(first["anchors"][0], expected_bank)
    assert not np.array_equal(first["teacher_u"], second["teacher_u"])
    assert not np.array_equal(first["actor_noise"], second["actor_noise"])
    assert not np.array_equal(first["ot"]["source_potential"], second["ot"]["source_potential"])


def test_teacher_weighting_occurs_once_before_compression_and_joint_draws():
    actor, obs, key = small_actor(), jnp.zeros((2, 3)), jax.random.PRNGKey(31)
    settings = small_settings()
    data = prepare_batch(actor, obs, key, quadratic_q, **settings)
    _, _, proposal_z_key, _, resample_key, index_key, _ = jax.random.split(key, 7)
    proposal_z = jax.random.normal(proposal_z_key, (2, 4, 2))
    mu, log_std = components(actor, actor.params, obs, proposal_z)
    actual_log_q = ConditionalGaussianProposal(mu, log_std, 0.05).log_prob(data["teacher_u"])
    np.testing.assert_allclose(data["teacher_log_q"], actual_log_q, atol=1e-6)
    expected_log_w = jax.nn.log_softmax(data["teacher_q"] / 0.3 - actual_log_q, axis=-1)
    np.testing.assert_allclose(data["teacher_log_w"], expected_log_w, atol=1e-6)
    expected_teacher_indices = stratified_resample(resample_key, jnp.exp(expected_log_w), 4)
    np.testing.assert_array_equal(data["teacher_indices"], expected_teacher_indices)
    np.testing.assert_array_equal(data["ot_weights"], jnp.full((2, 4), 0.25))
    np.testing.assert_array_equal(data["pair_teacher_indices"], jnp.broadcast_to(jnp.arange(4), (2, 4)))
    np.testing.assert_array_equal(data["pair_u"], data["ot_u"])
    np.testing.assert_array_equal(data["pair_u"],
                                  jnp.take_along_axis(data["teacher_u"], expected_teacher_indices[..., None], axis=1))

    log_r = latent_assignment_log_probs(data["ot_u"], data["anchors"],
                                        data["ot"]["source_potential"], 0.4)
    expected_sources = jax.random.categorical(index_key, log_r, axis=-1)
    np.testing.assert_array_equal(data["source_indices"], expected_sources)
    np.testing.assert_array_equal(data["actor_z"],
                                  jnp.take_along_axis(data["anchors"], expected_sources[..., None], axis=1))
    log_mass = jax.scipy.special.logsumexp(log_r - math.log(4), axis=1)
    np.testing.assert_allclose(data["source_log_mass"], log_mass, atol=2e-6)
    np.testing.assert_allclose(jnp.exp(log_mass), data["ot"]["source_mass"], atol=2e-7)
    expected_importance = jnp.exp(-math.log(8) - jnp.take_along_axis(log_mass, expected_sources, axis=1))
    np.testing.assert_allclose(data["source_importance"], expected_importance, atol=2e-6)


def test_default_teacher_uses_256_fresh_latents_one_draw_each_and_full_mixture_density():
    calls = []
    actor = distinct_component_actor(calls)
    obs, key = jnp.zeros((2, 1)), jax.random.PRNGKey(321)
    # Leave H, proposal count, samples/component, sampling mode and K at their
    # public defaults: this is a regression against silently returning to 16x16.
    data = prepare_batch(actor, obs, key, quadratic_q, temperature=0.3,
                         epsilon=0.4, iterations=3, action_scale=40)
    assert calls == [(2 * 256, 2)]
    assert data["anchors"].shape == (2, 4096, 2)
    assert data["teacher_u"].shape == (2, 256, 2)
    assert data["cost"].shape == (2, 4096, 16)
    assert data["actor_z"].shape == (2, 16, 2)
    np.testing.assert_array_equal(data["teacher_component_indices"],
                                  jnp.broadcast_to(jnp.arange(256), (2, 256)))
    _, _, proposal_z_key, proposal_key, resample_key, _, _ = jax.random.split(key, 7)
    latents = jax.random.normal(proposal_z_key, (2, 256, 2))
    mu = latents @ actor.params["mu"]["kernel"] + actor.params["mu"]["bias"]
    raw_std = jnp.exp(actor.params["log_std"]["bias"])
    std = jnp.maximum(raw_std, 0.05)
    assert float(raw_std[0]) < 0.05 < float(raw_std[1])
    _, noise_key = jax.random.split(proposal_key)
    expected_u = mu + std * jax.random.normal(noise_key, (2, 256, 2))
    # One draw from each matching latent, in order: IID mixture-index sampling
    # or 16 centers with 16 draws each cannot reproduce this array.
    np.testing.assert_allclose(data["teacher_u"], expected_u, atol=2e-7, rtol=2e-6)
    np.testing.assert_allclose(data["teacher_actions"], jnp.tanh(expected_u), atol=2e-7)

    u = data["teacher_u"]
    component_log_density = jnp.sum(
        -0.5 * ((u[:, :, None] - mu[:, None]) / std) ** 2
        - jnp.log(std) - 0.5 * math.log(2 * math.pi), axis=-1)
    log_tanh_jacobian = (2 * (math.log(2) - u - jax.nn.softplus(-2 * u))).sum(-1)
    expected_log_q = (jax.scipy.special.logsumexp(component_log_density, axis=-1)
                      - math.log(256) - log_tanh_jacobian - 2 * math.log(40))
    np.testing.assert_allclose(data["teacher_log_q"], expected_log_q, atol=4e-5, rtol=2e-6)
    # A density summing only 16 Gaussian components is observably wrong here.
    wrong_log_q = (jax.scipy.special.logsumexp(component_log_density[..., :16], axis=-1)
                   - math.log(16) - log_tanh_jacobian - 2 * math.log(40))
    assert float(jnp.mean(jnp.abs(wrong_log_q - expected_log_q))) > 0.1
    expected_log_w = jax.nn.log_softmax(data["teacher_q"] / 0.3 - expected_log_q, axis=-1)
    np.testing.assert_allclose(data["teacher_log_w"], expected_log_w, atol=4e-5, rtol=3e-6)
    expected_indices = stratified_resample(resample_key, jnp.exp(data["teacher_log_w"]), 16)
    np.testing.assert_array_equal(data["teacher_indices"], expected_indices)
    np.testing.assert_array_equal(data["ot_weights"], jnp.full((2, 16), 1 / 16))
    np.testing.assert_array_equal(data["pair_teacher_indices"], jnp.broadcast_to(jnp.arange(16), (2, 16)))
    np.testing.assert_array_equal(data["pair_u"], jnp.take_along_axis(u, expected_indices[..., None], axis=1))


def test_default_update_forwards_only_256_proposal_and_16_selected_latents():
    calls = []
    actor = distinct_component_actor(calls)
    updated, loss, _, metrics = update_actor(
        actor, jnp.zeros((1, 1)), jax.random.PRNGKey(322), quadratic_q,
        temperature=0.3, epsilon=0.4, iterations=3)
    assert calls == [(256, 2), (16, 2)]
    assert int(updated.step) == 1 and np.isfinite(loss)
    assert float(metrics["teacher_candidate_count"]) == 256
    assert float(metrics["teacher_proposal_component_count"]) == 256
    assert float(metrics["teacher_one_per_latent"]) == 1
    assert float(metrics["ot_teacher_count"]) == float(metrics["actor_training_pairs"]) == 16
    assert float(metrics["ot_source_count"]) == 4096
    assert float(metrics["v7_conditional_sac_used"]) == 1
    assert float(metrics["actor_assignment_term"]) > 0


def test_importance_corrects_objective_and_gradient_by_exact_enumeration():
    actor, obs = bias_actor(), jnp.zeros((1, 1))
    anchors = jnp.array([[[-1.0], [0.0], [1.0]]])
    potential = jnp.array([[0.15, -0.2, 0.1]])
    teacher_u = jnp.array([[[-0.8], [0.2]]])
    epsilon, temperature = 0.7, 0.3
    r = jnp.exp(latent_assignment_log_probs(teacher_u, anchors, potential, epsilon))[0]
    mass = r.mean(axis=0)
    assert float(jnp.max(jnp.abs(mass - 1 / 3))) > 0.1
    reference = manual_data(anchors, potential, jnp.array([[0, 1, 2]]))

    def value_and_gradient(data):
        return jax.value_and_grad(lambda parameters: actor_objective(
            parameters, actor, obs, data, quadratic_q, temperature=temperature,
            epsilon=epsilon)[0])(actor.params)

    expected_value, expected_gradient = value_and_gradient(reference)
    enumerated_value = 0.0
    enumerated_gradient = jax.tree.map(jnp.zeros_like, actor.params)
    uncorrected_value = 0.0
    for chosen in itertools.product(range(3), repeat=2):
        indices = jnp.array([chosen])
        probability = r[0, chosen[0]] * r[1, chosen[1]]
        importance = (1 / 3) / mass[indices]
        value, gradient = value_and_gradient(manual_data(anchors, potential, indices, importance))
        enumerated_value += probability * value
        enumerated_gradient = jax.tree.map(lambda accumulated, current: accumulated + probability * current,
                                           enumerated_gradient, gradient)
        uncorrected_value += probability * value_and_gradient(manual_data(anchors, potential, indices))[0]
    np.testing.assert_allclose(enumerated_value, expected_value, atol=2e-6)
    for actual, expected in zip(jax.tree.leaves(enumerated_gradient), jax.tree.leaves(expected_gradient)):
        np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-6)
    assert abs(float(uncorrected_value - expected_value)) > 0.01


def test_balanced_map_has_unit_importance_and_second_moment():
    actor, obs = bias_actor(), jnp.zeros((1, 1))
    # Duplicate sites make every posterior source mass exactly uniform.
    data = prepare_batch(actor, obs, jax.random.PRNGKey(41), quadratic_q,
                         num_students=4, proposal_components=2, proposals_per_component=2,
                         teacher_resample_count=2, actor_samples=2, student_latents=jnp.zeros((1, 4, 1)))
    np.testing.assert_allclose(data["source_importance"], 1.0, atol=1e-6)
    log_second_moment = jax.scipy.special.logsumexp(-2 * math.log(4) - data["source_log_mass"], axis=-1)
    np.testing.assert_allclose(log_second_moment, 0.0, atol=1e-6)


def test_log_mass_survives_float32_assignment_probability_underflow():
    actor, obs = bias_actor(), jnp.zeros((1, 1))
    data = prepare_batch(actor, obs, jax.random.PRNGKey(51), quadratic_q,
                         num_students=2, proposal_components=1, proposals_per_component=4,
                         teacher_resample_count=2, actor_samples=2, iterations=1, epsilon=0.001,
                         student_latents=jnp.array([[[-3.0], [3.0]]]))
    assert np.isfinite(data["source_log_mass"]).all()
    assert np.isfinite(data["source_importance"]).all()
    log_r = latent_assignment_log_probs(data["ot_u"], data["anchors"],
                                        data["ot"]["source_potential"], 0.001)
    # Individual assignments can underflow, but the complete row-then-column
    # Sinkhorn cycle bounds each row mass below by h_i/K in exact arithmetic.
    assert float(log_r.min()) < -100
    assert np.count_nonzero(np.asarray(jnp.exp(log_r)) == 0) > 0
    assert float(data["source_log_mass"].min()) >= -math.log(4) - 1e-3


@pytest.mark.parametrize("iterations", [1, 10, 100])
@pytest.mark.parametrize("epsilon", [0.001, 0.1])
@pytest.mark.parametrize("source_count", [8, 64])
def test_completed_sinkhorn_cycle_bounds_all_source_importance(source_count, epsilon, iterations):
    teacher_count = 4
    source_key, teacher_key = jax.random.split(jax.random.PRNGKey(62))
    anchors = 2 * jax.random.normal(source_key, (2, source_count, 2))
    teachers = 0.5 * jax.random.normal(teacher_key, (2, teacher_count, 2))
    weights = jnp.full((2, teacher_count), 1 / teacher_count)
    ot = sinkhorn_with_source_potential(latent_transport_cost(anchors, teachers), weights,
                                        epsilon, iterations)
    log_r = latent_assignment_log_probs(teachers, anchors, ot["source_potential"], epsilon)
    log_mass = jax.scipy.special.logsumexp(log_r - math.log(teacher_count), axis=1)
    log_importance = -math.log(source_count) - log_mass
    # If R is the last row projection, sum_j R_ij=h_i and d_j=sum_i R_ij<=1.
    # P_ij=R_ij/(K*d_j), hence m_i>=h_i/K and h_i/m_i<=K, even at 1 iteration.
    assert np.isfinite(log_importance).all()
    assert float(log_importance.max()) <= math.log(teacher_count) + 0.005
    log_second_moment = jax.scipy.special.logsumexp(
        -2 * math.log(source_count) - log_mass, axis=-1)
    assert float(log_second_moment.min()) >= -2e-6
    assert float(log_second_moment.max()) <= math.log(teacher_count) + 0.005


def test_actor_mean_gradient_contains_q_entropy_and_ot_input_derivatives():
    actor, obs = bias_actor(), jnp.zeros((1, 1))
    anchors, potential = jnp.array([[[-1.0], [1.0]]]), jnp.array([[0.2, -0.1]])
    data = manual_data(anchors, potential, jnp.array([[0]]))
    temperature, epsilon, q_slope = 0.3, 0.6, 1.7
    def q_fn(observations, actions):
        del observations
        return q_slope * actions[..., 0]
    def loss(parameters):
        return actor_objective(parameters, actor, obs, data, q_fn,
                               temperature=temperature, epsilon=epsilon)[0]
    gradient = jax.grad(loss)(actor.params)
    mu = actor.params["mu"]["bias"][0]
    probabilities = jnp.exp(latent_assignment_log_probs(mu.reshape(1, 1, 1), anchors, potential, epsilon))[0, 0]
    assignment_gradient = 2 / epsilon * (anchors[0, 0, 0] - jnp.sum(probabilities * anchors[0, :, 0]))
    expected_mu_gradient = (2 * temperature * jnp.tanh(mu)
                            - q_slope * (1 - jnp.tanh(mu) ** 2)
                            - temperature * assignment_gradient)
    np.testing.assert_allclose(gradient["mu"]["bias"][0], expected_mu_gradient, atol=2e-6)
    # With zero action noise, only conditional Gaussian entropy affects log_std.
    np.testing.assert_allclose(gradient["log_std"]["bias"], [-temperature], atol=2e-6)


def test_reported_loss_terms_sum_to_loss_with_nonunit_importance():
    actor, obs = bias_actor(), jnp.zeros((1, 1))
    anchors = jnp.array([[[-1.0], [1.0]]])
    data = manual_data(anchors, jnp.array([[0.2, -0.1]]), jnp.array([[0, 1]]),
                       importance=jnp.array([[0.4, 2.1]]))
    loss, metrics = actor_objective(actor.params, actor, obs, data, quadratic_q,
                                    temperature=0.3, epsilon=0.6)
    np.testing.assert_allclose(
        metrics["actor_entropy_term"] + metrics["actor_q_term"] + metrics["actor_assignment_term"],
        loss, atol=2e-6,
    )
    # Raw descriptive Q/entropy metrics are deliberately not loss components.
    assert not np.isclose(float(metrics["actor_q_term"]), -float(metrics["actor_q_mean"]))
    assert not np.isclose(float(metrics["actor_entropy_term"]),
                          -0.3 * float(metrics["actor_conditional_entropy"]))


def test_teacher_and_transport_are_frozen_while_actor_heads_update():
    actor, obs, key = small_actor(), jnp.zeros((2, 3)), jax.random.PRNGKey(61)
    settings = small_settings()
    def frozen_labels_sum(parameters):
        data = prepare_batch(actor.replace(params=parameters), obs, key, quadratic_q, **settings)
        return sum(jnp.sum(data[name]) for name in ("teacher_u", "teacher_log_w", "source_importance")) + jnp.sum(data["ot"]["source_potential"])
    label_gradient = jax.grad(frozen_labels_sum)(actor.params)
    assert all(np.count_nonzero(value) == 0 for value in jax.tree.leaves(label_gradient))

    step = jax.jit(lambda state, step_key: update_actor(state, obs, step_key, quadratic_q, **settings))
    new_state, loss, next_key, metrics = step(actor, key)
    assert int(new_state.step) == int(actor.step) + 1
    assert np.isfinite(loss) and all(np.isfinite(value).all() for value in metrics.values())
    np.testing.assert_array_equal(next_key, jax.random.split(key, 7)[0])
    assert float(metrics["actor_sampling_ot_joint"]) == 1
    assert float(metrics["actor_source_importance_used"]) == 1
    assert float(metrics["ot_source_log_importance_second_moment"]) >= -2e-6
    for head in ("mu", "log_std"):
        assert any(not np.array_equal(old, new) for old, new in zip(
            jax.tree.leaves(actor.params[head]), jax.tree.leaves(new_state.params[head])))


def test_conditional_density_has_tanh_and_physical_scale_jacobians():
    mu, log_std = jnp.array([[[0.2, -0.3]]]), jnp.log(jnp.array([[[0.5, 0.7]]]))
    u = jnp.array([[[1.2, -0.8]]])
    gaussian = jnp.sum(-0.5 * ((u - mu) / jnp.exp(log_std)) ** 2 - log_std
                        - 0.5 * math.log(2 * math.pi), axis=-1)
    expected = gaussian - jnp.log(1 - jnp.tanh(u) ** 2).sum(axis=-1) - 2 * math.log(40)
    np.testing.assert_allclose(conditional_action_log_prob(u, mu, log_std, 40), expected, atol=2e-6)
