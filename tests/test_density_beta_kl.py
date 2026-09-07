"""Independent float64-reference checks for the adaptive finite-candidate KL rule."""
import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Avoid importing the complete RL stack just to test transport math.
_path = Path(__file__).resolve().parents[1] / "optiq_dime" / "transport.py"
_spec = importlib.util.spec_from_file_location("transport_for_kl_test", _path)
_transport = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_transport)
select_density_beta_for_kl = _transport.select_density_beta_for_kl


def reference_solution(q, y, *, budget=None):
    q, y = np.asarray(q, np.float64), np.asarray(y, np.float64)
    q -= q.max()
    y -= y.max()
    log_r = q - np.log(np.exp(q).sum())
    # KL(uniform || r), independently evaluated in float64. Summing these
    # signed terms is accurate enough for the tested score scales (>= 1e-5),
    # unlike the same calculation in float32.
    if budget is None:
        budget = max(float(np.mean(-np.log(len(q)) - log_r)), 0.)

    def kl(beta):
        logits = q + beta*y
        logits -= logits.max()
        log_w = logits - np.log(np.exp(logits).sum())
        return max(float(np.sum(np.exp(log_w) * (log_w-log_r))), 0.)

    if kl(1.) <= budget:
        return 1., kl(1.), budget
    lo, hi = 0., 1.
    for _ in range(70):
        mid = (lo+hi)/2
        if kl(mid) <= budget:
            lo = mid
        else:
            hi = mid
    return lo, kl(lo), budget


@pytest.mark.parametrize("scale", [1., .1, 1e-3, 1e-4, 1e-5])
def test_matches_float64_reference_even_near_flat(scale):
    rng = np.random.default_rng(20260907)
    q = (rng.normal(size=(6, 80))*scale).astype(np.float32)
    y = (rng.normal(size=(6, 80))*7).astype(np.float32)
    result = jax.jit(select_density_beta_for_kl)(jnp.asarray(q), jnp.asarray(y))
    beta, selected, budget = [np.asarray(v) for v in result]
    expected = np.asarray([reference_solution(a.copy(), b.copy()) for a,b in zip(q,y)])
    np.testing.assert_allclose(beta, expected[:, 0], rtol=1e-3, atol=3e-10)
    np.testing.assert_allclose(selected, expected[:, 1], rtol=1e-3, atol=3e-16)
    np.testing.assert_allclose(budget, expected[:, 2], rtol=3e-4, atol=3e-16)
    assert np.all(selected <= budget * (1 + 2e-6))
    assert np.all((beta >= 0) & (beta <= 1))
    assert np.all(budget > 0)


def test_flat_q_and_constant_density_endpoints():
    q = jnp.array([[1000., 1000., 1000., 1000.], [0., 1., 2., 3.], [0., 0., 0., 0.]])
    y = jnp.array([[0., 1., 2., 3.], [-1000., -1000., -1000., -1000.], [5., 5., 5., 5.]])
    beta, selected, budget = select_density_beta_for_kl(q, y)
    np.testing.assert_array_equal(beta, [0., 1., 1.])
    np.testing.assert_array_equal(selected, [0., 0., 0.])
    assert np.asarray(budget)[0] == 0


def test_full_correction_when_budget_sufficient():
    q = jnp.array([[0., 3., 6., 9.]], dtype=jnp.float32)
    y = jnp.array([[0., .01, -.01, .02]], dtype=jnp.float32)
    beta, selected, budget = select_density_beta_for_kl(q, y)
    assert float(beta[0]) == 1.
    assert float(selected[0]) <= float(budget[0])


def test_budget_is_uniform_to_q_kl_for_both_asymmetry_directions():
    # A single high-Q candidate makes KL(r || u) larger; a single low-Q
    # candidate makes it smaller. Neither orientation dominates the other.
    q = np.zeros((2, 8), dtype=np.float32)
    q[:, 0] = [2., -2.]
    y = -2. * q
    beta, selected, budget = jax.jit(select_density_beta_for_kl)(q, y)

    q64 = q.astype(np.float64)
    log_r = q64 - np.log(np.exp(q64).sum(axis=-1, keepdims=True))
    reverse_budget = np.mean(-np.log(q.shape[-1]) - log_r, axis=-1)
    forward_budget = np.sum(np.exp(log_r) * (log_r + np.log(q.shape[-1])), axis=-1)
    assert forward_budget[0] > reverse_budget[0] + .05
    assert forward_budget[1] < reverse_budget[1] - .05
    np.testing.assert_allclose(budget, reverse_budget, rtol=3e-6, atol=1e-7)
    # At beta=1/2 these opposing corrections give exactly uniform weights,
    # which is the KL boundary for the required orientation.
    np.testing.assert_allclose(beta, [.5, .5], rtol=0., atol=1e-6)
    np.testing.assert_allclose(selected, reverse_budget, rtol=3e-6, atol=1e-7)


@pytest.mark.parametrize("tau", [.25, 1., 4.])
def test_selected_candidates_satisfy_expected_q_improvement_bound(tau):
    rng = np.random.default_rng(20260908)
    q_values = (rng.normal(size=(12, 32)) * np.tile([.1, 1., 3.], 4)[:, None]).astype(np.float32)
    q_values[0] = 0.
    q_score = q_values / np.float32(tau)
    density_score = (rng.normal(size=q_values.shape) * 5.).astype(np.float32)
    density_score[1] = 2.  # Full correction with constant density.
    density_score[2] = 2. * q_score[2]  # Correction aligned with Q.
    density_score[3] = -2. * q_score[3]  # Correction opposing Q.

    beta, selected, budget = jax.jit(select_density_beta_for_kl)(q_score, density_score)
    # Evaluate the actual float32 candidate weights, then accumulate the
    # expected-Q gain and entropy independently in float64.
    weights = np.asarray(jax.nn.softmax(q_score + beta[:, None] * density_score)).astype(np.float64)
    uniform = 1. / q_values.shape[-1]
    expected_q_gain = np.sum((weights - uniform) * q_values.astype(np.float64), axis=-1)
    # Sharply aligned corrections can underflow some float32 weights to zero;
    # their entropy contribution is exactly zero by continuity.
    log_w_over_u = np.log(np.where(weights > 0., weights / uniform, 1.))
    kl_w_uniform = np.sum(weights * log_w_over_u, axis=-1)
    tolerance = 32. * np.finfo(np.float32).eps * np.maximum(
        tau, np.max(np.abs(q_values), axis=-1)
    )
    assert np.all(expected_q_gain >= tau * kl_w_uniform - tolerance)
    assert np.all(np.asarray(selected) <= np.asarray(budget) * (1. + 2e-6))
    assert np.any((np.asarray(beta) > 0.) & (np.asarray(beta) < 1.))
    assert float(beta[0]) == 0.
    assert float(beta[1]) == 1.


def test_old_forward_budget_can_lower_expected_q_below_uniform():
    q = np.array([2., 0., 0., 0., 0., 0., 0., 0.], dtype=np.float32)
    y = -2. * q
    q64 = q.astype(np.float64)
    log_r = q64 - np.log(np.exp(q64).sum())
    old_budget = float(np.sum(np.exp(log_r) * (log_r + np.log(len(q)))))
    old_beta, _, _ = reference_solution(q.copy(), y.copy(), budget=old_budget)
    old_logits = q64 + old_beta * y.astype(np.float64)
    old_weights = np.exp(old_logits - old_logits.max())
    old_weights /= old_weights.sum()
    assert old_beta > .6
    assert float(np.dot(old_weights, q64) - q64.mean()) < -.05

    beta, _, _ = select_density_beta_for_kl(q, y)
    weights = np.asarray(jax.nn.softmax(q + beta * y)).astype(np.float64)
    assert float(np.dot(weights, q64) - q64.mean()) >= -1e-6


@pytest.mark.parametrize("density_scale", [1e-10, 1e-20])
def test_flat_q_endpoint_is_exact_even_when_correction_kl_underflows(density_scale):
    q = jnp.zeros((1, 2), dtype=jnp.float32)
    y = jnp.array([[0., density_scale]], dtype=jnp.float32)
    beta, selected, budget = jax.jit(select_density_beta_for_kl)(q, y)
    np.testing.assert_array_equal(beta, [0.])
    np.testing.assert_array_equal(selected, [0.])
    np.testing.assert_array_equal(budget, [0.])


def test_large_scores_and_offsets_remain_finite():
    q = jnp.array([[-1000., 0.], [1000., 1001.]], dtype=jnp.float32)
    y = jnp.array([[2000., 0.], [-1000., 1000.]], dtype=jnp.float32)
    beta, selected, budget = select_density_beta_for_kl(q, y)
    assert np.all(np.isfinite(np.asarray(beta)))
    assert np.all(np.isfinite(np.asarray(selected)))
    assert np.all(np.isfinite(np.asarray(budget)))
    expected = np.asarray([reference_solution(a.copy(),b.copy()) for a,b in zip(np.asarray(q),np.asarray(y))])
    np.testing.assert_allclose(beta, expected[:, 0], rtol=2e-4, atol=1e-7)


def test_shift_invariance_for_exactly_representable_offsets():
    q = jnp.array([[0., .125, -.25, .5]], dtype=jnp.float32)
    y = jnp.array([[1., -2., 3., -4.]], dtype=jnp.float32)
    a = select_density_beta_for_kl(q, y)
    b = select_density_beta_for_kl(q + 1024., y - 2048.)
    for left, right in zip(a,b):
        np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-7)


def test_validation_and_single_candidate():
    with pytest.raises(ValueError):
        select_density_beta_for_kl(jnp.zeros((2, 3)), jnp.zeros((2, 4)))
    with pytest.raises(ValueError):
        select_density_beta_for_kl(jnp.zeros((2, 3)), jnp.zeros((2, 3)), 0)
    with pytest.raises(ValueError):
        select_density_beta_for_kl(jnp.zeros((2, 0)), jnp.zeros((2, 0)))
    with pytest.raises(ValueError):
        select_density_beta_for_kl(jnp.array(0.), jnp.array(0.))
    beta, selected, budget = select_density_beta_for_kl(jnp.ones((2, 1)), jnp.ones((2, 1)))
    np.testing.assert_array_equal(beta, [1., 1.])
    np.testing.assert_array_equal(selected, [0., 0.])
    np.testing.assert_array_equal(budget, [0., 0.])


def test_correction_kl_is_monotone_even_when_density_opposes_q():
    q = jnp.array([[0., 1., 2., 3.], [0., 1., 2., 3.], [0., .1, -.2, .3]])
    y = jnp.array([[0., 1., 2., 3.], [0., -1., -2., -3.], [1., -2., 4., -3.]])
    betas = jnp.linspace(0., 1., 129)
    kls = jax.jit(jax.vmap(lambda beta: _transport._exponential_tilt_kl(q, beta * y)))(betas)
    np.testing.assert_array_equal(kls[0], np.zeros(3))
    assert np.all(np.diff(np.asarray(kls), axis=0) >= -1e-7)
    beta, _, _ = select_density_beta_for_kl(q, y)
    # Increasing the density scale tightens the feasible beta by the same factor
    # for an interior solution; no assumption about ESS monotonicity is needed.
    stronger_beta, _, _ = select_density_beta_for_kl(q, 2. * y)
    interior = np.asarray(beta) < 1.
    assert np.any(interior)
    np.testing.assert_allclose(
        np.asarray(stronger_beta)[interior], np.asarray(beta)[interior] / 2.,
        rtol=2e-6, atol=2e-9,
    )


@pytest.mark.parametrize("beta", [1e-5, .02, .4, 1.])
def test_kl_derivative_equals_beta_times_density_variance(beta):
    q = jnp.array([.125, -.25, .5, -.75], dtype=jnp.float32)
    y = jnp.array([1., -2., 3., -4.], dtype=jnp.float32)
    derivative = jax.jit(jax.grad(lambda b: _transport._exponential_tilt_kl(q, b * y)))(beta)
    weights = jax.nn.softmax(q + beta * y)
    mean = jnp.sum(weights * y)
    expected = beta * jnp.sum(weights * jnp.square(y - mean))
    np.testing.assert_allclose(derivative, expected, rtol=2e-5, atol=1e-9)


def test_arbitrary_batch_axes_and_vmap_agree():
    q = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4) / 10.
    y = jnp.sin(jnp.arange(24, dtype=jnp.float32)).reshape(2, 3, 4)
    batched = jax.jit(select_density_beta_for_kl)(q, y)
    mapped = jax.jit(jax.vmap(jax.vmap(select_density_beta_for_kl)))(q, y)
    single = select_density_beta_for_kl(q[0, 0], y[0, 0])
    for all_rows, mapped_rows, first_row in zip(batched, mapped, single):
        assert all_rows.shape == (2, 3)
        assert first_row.shape == ()
        np.testing.assert_allclose(all_rows, mapped_rows, rtol=2e-6, atol=1e-8)
        np.testing.assert_allclose(all_rows[0, 0], first_row, rtol=2e-6, atol=1e-8)


def test_selector_has_finite_gradients_inside_jitted_loss():
    q = jnp.array([[.1, -.2, .3, -.4], [0., 0., 0., 0.], [0., 2., 4., 6.]])
    y = jnp.array([[1., -2., 3., -4.], [1., 1., 1., 1.], [0., .01, -.01, .02]])

    def loss(q_values, density_values):
        beta, selected_kl, budget = select_density_beta_for_kl(q_values, density_values)
        # The actor treats beta as a selected target, so no implicit derivative
        # of the bisection boundary is assumed here.
        weights = jax.nn.softmax(q_values + jax.lax.stop_gradient(beta)[:, None] * density_values)
        return jnp.sum(weights * q_values) + jnp.sum(selected_kl + budget)

    value, gradients = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))(q, y)
    assert np.isfinite(np.asarray(value))
    for gradient in gradients:
        assert gradient.shape == q.shape
        assert np.isfinite(np.asarray(gradient)).all()
    assert np.any(np.asarray(gradients[0]) != 0.)
