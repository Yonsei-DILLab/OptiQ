"""Independent NumPy arithmetic for porting final v2; not a training backend.

No repository/JAX imports. Random draws and network outputs are inputs, so a
second implementation can compare numerical results without matching PRNGs.
Run `python docs/v2/reference_math.py` for local identity checks.
"""
import numpy as np


def logsumexp(x, axis=-1, keepdims=False):
    maximum = np.max(x, axis=axis, keepdims=True)
    result = maximum + np.log(np.exp(x - maximum).sum(axis=axis, keepdims=True))
    return result if keepdims else np.squeeze(result, axis=axis)


def tanh_log_jacobian(u):
    return (2 * (np.log(2) - u - np.logaddexp(0, -2 * u))).sum(axis=-1)


def log_mixture(u, mu, log_std):
    """[B,K,D], [B,M,D], [B,M,D] -> joint action log density [B,K]."""
    delta = (u[:, :, None, :] - mu[:, None, :, :]) * np.exp(-log_std[:, None, :, :])
    components = (-0.5 * delta**2 - log_std[:, None, :, :]
                  - 0.5 * np.log(2 * np.pi)).sum(axis=-1)
    return logsumexp(components) - np.log(mu.shape[1]) - tanh_log_jacobian(u)


def teacher_from_draws(mu, log_std, indices, noise, floor=0.05):
    """Generate teacher samples and matching density using supplied base draws."""
    scales = np.maximum(log_std, np.log(floor))
    rows = np.arange(mu.shape[0])[:, None]
    v = mu[rows, indices] + np.exp(scales[rows, indices]) * noise
    return np.tanh(v), v, log_mixture(v, mu, scales)


def importance_weights(q1, q2, log_q, temperature=0.1):
    logits = (q1 + q2) / (2 * temperature) - log_q
    weights = np.exp(logits - logsumexp(logits, keepdims=True))
    return weights, 1 / (weights**2).sum(axis=-1)


def sinkhorn(cost, weights, epsilon=0.25, iterations=100):
    batch, rows, _ = cost.shape
    columns = np.clip(weights, 1e-20, 1)
    columns = columns / columns.sum(axis=-1, keepdims=True)
    log_rows = np.full((batch, rows), -np.log(rows))
    log_columns = np.log(columns)
    kernel = -cost / epsilon
    u, v = np.zeros_like(log_rows), np.zeros_like(log_columns)
    for _ in range(iterations):
        u = log_rows - logsumexp(kernel + v[:, None, :], axis=-1)
        v = log_columns - logsumexp(kernel + u[:, :, None], axis=-2)
    return np.exp(kernel + u[:, :, None] + v[:, None, :])


def row_probabilities(plan):
    return plan / np.maximum(plan.sum(axis=-1, keepdims=True), 1e-20)


def conditional_nll_and_gradients(mu, log_std, teacher_u, rows):
    """NLL + analytic derivatives wrt mu/log_std (before log_std clipping).

    Gradients through stopped teacher_u/rows are deliberately not defined.
    This assumes each row sums to one, as after row_probabilities().
    """
    mean = np.einsum('bnk,bkd->bnd', rows, teacher_u)
    variance = np.maximum(np.einsum('bnk,bkd->bnd', rows, teacher_u**2) - mean**2, 0)
    error = (mu - mean)**2 + variance
    precision = np.exp(-2 * log_std)
    loss = (0.5 * error * precision + log_std + 0.5 * np.log(2 * np.pi)).sum(-1).mean()
    normalizer = mu.shape[0] * mu.shape[1]
    dmu = (mu - mean) * precision / normalizer
    dlog_std = (1 - error * precision) / normalizer
    return loss, dmu, dlog_std


def direct_conditional_nll(mu, log_std, teacher_u, rows):
    delta = (teacher_u[:, None, :, :] - mu[:, :, None, :]) * np.exp(-log_std[:, :, None, :])
    nll = (0.5 * delta**2 + log_std[:, :, None, :] + 0.5 * np.log(2 * np.pi)).sum(-1)
    return (rows * nll).sum(-1).mean()


def entropy_bracket_from_draws(mu, log_std, eps):
    """17 component outputs; use 0..15 lower and 1..16 upper, same action."""
    u = mu[:, 0] + np.exp(log_std[:, 0]) * eps
    lower = -log_mixture(u[:, None], mu[:, :-1], log_std[:, :-1])[:, 0]
    upper = -log_mixture(u[:, None], mu[:, 1:], log_std[:, 1:])[:, 0]
    return np.tanh(u), lower, upper


def soft_target(reward, terminal, target_q1, target_q2, log_g, gamma=.99, temperature=.1):
    return reward + gamma * (1 - terminal) * (np.minimum(target_q1, target_q2) - temperature * log_g)


def critic_loss(q1, q2, target):
    return ((q1 - target)**2).mean() + ((q2 - target)**2).mean()


def guard_decision(gaps):
    states = gaps.mean(axis=0)
    mean = states.mean()
    se = states.std(ddof=1) / np.sqrt(states.size)
    margin = mean - 0.0 * se
    return bool(np.isfinite(gaps).all() and np.isfinite(margin) and margin > 0), mean, se


def self_check():
    rng = np.random.default_rng(20260911)
    mu = rng.normal(size=(2, 16, 3)) * .2
    ls = np.full_like(mu, np.log(.5))
    indices = rng.integers(16, size=(2, 64))
    a, u, log_q = teacher_from_draws(mu, ls, indices, rng.normal(size=(2, 64, 3)))
    q1, q2 = rng.normal(size=(2, 64)) * .01, rng.normal(size=(2, 64)) * .01
    w, ess = importance_weights(q1, q2, log_q)
    student = np.tanh(mu + np.exp(ls) * rng.normal(size=mu.shape))
    cost = ((student[:, :, None] - a[:, None])**2).sum(-1)
    plan = sinkhorn(cost, w)
    rows = row_probabilities(plan)
    loss, dmu, dls = conditional_nll_and_gradients(mu, ls, u, rows)
    np.testing.assert_allclose(loss, direct_conditional_nll(mu, ls, u, rows), rtol=1e-12)
    np.testing.assert_allclose(plan.sum(1), w, atol=1e-12)
    # The clipped conditional scale is inactive at this test point.
    for array, grad in [(mu, dmu), (ls, dls)]:
        index = (0, 3, 1); original = array[index]; h = 1e-5
        array[index] = original + h; plus = direct_conditional_nll(mu, ls, u, rows)
        array[index] = original - h; minus = direct_conditional_nll(mu, ls, u, rows)
        array[index] = original
        np.testing.assert_allclose((plus-minus)/(2*h), grad[index], rtol=1e-7, atol=1e-9)
    assert guard_decision(np.ones((8, 32)))[0]
    assert not guard_decision(np.zeros((8, 32)))[0]
    assert not guard_decision(-np.ones((8, 32)))[0]
    assert np.isfinite(tanh_log_jacobian(np.array([[-100., 100.]]))).all()
    assert np.all((ess >= 1) & (ess <= 64))
    print('PASS: density, OT column masses, full NLL moments/finite differences, guard boundaries.')


if __name__ == '__main__':
    self_check()
