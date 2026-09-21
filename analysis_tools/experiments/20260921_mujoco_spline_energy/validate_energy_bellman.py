"""Targeted checks for the new loss, its proofs, and the real update path.

Run on CPU. Writes a local provenance receipt; creates no W&B run.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from scipy.special import logsumexp

from algorithms.spline_energy.energy_bellman import (
    bounds_from_uniform_residual, relative_energy_loss,
    residual_bound_from_uniform_loss,
)
from circuit import ConditionalSplineCircuit, log_prob_from_output
from train import make_update


def check_scalar_loss():
    eta, kappa = 2., 4.
    d = jnp.asarray([-8., -2., -.1, 0., .1, 2., 7.], dtype=jnp.float32)
    y = jnp.full_like(d, 3.)
    u, v = jnp.exp(y / eta), jnp.exp((y + d) / eta)
    kl = eta**2 * (u * jnp.log(u / v) - u + v) / u
    np.testing.assert_allclose(relative_energy_loss(d, eta, kappa), kl,
                               atol=3e-5, rtol=2e-5)
    grad = jax.grad(lambda x: relative_energy_loss(x, eta, kappa))
    curvature = jax.grad(grad)
    x = jnp.asarray([-1e6, -100., -4., 0., 4., 8., 10., 100., 1e6])
    vals = np.asarray(relative_energy_loss(x, eta, kappa))
    grads = np.asarray(jax.vmap(grad)(x))
    curves = np.asarray(jax.vmap(curvature)(x))
    assert np.isfinite(vals).all() and np.isfinite(grads).all()
    assert (vals >= 0).all() and float(grad(0.)) == 0.
    assert np.all(grads[x < 0] < 0) and np.all(grads[x > 0] > 0)
    assert np.all(grads >= -eta - 1e-5)
    assert np.all(curves >= 0) and np.max(curves) <= math.exp(kappa) + 1e-4
    np.testing.assert_allclose(curvature(eta * kappa), math.exp(kappa), rtol=1e-6)
    assert abs(float(grad(eta * kappa - 1e-4) - grad(eta * kappa + 1e-4))) < .02
    for magnitude in [0., .001, .1, 1., 10., 100.]:
        for sign in [-1., 1.]:
            loss = float(relative_energy_loss(jnp.asarray(sign * magnitude), eta, kappa))
            bound = residual_bound_from_uniform_loss(loss, eta)
            assert magnitude <= bound + 2e-4 * max(1., magnitude)
    return {"max_checked_abs_residual": float(np.max(np.abs(x))),
            "max_checked_scalar_curvature": float(np.max(curves)),
            "curvature_bound": math.exp(kappa)}


def check_bellman_bounds():
    rng = np.random.default_rng(33)
    n, a, gamma, alpha = 4, 5, .93, .25
    p = rng.uniform(size=(n, a, n)); p /= p.sum(-1, keepdims=True)
    p *= rng.uniform(.4, 1., size=(n, a, 1))  # missing mass is true termination
    reward = rng.uniform(-2., 3., size=(n, a))
    value = lambda q: alpha * logsumexp(q / alpha, axis=-1)
    backup = lambda q: reward + gamma * np.einsum('sat,t->sa', p, value(q))
    qstar = np.zeros_like(reward)
    for _ in range(2000):
        new = backup(qstar)
        if np.max(np.abs(new - qstar)) < 1e-13:
            qstar = new; break
        qstar = new
    assert np.max(np.abs(qstar - backup(qstar))) < 1e-11
    q = qstar + rng.normal(size=qstar.shape)
    pi = np.exp((q - value(q)[:, None]) / alpha)
    pi_star = np.exp((qstar - value(qstar)[:, None]) / alpha)
    ppi = np.einsum('sa,sat->st', pi, p)
    rhs = np.sum(pi * (reward - alpha * np.log(pi)), axis=-1)
    vpi = np.linalg.solve(np.eye(n) - gamma * ppi, rhs)
    qpi = reward + gamma * np.einsum('sat,t->sa', p, vpi)
    residual = float(np.max(np.abs(q - backup(q))))
    bounds = bounds_from_uniform_residual(residual, gamma, alpha)
    qerr = float(np.max(np.abs(q - qstar)))
    gap = value(qstar) - vpi
    kl_f = np.sum(pi * np.log(pi / pi_star), axis=-1)
    kl_r = np.sum(pi_star * np.log(pi_star / pi), axis=-1)
    assert qerr <= bounds['q_sup_error_bound'] + 1e-9
    assert np.max(np.abs(q - qpi)) <= bounds['q_sup_error_bound'] + 1e-9
    assert gap.min() >= -1e-9
    assert gap.max() <= bounds['soft_value_suboptimality_bound'] + 1e-9
    assert max(kl_f.max(), kl_r.max()) <= bounds['policy_kl_either_direction_bound']
    contraction = float(np.max(np.abs(backup(q) - backup(qstar))) / qerr)
    assert contraction <= gamma + 1e-10
    # The Gibbs identity alone does not imply optimality: a perturbed Q still
    # defines an exact Gibbs policy but has a positive policy value gap.
    assert gap.max() > .01
    return dict(residual=residual, q_error=qerr, policy_value_gap=float(gap.max()),
                contraction_ratio=contraction, gamma=gamma, **bounds)


def check_stochastic_bias():
    eta, kappa = 3., 4.

    def risk(y, p):
        lo, hi = np.min(y, axis=-1), np.max(y, axis=-1)
        for _ in range(100):
            mid = (lo + hi) / 2
            x = (mid[..., None] - y) / eta
            base = np.minimum(x, kappa)
            g = np.expm1(base) + math.exp(kappa) * np.maximum(x-kappa, 0)
            score = np.sum(p*g, axis=-1)
            lo, hi = np.where(score < 0, mid, lo), np.where(score >= 0, mid, hi)
        return (lo + hi) / 2

    y = np.array([[-2., 0., 3.], [-80., 0., 50.], [7., 7., 7.]])
    p = np.array([[.2, .5, .3], [.01, .69, .3], [.2, .3, .5]])
    fitted = risk(y, p)
    mean = np.sum(p*y, axis=-1)
    entropic = -eta * logsumexp(np.log(p) - y / eta, axis=-1)
    width = np.ptp(y, axis=-1)
    assert np.all(fitted >= entropic - 1e-9) and np.all(fitted <= mean + 1e-9)
    assert np.all(mean - fitted <= width**2 / (8*eta) + 1e-9)
    np.testing.assert_allclose(fitted[-1], 7., atol=1e-12)
    np.testing.assert_allclose(risk(y+11, p), fitted+11, atol=1e-10)
    perturbed = y + np.array([[.7, -.4, .2]])
    assert np.all(np.abs(risk(perturbed, p)-fitted) <= .7 + 1e-9)
    # Compose this M-estimator with a soft backup, including terminal outcomes.
    rng = np.random.default_rng(9); gamma, alpha = .9, .25
    weights = rng.uniform(size=(3, 4, 4)); weights /= weights.sum(-1, keepdims=True)
    rewards = rng.uniform(-1., 1., size=(3, 4, 4))
    def backup(q):
        v = alpha * logsumexp(q/alpha, axis=-1)
        next_v = np.concatenate([v, [0.]])
        return risk(rewards + gamma*next_v, weights)
    q0 = rng.normal(size=(3, 4)); q1 = q0 + rng.normal(size=(3, 4))
    ratio = np.max(np.abs(backup(q1)-backup(q0))) / np.max(np.abs(q1-q0))
    assert ratio <= gamma + 1e-10
    return {"ordinary_mean": mean.tolist(), "entropic_softmin": entropic.tolist(),
            "continued_loss_minimizer": fitted.tolist(),
            "hoeffding_bias_bound": (width**2/(8*eta)).tolist(),
            "risk_bellman_contraction_ratio": float(ratio)}


def check_training_path():
    model = ConditionalSplineCircuit(action_dim=2, rank=4, knots=17, hidden_dims=(32, 32))
    key = jax.random.PRNGKey(5)
    key, ok, ak = jax.random.split(key, 3)
    obs = jax.random.normal(ok, (128, 4))
    actions = jax.random.uniform(ak, (128, 2), minval=-1., maxval=1.)
    rewards = 2. - 3.*jnp.sum((actions - .4*jnp.tanh(obs[:, :2]))**2, axis=-1)
    batch = dict(obs=obs, actions=actions, rewards=rewards, next_obs=obs,
                 not_terminal=jnp.zeros(128))
    params = model.init(key, obs)['params']
    state = TrainState.create(apply_fn=model.apply, params=params,
                              tx=optax.chain(optax.clip_by_global_norm(10.), optax.adam(1e-3)))
    update = make_update(model, .25, .99, .005, 10., 'relative_energy', 10., 4.)
    target = params
    @jax.jit
    def loss(p):
        out = model.apply({'params':p}, obs)
        q = out['value'] + .25*log_prob_from_output(out, actions)
        return relative_energy_loss(q-rewards, 10., 4.).mean()
    initial_loss = float(loss(params))
    for _ in range(128):
        state, target, metrics = update(state, target, batch)
    final_loss = float(loss(state.params))
    assert final_loss < initial_loss*.75, (initial_loss, final_loss)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves((state, target, metrics)))
    # Actual target parameters must not receive gradients through the TD target.
    live_batch = dict(batch, not_terminal=jnp.ones(128))
    target_grad = jax.grad(lambda tp: update(state, tp, live_batch)[2]['train/loss'])(target)
    assert max(float(jnp.max(jnp.abs(x))) for x in jax.tree_util.tree_leaves(target_grad)) == 0.
    assert int(state.step) == 128
    return {'initial_loss':initial_loss, 'final_loss':final_loss, 'updates':128,
            'target_gradient_max':0., 'model_unchanged':True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    record = {'passed':True, 'scalar_loss':check_scalar_loss(),
              'bellman_bounds':check_bellman_bounds(),
              'stochastic_bias':check_stochastic_bias(),
              'training_path':check_training_path(),
              'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'command':sys.argv,
              'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in [Path(__file__), ROOT/'algorithms/spline_energy/energy_bellman.py',
                           ROOT/'algorithms/spline_energy/model.py', Path(__file__).with_name('train.py')]}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__ == '__main__':
    main()
