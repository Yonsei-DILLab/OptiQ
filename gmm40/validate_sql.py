"""CPU validation against independent NumPy/PyTorch equations, without TF1.

Run: JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' python -m gmm40.validate_sql
No environment training campaign or reference-distribution samples are used.
"""
from dataclasses import replace
import json
from pathlib import Path
import tempfile

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import torch

from .sql_jax import (SQLConfig, SQLLearner, adaptive_isotropic_gaussian_kernel,
                      actor_gradients, svgd_direction, td_targets, tf1_adam)


def check(actual, expected, atol=2e-5, rtol=2e-5):
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=atol, rtol=rtol)


def check_tree(actual, expected):
    a, ad = jax.tree_util.tree_flatten(actual)
    b, bd = jax.tree_util.tree_flatten(expected)
    assert ad == bd
    for x, y in zip(a, b):
        check(x, y)


def torch_net(params, obs, inputs, squash):
    x = obs @ params['observation']['kernel'] + inputs @ params['input']['kernel'] + params['first_bias']
    x = torch.relu(x)
    for key in sorted(k for k in params if k.startswith('hidden_')):
        x = torch.relu(x @ params[key]['kernel'] + params[key]['bias'])
    x = x @ params['output']['kernel'] + params['output']['bias']
    return torch.tanh(x) if squash else x[..., 0]


def main():
    torch.set_num_threads(1)
    rng = np.random.default_rng(781)
    cfg = SQLConfig(hidden_dims=(8, 8), kernel_particles=6, temperature=.4, target_update_interval=2)
    learner = SQLLearner(3, 2, seed=12, config=cfg)
    obs = rng.normal(size=(4, 3)).astype(np.float32)
    latents = rng.normal(size=(4, 6, 2)).astype(np.float32)
    fixed = rng.uniform(-.8, .8, size=(4, 4, 2)).astype(np.float32)
    updated = rng.uniform(-.8, .8, size=(4, 4, 2)).astype(np.float32)
    diffs = fixed[:, :, None] - updated[:, None]
    distances = (diffs ** 2).sum(-1)
    # Upstream top_k lower median, intentionally not NumPy median.
    h = np.maximum(np.sort(distances.reshape(4, -1))[:, 7] / np.log(4), .001)
    k = np.exp(-distances / h[:, None, None])
    kg = -2 * diffs * k[..., None] / h[:, None, None, None]
    result = adaptive_isotropic_gaussian_kernel(jnp.array(fixed), jnp.array(updated))
    check(result['bandwidth'], h)
    check(result['output'], k)
    check(result['gradient'], kg)
    floor = adaptive_isotropic_gaussian_kernel(jnp.zeros_like(fixed), jnp.zeros_like(updated))
    check(floor['bandwidth'], np.full(4, .001))
    check(floor['gradient'], np.zeros_like(kg))

    def q(s, a):
        return -.5 * ((a - .1 * s[..., :2]) ** 2).sum(-1)
    score = -(fixed - .1 * obs[:, None, :2]) / cfg.temperature - 2 * fixed / (1 - fixed ** 2 + 1e-6)
    expected_phi = (k[..., None] * score[:, :, None] + kg).mean(1)
    phi, _ = svgd_direction(q, jnp.array(obs), jnp.array(fixed), jnp.array(updated), cfg.temperature)
    check(phi, expected_phi)

    params = jax.tree_util.tree_map(lambda x: torch.tensor(np.array(x), requires_grad=True), learner.state.actor.params)
    ts, tz = torch.tensor(obs), torch.tensor(latents)
    actions = torch_net(params, ts[:, None], tz, True)
    check(learner.actor.apply({'params': learner.state.actor.params}, obs[:, None], latents), actions.detach())
    x, y = actions[:, :3].detach().requires_grad_(), actions[:, 3:]
    q_t = -.5 * ((x - .1 * ts[:, None, :2]) ** 2).sum(-1)
    score_t = torch.autograd.grad((q_t / cfg.temperature + torch.log(1 - x*x + 1e-6).sum(-1)).sum(), x)[0].detach()
    diff_t = x[:, :, None] - y[:, None]
    ds_t = (diff_t ** 2).sum(-1)
    h_t = torch.clamp(torch.topk(ds_t.reshape(4, -1), 5).values[:, -1] / np.log(3), min=.001).detach()
    k_t = torch.exp(-ds_t / h_t[:, None, None])
    phi_t = (k_t[..., None] * score_t[:, :, None] - 2*diff_t*k_t[..., None]/h_t[:, None, None, None]).mean(1).detach()
    (-torch.sum(y * phi_t)).backward()
    grads, _ = actor_gradients(learner.state.actor, jnp.array(obs), jnp.array(latents), q, cfg)
    references = jax.tree_util.tree_map(lambda x: x.grad.detach().numpy(), params)
    check_tree(grads, references)
    gradient_error = max(float(np.max(np.abs(np.asarray(a)-b))) for a, b in zip(jax.tree_util.tree_leaves(grads), jax.tree_util.tree_leaves(references)))

    # Tiny gradients make incorrect Adam epsilon placement visibly fail.
    tx = tf1_adam(3e-4)
    p = jnp.array([.1, -.2]); state = tx.init(p)
    p_ref = np.array(p); m = np.zeros(2); v = np.zeros(2)
    for step in range(1, 4):
        g = np.array([1e-9, -.3*step], np.float32)
        delta, state = tx.update(jnp.array(g), state)
        p = p + delta
        m = .9*m + .1*g; v = .999*v + .001*g*g
        p_ref -= 3e-4 * np.sqrt(1-.999**step)/(1-.9**step) * m / (np.sqrt(v)+1e-8)
        check(p, p_ref, atol=2e-7)

    next_q = rng.normal(size=(4, 7)).astype(np.float32)
    rewards = np.array([1., -3., 4., -2.], np.float32)
    terminals = np.array([0., 1., 0., 1.], np.float32)
    from scipy.special import logsumexp
    for temperature in (1., .25):
        c = replace(cfg, temperature=temperature)
        value = temperature * (logsumexp(next_q / temperature, axis=-1)-np.log(7)+5*np.log(2))
        check(td_targets(next_q, rewards, terminals, 5, c), rewards+(1-terminals)*.99*value)

    batch = dict(observations=obs, actions=rng.uniform(-1, 1, (4, 2)).astype(np.float32),
                 next_observations=obs*.9, rewards=rewards, terminals=terminals)
    # Independent critic gradient for the actual learner's first sampled backup.
    _, _, value_key = jax.random.split(learner.state.key, 3)
    values = jax.random.uniform(value_key, (1, cfg.value_particles, 2), minval=-1, maxval=1)
    cp = jax.tree_util.tree_map(lambda x: torch.tensor(np.array(x), requires_grad=True), learner.state.critic.params)
    nq = torch_net(cp, torch.tensor(batch['next_observations'])[:, None], torch.tensor(np.array(values)), False).detach()
    v = cfg.temperature * (torch.logsumexp(nq/cfg.temperature, -1)-np.log(cfg.value_particles)+2*np.log(2))
    ytd = torch.tensor(rewards)+(1-torch.tensor(terminals))*.99*v
    critic_loss = .5*torch.mean((ytd-torch_net(cp, ts, torch.tensor(batch['actions']), False))**2)
    critic_loss.backward()
    expected = learner.state.critic.apply_gradients(grads=jax.tree_util.tree_map(lambda x: jnp.array(x.grad.numpy()), cp))
    info = learner.update(batch, iteration=0)
    check(info['critic_loss'], critic_loss.detach())
    check_tree(learner.state.critic.params, expected.params)
    check_tree(learner.state.target_critic, learner.state.critic.params)
    previous_target = learner.state.target_critic
    learner.update(batch, iteration=1)
    check_tree(learner.state.target_critic, previous_target)
    learner.update(batch, iteration=2)
    check_tree(learner.state.target_critic, learner.state.critic.params)

    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp)/'checkpoint.bin'
        learner.save(path)
        restored = SQLLearner(3, 2, seed=99, config=cfg)
        restored.restore(path)
        learner.update(batch, iteration=3); restored.update(batch, iteration=3)
        assert flax.serialization.to_bytes(learner.state) == flax.serialization.to_bytes(restored.state)
        mismatch = SQLLearner(3, 2, config=replace(cfg, temperature=1.))
        try:
            mismatch.restore(path)
        except ValueError:
            pass
        else:
            raise AssertionError('Configuration mismatch was accepted')

        # Fixed-Q adapter checkpoint, bounded stochastic sampling, no reference samples.
        from .sql import SQL, SQLOnline
        class Energy:
            def jax_log_prob(self, x):return -.5*jnp.sum((x/10)**2, -1)
        fixed = SQL(Energy(), seed=5, batch=4, config=cfg)
        fixed.advance(2)
        key_before = np.array(fixed.learner.state.key)
        sample = fixed.evaluate_samples(64, 7)[0]
        assert sample.shape == (64, 2) and np.isfinite(sample).all() and np.max(np.abs(sample)) <= 40
        check(fixed.learner.state.key, key_before, atol=0, rtol=0)
        fixed.save(path)
        restored_fixed = SQL(Energy(), seed=99, batch=4, config=cfg)
        restored_fixed.restore(path)
        fixed.advance(1); restored_fixed.advance(1)
        assert flax.serialization.to_bytes(fixed.learner.state) == flax.serialization.to_bytes(restored_fixed.learner.state)

        import gymnasium as gym
        from types import SimpleNamespace
        env = SimpleNamespace(observation_space=gym.spaces.Box(-1., 1., (3,)), action_space=gym.spaces.Box(-1., 1., (2,)))
        online = SQLOnline(env, seed=1, batch=4, config=cfg, capacity=5)
        for i in range(7):online.store(obs[i % 4], [0., .1], float(i), obs[(i+1) % 4], i % 3 == 0)
        online.update()
        state_before = flax.serialization.to_bytes(online.learner.state)
        online.eval_key = jax.random.PRNGKey(111)
        online.act(obs); online.q(obs, batch['actions'])
        assert state_before == flax.serialization.to_bytes(online.learner.state)
        online.eval_key = None
        online.save(path)
        online2 = SQLOnline(env, seed=99, batch=4, config=cfg, capacity=5)
        online2.restore(path)
        check(online.act(obs), online2.act(obs), atol=0, rtol=0)
        online.update(); online2.update()
        assert flax.serialization.to_bytes(online.learner.state) == flax.serialization.to_bytes(online2.learner.state)
        assert online.rng.bit_generator.state == online2.rng.bit_generator.state

    for invalid in (dict(kernel_particles=2), dict(kernel_update_ratio=0.), dict(temperature=0.), dict(value_particles=0)):
        try:SQLConfig(**invalid)
        except ValueError:pass
        else:raise AssertionError(invalid)
    print(json.dumps(dict(status='passed', actor_gradient_max_abs_error=gradient_error,
                         checks=['kernel lower median/floor', 'bounded-action score', 'SVGD parameter gradient vs PyTorch',
                                 'TF1 Adam formula', 'soft TD target and critic gradient', 'hard target cadence',
                                 'checkpoint deterministic continuation', 'evaluation RNG isolation', 'replay wrap/restore',
                                 'configuration guards'],
                         limitation='Independent NumPy/PyTorch equation parity; original TensorFlow runtime not executed'), indent=2))


if __name__ == '__main__':
    main()
