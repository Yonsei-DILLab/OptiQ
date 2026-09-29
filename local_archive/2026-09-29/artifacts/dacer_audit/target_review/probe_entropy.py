"""Read-only CPU checkpoint diagnostic; never initializes a trainer or logger.

Uses the frozen TRG actor and exact DACER GMM proxy (K=3, full covariance,
random_state=42, 200 actions/state). Stored warmup probe states compare training
checkpoints; separate final-policy rollouts check sensitivity to state choice.
This is a seed-0 diagnostic, not a reward ablation or target selection experiment.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])

COMMIT = '84f1e0a884349d6c4b0dae521839a8d4e5f46437'
SOURCE = Path('/home/heechan/OptiQ-ops/sources') / COMMIT
sys.path.insert(0, str(SOURCE / 'analysis_tools/studies/20260918_nonstationary_nd/v5'))
sys.path.insert(0, str(SOURCE / 'analysis_tools/experiments/20260920_truncated_mll'))
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax.training.train_state import TrainState
from sklearn.mixture import GaussianMixture
from optiq_dime.policy import SemiImplicitActor, OptiQPolicy
from optiq_dime.semi_implicit import actor_components


def proxy(actions):
    hs, converged = [], []
    for x in np.asarray(actions, dtype=np.float64):
        g = GaussianMixture(n_components=3, covariance_type='full', random_state=42).fit(x)
        sign, ld = np.linalg.slogdet(g.covariances_)
        assert np.all(sign > 0)
        h = -np.dot(g.weights_, np.log(g.weights_)) + np.dot(
            g.weights_, .5 * (x.shape[-1] * (1 + np.log(2*np.pi)) + ld))
        hs.append(float(h)); converged.append(bool(g.converged_))
    return np.asarray(hs), float(np.mean(converged))


@jax.jit
def draw(actor, obs, keys):
    return jax.vmap(lambda k: OptiQPolicy.sample_action(actor, obs, k), out_axes=1)(keys)


def measure(actor, obs, task, label, rng_seed):
    dim = actor.params['mu']['bias'].shape[0]
    actions = np.asarray(draw(actor, jnp.asarray(obs), jax.random.split(jax.random.PRNGKey(rng_seed), 200)))
    mu, ls = jax.jit(lambda st, o: actor_components(st, o, jax.random.PRNGKey(612), 64))(actor, jnp.asarray(obs))
    ls = np.asarray(ls); mu = np.asarray(mu)
    result = {'state_source': label, 'states': len(obs), 'actions_per_state': 200,
              'action_dim': dim, 'conditional_sigma_mean': float(np.exp(ls).mean()),
              'conditional_log_sigma_mean': float(ls.mean()),
              'latent_mu_std_mean': float(mu.std(axis=1).mean()), 'noise': []}
    eps = np.random.default_rng(rng_seed).normal(size=actions.shape)
    initial_std = .27 * (.15 if task == 'humanoid' else .1)
    for std in (0., initial_std, .1):
        noisy = np.clip(actions + std*eps, -1., 1.)
        hs, conv = proxy(noisy)
        result['noise'].append({'std': std, 'proxy_per_dim_mean': float(hs.mean()/dim),
            'proxy_per_dim_state_sd': float(hs.std(ddof=1)/dim),
            'proxy_per_dim_state_quantiles': np.quantile(hs/dim, [.1,.5,.9]).tolist(),
            'state_fraction_below_minus09': float(np.mean(hs/dim < -.9)),
            'gmm_converged_fraction': conv, 'clip_fraction': float(np.mean(np.abs(noisy)>=1.)),
            'state_proxy_per_dim': (hs/dim).tolist()})
    if label == 'final_policy_visited':
        bigger = np.asarray(draw(actor, jnp.asarray(obs[:8]), jax.random.split(jax.random.PRNGKey(rng_seed+1), 2000)))
        hs, conv = proxy(bigger)
        result['sample_size_check'] = {'states': 8, 'actions_per_state': 2000,
            'proxy_per_dim_mean': float(hs.mean()/dim), 'gmm_converged_fraction': conv,
            'same_first8_states_200_mean': float(np.mean(result['noise'][0]['state_proxy_per_dim'][:8]))}
    return result


def visited(actor, env_name):
    env = gym.make(env_name)
    obs, _ = env.reset(seed=92721)
    observations = []
    step_action = jax.jit(OptiQPolicy.sample_action)
    key = jax.random.PRNGKey(92721)
    episodes = 0
    for i in range(1200):
        observations.append(obs.copy())
        key, sub = jax.random.split(key)
        a = np.asarray(step_action(actor, jnp.asarray(obs[None], dtype=jnp.float32), sub))[0]
        a = env.action_space.low + (a+1.)*.5*(env.action_space.high-env.action_space.low)
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            episodes += 1
            obs, _ = env.reset(seed=92721+episodes)
    env.close()
    indices = np.linspace(0, len(observations)-1, 32, dtype=int)
    return np.asarray(observations, dtype=np.float32)[indices], episodes


def main():
    p = argparse.ArgumentParser()
    p.add_argument('task', choices=['ant','humanoid'])
    p.add_argument('--output', required=True)
    args = p.parse_args()
    assert all(d.platform == 'cpu' for d in jax.devices())
    root = Path('/home/heechan/optiq-experiments/trg-temp-beta-20260921/outputs')
    record = {'task': args.task, 'source_commit': COMMIT, 'training_seed': 0,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'started_unix': time.time(), 'results': [], 'complete': False}
    for run in sorted(root.glob(args.task+'-trg-temperature-*-s0_*')):
        cfg = json.loads((run/'config.json').read_text())
        assert cfg['runtime']['git_commit'] == COMMIT
        a = cfg['alg']['actor']
        assert a.get('latent_prior', 'normal') == 'normal'
        batch = np.load(next(run.glob('checkpoints/*/landscape_probe_batch.npz')))
        print('PROBE_KEYS', run.name, {k: batch[k].shape for k in batch.files}, flush=True)
        obs_key = next(k for k in batch.files if k in ('observations','obs'))
        all_obs = batch[obs_key]
        indices = np.linspace(0, len(all_obs)-1, 32, dtype=int)
        obs = all_obs[indices].astype(np.float32)
        for step in (5001, 100000, 1000000):
            checkpoint = next(run.glob(f'checkpoints/*/actor_state_{step}.msgpack'))
            payload = checkpoint.read_bytes()
            raw = serialization.msgpack_restore(payload)
            dim = raw['params']['mu']['bias'].shape[0]
            net = SemiImplicitActor(dim, tuple(a['hidden_dims']), a['log_std_min'],
                a['log_std_max'], a['initial_log_std'], a['mean_output_init_scale'],
                a['log_std_output_init_scale'], a.get('mean_latent_skip_scale', 0.))
            actor = TrainState.create(apply_fn=net.apply, params=raw['params'], tx=optax.identity())
            entry = {'run': run.name, 'temperature': a['temperature'], 'step': step,
                     'checkpoint': str(checkpoint), 'checkpoint_sha256': hashlib.sha256(payload).hexdigest(),
                     'measurements': [measure(actor, obs, args.task, 'stored_probe', 4200)]}
            if step == 1000000:
                vis, episodes = visited(actor, cfg['env_name'])
                entry['visited_rollout_steps'] = 1200
                entry['visited_completed_episodes'] = episodes
                entry['measurements'].append(measure(actor, vis, args.task, 'final_policy_visited', 4310))
            record['results'].append(entry)
            Path(args.output).write_text(json.dumps(record, indent=2)+'\n')
            print(json.dumps({'run': run.name, 'step': step, 'measurements': [
                {'states': m['state_source'], 'sigma_mean': m['conditional_sigma_mean'],
                 'h_per_dim': [n['proxy_per_dim_mean'] for n in m['noise']]}
                for m in entry['measurements']]}), flush=True)
    assert len(record['results']) == 6
    record.update(complete=True, finished_unix=time.time())
    Path(args.output).write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    main()
