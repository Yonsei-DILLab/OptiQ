"""Pure inference from saved Flax trees; no training code or mutable statistics."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import jax
import jax.numpy as jnp
import msgpack
import numpy as np

jax.config.update('jax_threefry_partitionable', False)


def decode(code, data):
    if code in (1, 3):
        shape, dtype, contents = msgpack.unpackb(data, raw=True)
        value = np.frombuffer(contents, dtype=np.dtype(dtype)).reshape(shape).copy()
        return value if code == 1 else value[()]
    if code == 2:
        return complex(*msgpack.unpackb(data))
    raise ValueError(f'Unsupported msgpack extension {code}')


def load_actor(path):
    tree = msgpack.unpackb(path.read_bytes(), ext_hook=decode, raw=False, strict_map_key=False)
    # Flax restoration produces JAX leaves. Keep arithmetic inside the same JIT
    # graph; NumPy constants and nested JIT boundaries can change fp32 folding.
    return tuple(jax.tree_util.tree_map(jnp.asarray,tree[key]) for key in ('params','batch_stats'))


def forward(params, stats, obs, z):
    def norm(x, name):
        p, s = params[name], stats[name]
        return (x - s['mean']) * jax.lax.rsqrt(s['var'] + .001) * p['scale'] + p['bias']
    x = norm(jnp.concatenate((obs, z), axis=-1), 'BatchRenorm_0')
    for i in range(2):
        d = params[f'Dense_{i}']
        x = norm(jax.nn.gelu(x @ d['kernel'] + d['bias'], approximate=True), f'BatchRenorm_{i+1}')
    mean = jnp.tanh(x @ params['mu']['kernel'] + params['mu']['bias'])
    ls = jnp.clip(x @ params['log_std']['kernel'] + params['log_std']['bias'], -5., -1.)
    return mean, ls


def evaluate(params, stats, states, z, batch=4096):
    # Upstream extract.py closes over saved variables before jitting. Preserve
    # that constant-folding graph rather than supplying parameters dynamically.
    captured = jax.jit(lambda obs, zz: forward(params, stats, obs, zz))
    means, logstds = [], []
    for state in states:
        mm, ll = [], []
        for start in range(0, len(z), batch):
            zz = z[start:start+batch]
            obs = np.broadcast_to(state, (len(zz), len(state))).copy()
            m, l = captured(jnp.asarray(obs), jnp.asarray(zz))
            mm.append(np.asarray(m)); ll.append(np.asarray(l))
        means.append(np.concatenate(mm)); logstds.append(np.concatenate(ll))
    return np.stack(means), np.stack(logstds)


def locate(root, env, seed):
    files = list(root.glob(f'**/{env}-ibolt-crossqAC-*-s{seed}_*/checkpoints/**/actor_state_1000000.msgpack'))
    if len(files) != 1:
        raise ValueError((env, seed, files))
    return files[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoints', type=Path, required=True)
    p.add_argument('--core', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--commit', required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(__file__).with_name('phase_a.json').read_text())
    provenance = {'source_commit':a.commit, 'jax':jax.__version__, 'numpy':np.__version__, 'fixtures':[]}
    for seed in range(5):
        path = locate(a.checkpoints, 'hopper', seed)
        params, stats = load_actor(path)
        with np.load(a.core / 'data' / f'hopper_s{seed}.npz') as f:
            idx = np.arange(len(f['states']))
            obs, z = f['states'][idx], f['z']
            m, ls = evaluate(params, stats, obs, z)
            errors = dict(mu=float(np.max(abs(m-f['mu'][idx]))), log_std=float(np.max(abs(ls-f['log_std'][idx]))))
            m2, l2 = evaluate(params, stats, obs[:1], z[:64], batch=17)
            errors['batch_mu'] = float(np.max(abs(m2-m[:1,:64])))
            errors['batch_log_std'] = float(np.max(abs(l2-ls[:1,:64])))
            z_original = np.asarray(jax.random.normal(jax.random.PRNGKey(config['latent_seed']), z.shape), np.float32)
            errors['latent'] = float(np.max(abs(z-z_original)))
        item = dict(seed=seed, checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), errors=errors)
        provenance['fixtures'].append(item)
        print('FIXTURE', json.dumps(item), flush=True)
        (a.out / 'restoration.json').write_text(json.dumps(provenance, indent=2)+'\n')
        if max(errors.values()) > config['fixture_atol']:
            raise AssertionError(f'Checkpoint restoration mismatch: {item}')
    provenance['fixture_gate_passed'] = True
    provenance['cases'] = []
    for case in config['cases']:
        env, seed, state = case['env'], case['seed'], case['state']
        path = locate(a.checkpoints, env, seed)
        params, stats = load_actor(path)
        with np.load(a.core / 'states' / f'{env}_s{seed}_states.npz') as f:
            obs = f['states'][state:state+1]
        dim = params['mu']['bias'].shape[0]
        z = np.asarray(jax.random.normal(jax.random.PRNGKey(config['latent_seed']), (config['n'], dim)), np.float32)
        m, ls = evaluate(params, stats, obs, z)
        name = f'{env}_s{seed}_state{state}_n{config["n"]}.npz'
        np.savez(a.out/name, mu=m, log_std=ls, states=obs, z=z, original_state=state, seed=seed, env=env)
        provenance['cases'].append(dict(case, file=name, sha256=hashlib.sha256((a.out/name).read_bytes()).hexdigest()))
        print('EXTRACTED',name,flush=True)
    (a.out / 'restoration.json').write_text(json.dumps(provenance, indent=2)+'\n')


if __name__ == '__main__':
    main()
