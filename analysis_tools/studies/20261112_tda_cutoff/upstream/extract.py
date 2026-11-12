"""Stage A (iBOLT venv, JAX on CPU): states + per-state Gaussian-mixture parameters from a checkpoint.

Usage (repo root): JAX_PLATFORMS=cpu python results/modal_em/extract.py <run_dir> <out.npz> [--states 512 --n 1024 --step 1000000]
States: zero-z evaluation policy (the paper's evaluation rule), episodes env.reset(seed=10000+i), pooled and
  uniformly subsampled (fixed seed) to --states.
Mixture at each state: z_1..z_N ~ N(0, I) (fixed key, same z for every state), component i =
  box-truncated diagonal Gaussian with center tanh(mu(s, z_i)) and clipped log sigma(s, z_i); weights pi_i = 1/N.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import flax.serialization
import gymnasium as gym
import jax, jax.numpy as jnp
import numpy as np
from train import compose_config
from ibolt.policy import IBOLTPolicy, actor_variables

p = argparse.ArgumentParser()
p.add_argument("run"); p.add_argument("out")
p.add_argument("--states", type=int, default=512)
p.add_argument("--n", type=int, default=1024)
p.add_argument("--step", type=int, default=1_000_000)
p.add_argument("--states-from", default=None, help="npz with 'states' (+ optional --idx file of indices): use these states")
p.add_argument("--idx", default=None)
a = p.parse_args()
run = Path(a.run); cfgj = json.loads((run / 'config.json').read_text())
ov = [f"benchmark={cfgj['task']}"]
if cfgj['alg']['critic'].get('type', 'default') == 'crossq':
    ov += ["+critic=crossq", f"alg.critic.hs=[{','.join(map(str, cfgj['alg']['critic']['hs']))}]"]
if cfgj['alg']['actor'].get('batch_renorm', False):
    ov += ["+actor=crossq"]
ov += [f"alg.actor.hidden_dims=[{','.join(map(str, cfgj['alg']['actor']['hidden_dims']))}]"]
cfg = compose_config(ov)
env = gym.make(cfg.env_name)
pol = IBOLTPolicy(env.observation_space, env.action_space, cfg=cfg)
pol.build(jax.random.PRNGKey(0), None, 3e-4)
cdir = next((run / 'checkpoints').glob(f'*/actor_state_{a.step}.msgpack')).parent
actor = flax.serialization.from_bytes(pol.actor_state, (cdir / f'actor_state_{a.step}.msgpack').read_bytes())
var = actor_variables(actor)
d = actor.params['mu']['bias'].shape[0]

@jax.jit
def act0(obs):
    return actor.apply_fn(var, obs, jnp.zeros((obs.shape[0], d), obs.dtype))[0]

@jax.jit
def mixture(obs, z):
    b, n = obs.shape[0], z.shape[0]
    o = jnp.broadcast_to(obs[:, None], (b, n, obs.shape[1])).reshape(b * n, -1)
    zz = jnp.broadcast_to(z[None], (b, n, d)).reshape(b * n, d)
    mu, ls = actor.apply_fn(var, o, zz)
    return mu.reshape(b, n, d), ls.reshape(b, n, d)

states, rets, i = [], [], 0
while a.states_from is None and len(states) < 4 * a.states:
    obs, _ = env.reset(seed=10_000 + i); done, R = False, 0.0
    while not done:
        states.append(obs.astype(np.float32))
        act = np.asarray(act0(jnp.asarray(obs[None], jnp.float32)))[0]
        obs, r, term, trunc, _ = env.step(pol.unscale_action(np.clip(act, -1, 1)))
        R += r; done = term or trunc
    rets.append(R); i += 1
if a.states_from is None:
    S = np.stack(states)
    idx = np.sort(np.random.default_rng(0).choice(len(S), a.states, replace=False))
    S = S[idx]
else:
    S = np.load(a.states_from)["states"]
    idx = np.loadtxt(a.idx, dtype=int) if a.idx else np.arange(len(S))
    S = S[idx].astype(np.float32); rets = [float("nan")]
z = np.asarray(jax.random.normal(jax.random.PRNGKey(12345), (a.n, d)), np.float32)
mus, lss = [], []
for k in range(0, len(S), 64):
    m, l = mixture(jnp.asarray(S[k:k + 64]), jnp.asarray(z))
    mus.append(np.asarray(m)); lss.append(np.asarray(l))
np.savez(a.out, states=S, src_idx=idx, mu=np.concatenate(mus), log_std=np.concatenate(lss), z=z,
         ep_returns=np.array(rets), run=run.name, task=cfgj['task'], T=cfgj['temperature'], seed=cfgj['seed'])
print(f"{run.name.split('_2026')[0]}: {len(rets)} eps, return {np.mean(rets):.0f}, states {S.shape}, D={d}, "
      f"sigma mean {np.exp(np.concatenate(lss)).mean():.4f}", flush=True)
