"""iBOLT with Gamid/AEAP-style critic selection at inference (best-of-K), from saved checkpoints.

Usage (repo root, iBOLT venv): python results/best_of_k/bok_eval_ibolt.py <run_dir> --step 1000000 --episodes 20
Modes per checkpoint, all on the same evaluation episodes (env.reset(seed=10000 + i)):
  z0   : a = tanh(mu(s, 0))                      (paper evaluation rule)
  k1   : a = tanh(mu(s, z)), z ~ N(0, I)          (one random latent, no critic)
  kK   : z_1..z_K ~ N(0, I), a_k = tanh(mu(s, z_k)), pick argmax_k min_j Q_j(s, a_k)
         (twin online critics, train=False) -- K candidates in one batched actor + critic call.
Writes one JSON per run to --out.
"""
import argparse, json, sys, time
from functools import partial
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import flax.serialization
import gymnasium as gym
import jax, jax.numpy as jnp
import numpy as np
from train import compose_config
from ibolt.policy import IBOLTPolicy, actor_variables


@partial(jax.jit, static_argnames=["k"])
def act(actor, critic, obs, key, k):
    d = actor.params['mu']['bias'].shape[0]
    key, zk, dk = jax.random.split(key, 3)
    if k == 0:
        mu, _ = actor.apply_fn(actor_variables(actor), obs, jnp.zeros((1, d), obs.dtype))
        return mu[0], key
    o = jnp.broadcast_to(obs, (k, obs.shape[1]))
    mu, _ = actor.apply_fn(actor_variables(actor), o, jax.random.normal(zk, (k, d), obs.dtype))
    if k == 1:
        return mu[0], key
    q = critic.apply_fn({'params': critic.params, 'batch_stats': critic.batch_stats}, o, mu,
                        rngs={'dropout': dk}, train=False)
    return mu[jnp.argmax(q.reshape(q.shape[0], k).min(0))], key


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--step", type=int, default=1_000_000)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--ks", default="0,1,4,8,16,32")
    p.add_argument("--out", default="results/best_of_k/out")
    a = p.parse_args()
    run = Path(a.run); cfgj = json.loads((run / 'config.json').read_text())
    # rebuild the network structure the run was trained with (CrossQ critic / CrossQ-style actor)
    ov = [f"benchmark={cfgj['task']}"]
    if cfgj['alg']['critic'].get('type', 'default') == 'crossq':
        ov += ["+critic=crossq", f"alg.critic.hs=[{','.join(map(str, cfgj['alg']['critic']['hs']))}]"]
    if cfgj['alg']['actor'].get('batch_renorm', False):
        ov += ["+actor=crossq"]
    cfg = compose_config(ov)
    env = gym.make(cfg.env_name)
    pol = IBOLTPolicy(env.observation_space, env.action_space, cfg=cfg)
    pol.build(jax.random.PRNGKey(0), None, 3e-4)
    cdir = next((run / 'checkpoints').glob(f'*/actor_state_{a.step}.msgpack')).parent
    actor = flax.serialization.from_bytes(pol.actor_state, (cdir / f'actor_state_{a.step}.msgpack').read_bytes())
    critic = flax.serialization.from_bytes(pol.qf_state, (cdir / f'critic_state_{a.step}.msgpack').read_bytes())
    res = dict(run=run.name, seed=cfgj['seed'], step=a.step, episodes=a.episodes, modes={})
    for k in map(int, a.ks.split(",")):
        key = jax.random.PRNGKey(1000 + k); rets = []; t0 = time.time()
        for i in range(a.episodes):
            obs, _ = env.reset(seed=10_000 + i); done, R = False, 0.0
            while not done:
                act_n, key = act(actor, critic, jnp.asarray(obs[None], jnp.float32), key, k)
                obs, r, term, trunc, _ = env.step(pol.unscale_action(np.clip(np.asarray(act_n), -1, 1)))
                R += r; done = term or trunc
            rets.append(R)
        name = "z0" if k == 0 else f"k{k}"
        res["modes"][name] = rets
        print(f"{run.name.split('_')[0]} step {a.step} {name:4s} {np.mean(rets):7.0f} ± {np.std(rets):5.0f}  ({time.time()-t0:.0f}s)", flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"ibolt_{run.name.split('_')[0]}_{a.step}.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main()
