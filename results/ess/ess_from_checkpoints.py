"""ESS/M of iBOLT's actor-update importance weights, recomputed from saved checkpoints.

Usage (repo root): python results/ess/ess_from_checkpoints.py <run_dir> [<run_dir> ...]
For each checkpoint (every 100K steps) of each run:
  - states: 256 states from a seeded rollout of that checkpoint's full policy (z + sigma);
  - exactly the update_actor weights: N=64 latents, M=N*repeats candidates from the box-truncated
    mixture, q = mean of the twin critics (train=False), w = softmax(q/T - beta*log pi_hat);
  - ESS/M = 1/(M * sum w^2), per state, for the combined weight and for its two factors alone:
    Q only = softmax(q/T), density only = softmax(-beta*log pi_hat).
Prints mean over states (and median of the combined ESS) per run and step.
"""
import glob, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import flax.serialization
import gymnasium as gym
import jax, jax.numpy as jnp
import numpy as np
from train import compose_config
from ibolt.policy import IBOLTPolicy
from ibolt.box_gaussian import sample_box, mixture_log_prob

N_STATES = 256


@jax.jit
def weights(actor, critic, obs, key, temperature, beta):
    n = 64  # num_policy_samples; proposals_per_policy_sample = 1 -> M = 64
    b, od = obs.shape
    d = actor.params['mu']['bias'].shape[0]
    zk, ck, nk, dk = jax.random.split(key, 4)
    z = jax.random.normal(zk, (b, n, d))
    o = jnp.broadcast_to(obs[:, None], (b, n, od)).reshape(b * n, od)
    mu, ls = actor.apply_fn({'params': actor.params}, o, z.reshape(b * n, d))
    mu, ls = mu.reshape(b, n, d), jnp.maximum(ls.reshape(b, n, d), jnp.log(0.006737946999085467))
    idx = jax.random.randint(ck, (b, n), 0, n)
    cand = sample_box(nk, jnp.take_along_axis(mu, idx[:, :, None], 1), jnp.take_along_axis(ls, idx[:, :, None], 1))
    v = critic.apply_fn({'params': critic.params, 'batch_stats': critic.batch_stats},
                        o, cand.reshape(b * n, d), rngs={'dropout': dk}, train=False)
    q = v.reshape(2, b, n, -1)[..., 0].mean(0)
    dens = mixture_log_prob(cand, mu, ls)
    ess = lambda logits: 1.0 / (n * (jax.nn.softmax(logits, -1) ** 2).sum(-1))
    return ess(q / temperature - beta * dens), ess(q / temperature), ess(-beta * dens), q.std(-1)


def main(runs):
    print(f"{'run':40s} {'step':>7s}  {'ESS/M comb':>10s} {'(median)':>8s}  {'Q only':>7s}  {'dens only':>9s}  {'Q std/T':>7s}")
    out = []
    for run in runs:
        run = Path(run)
        cfgj = json.loads((run / 'config.json').read_text())
        crossq = cfgj['alg']['critic'].get('type', 'default') == 'crossq'
        ov = [f"benchmark={cfgj['task']}"] + (['+critic=crossq', f"alg.critic.hs=[{','.join(map(str, cfgj['alg']['critic']['hs']))}]"] if crossq else [])
        cfg = compose_config(ov)
        T, beta = float(cfgj['alg']['actor']['temperature']), float(cfgj['alg']['actor']['density_correction_beta'])
        env = gym.make(cfg.env_name)
        pol = IBOLTPolicy(env.observation_space, env.action_space, cfg=cfg)
        pol.build(jax.random.PRNGKey(0), None, 3e-4)
        ck = glob.glob(str(run / 'checkpoints' / '*' / 'actor_state_*.msgpack'))
        steps = sorted(int(re.search(r'_(\d+)\.msgpack', c).group(1)) for c in ck)
        steps = [s for s in steps if s % 100_000 == 0]
        cdir = Path(ck[0]).parent
        for s in steps:
            pol.actor_state = flax.serialization.from_bytes(pol.actor_state, (cdir / f'actor_state_{s}.msgpack').read_bytes())
            pol.qf_state = flax.serialization.from_bytes(pol.qf_state, (cdir / f'critic_state_{s}.msgpack').read_bytes())
            pol.key = jax.random.PRNGKey(s)
            pol.evaluation_mu_only = False
            states, obs = [], env.reset(seed=s)[0]
            while len(states) < N_STATES * 4:
                states.append(obs.astype(np.float32))
                obs, _, term, trunc, _ = env.step(pol.predict(obs, deterministic=False)[0])
                if term or trunc:
                    obs = env.reset()[0]
            obs_b = jnp.asarray(np.stack(states[::4]))
            c, qo, do, qs = map(np.asarray, weights(pol.actor_state, pol.qf_state, obs_b, jax.random.PRNGKey(7), T, beta))
            name = run.name.split('_')[0]
            out.append(dict(run=name, step=s, crossq=crossq, ess=float(c.mean()), ess_med=float(np.median(c)),
                            ess_q=float(qo.mean()), ess_dens=float(do.mean()), q_std_over_T=float(qs.mean() / T)))
            print(f"{name:40s} {s:7d}  {c.mean():10.3f} {np.median(c):8.3f}  {qo.mean():7.3f}  {do.mean():9.3f}  {qs.mean()/T:7.2f}", flush=True)
        env.close()
    Path('ess_results.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:])
