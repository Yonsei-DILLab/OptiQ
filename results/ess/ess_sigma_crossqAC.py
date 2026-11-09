"""ESS and log-sigma saturation of iBOLT runs (incl. CrossQ critic / CrossQ-style actor), from checkpoints.

Usage (repo root): python results/ess/ess_sigma_crossqAC.py <out_json> <run_dir> [<run_dir> ...]
Per checkpoint (every 100K steps):
  states: 256 states from a seeded rollout of that checkpoint's full policy (z + sigma);
  ESS/M exactly as in update_actor (N = M = 64 box-truncated mixture candidates, q = mean twin critic,
    train=False): combined softmax(q/T - beta*log pi_hat), Q only softmax(q/T), density only softmax(-beta*log pi_hat);
  log sigma: the actor clips raw_log_std to [log_std_min, log_std_max] = [-5, -1] with a hard clip
    (zero gradient outside), and initialises it at -1. We read the raw (pre-clip) value and report the
    share of (state, z, dim) entries at/above the upper bound, at/below the lower bound, and per action dim;
  spread: std of the centre tanh(mu(s, z)) over z (latent spread) next to the mean sigma.
"""
import glob, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import flax.serialization
import gymnasium as gym
import jax, jax.numpy as jnp
import numpy as np
from train import compose_config
from ibolt.policy import IBOLTPolicy, actor_variables
from ibolt.box_gaussian import sample_box, mixture_log_prob

N_STATES, N = 256, 64
FLOOR = float(np.log(0.006737946999085467))


def make_fn(actor_model, crossq):
    raw_model = actor_model.clone(log_std_min=-1e9, log_std_max=1e9)  # same params, no clip -> raw log sigma

    @jax.jit
    def fn(actor, critic, obs, key, temperature, beta):
        b, od = obs.shape
        d = actor.params['mu']['bias'].shape[0]
        zk, ck, nk, dk = jax.random.split(key, 4)
        z = jax.random.normal(zk, (b, N, d)).reshape(b * N, d)
        o = jnp.broadcast_to(obs[:, None], (b, N, od)).reshape(b * N, od)
        var = actor_variables(actor)
        mu, ls = actor.apply_fn(var, o, z)
        _, raw = raw_model.apply(var, o, z)
        mu, ls, raw = mu.reshape(b, N, d), jnp.maximum(ls.reshape(b, N, d), FLOOR), raw.reshape(b, N, d)
        idx = jax.random.randint(ck, (b, N), 0, N)
        cand = sample_box(nk, jnp.take_along_axis(mu, idx[:, :, None], 1), jnp.take_along_axis(ls, idx[:, :, None], 1))
        cv = {'params': critic.params, 'batch_stats': critic.batch_stats}
        v = critic.apply_fn(cv, o, cand.reshape(b * N, d), rngs={'dropout': dk}, train=False)
        q = v.reshape(2, b, N, -1)[..., 0].mean(0)
        dens = mixture_log_prob(cand, mu, ls)
        ess = lambda lg: 1.0 / (N * (jax.nn.softmax(lg, -1) ** 2).sum(-1))
        return dict(ess=ess(q / temperature - beta * dens), ess_q=ess(q / temperature), ess_dens=ess(-beta * dens),
                    q_std_over_T=q.std(-1) / temperature, raw=raw, sigma=jnp.exp(ls), mu_spread=mu.std(1))
    return fn


def main(out_json, runs):
    out = []
    for run in map(Path, runs):
        cfgj = json.loads((run / 'config.json').read_text())
        crossq = cfgj['alg']['critic'].get('type', 'default') == 'crossq'
        ov = [f"benchmark={cfgj['task']}"]
        if crossq:
            ov += ['+critic=crossq', f"alg.critic.hs=[{','.join(map(str, cfgj['alg']['critic']['hs']))}]"]
        if cfgj['alg']['actor'].get('batch_renorm', False):
            ov += ['+actor=crossq']
        cfg = compose_config(ov)
        a = cfgj['alg']['actor']
        T, beta, lo, hi = float(a['temperature']), float(a['density_correction_beta']), float(a['log_std_min']), float(a['log_std_max'])
        env = gym.make(cfg.env_name)
        pol = IBOLTPolicy(env.observation_space, env.action_space, cfg=cfg)
        pol.build(jax.random.PRNGKey(0), None, 3e-4)
        fn = make_fn(pol.actor_model, crossq)
        ck = glob.glob(str(run / 'checkpoints' / '*' / 'actor_state_*.msgpack'))
        steps = sorted(s for s in (int(re.search(r'_(\d+)\.msgpack', c).group(1)) for c in ck) if s % 100_000 == 0)
        cdir = Path(ck[0]).parent
        name = run.name.split('_')[0]
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
            r = {k: np.asarray(v) for k, v in fn(pol.actor_state, pol.qf_state, jnp.asarray(np.stack(states[::4])),
                                                 jax.random.PRNGKey(7), T, beta).items()}
            raw = r['raw']
            at_hi_dim = (raw >= hi).mean((0, 1))  # per action dim
            rec = dict(run=name, step=s, T=T, ess=float(r['ess'].mean()), ess_med=float(np.median(r['ess'])),
                       ess_q=float(r['ess_q'].mean()), ess_dens=float(r['ess_dens'].mean()),
                       q_std_over_T=float(r['q_std_over_T'].mean()),
                       at_hi=float((raw >= hi).mean()), at_lo=float((raw <= lo).mean()),
                       raw_mean=float(raw.mean()), raw_max=float(raw.max()),
                       dims_all_hi=int((at_hi_dim > 0.99).sum()), dims_any_lo=int(((raw <= lo).mean((0, 1)) > 0.5).sum()),
                       n_dims=int(raw.shape[-1]), at_hi_per_dim=at_hi_dim.round(3).tolist(),
                       sigma_mean=float(r['sigma'].mean()), mu_spread=float(r['mu_spread'].mean()))
            out.append(rec)
            print(f"{name:36s} {s//1000:5d}K  ESS comb {rec['ess']:.3f} Q {rec['ess_q']:.3f} dens {rec['ess_dens']:.3f}  "
                  f"Qstd/T {rec['q_std_over_T']:6.1f} | raw≥max {rec['at_hi']:.2f} raw≤min {rec['at_lo']:.2f} "
                  f"dims@max {rec['dims_all_hi']}/{rec['n_dims']} dims@min {rec['dims_any_lo']} raw mean {rec['raw_mean']:+.2f} max {rec['raw_max']:+.2f} | "
                  f"σ {rec['sigma_mean']:.3f} μ-spread(z) {rec['mu_spread']:.3f}", flush=True)
        env.close()
    Path(out_json).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
