"""Latency of iBOLT with best-of-K critic selection (z_1..z_K ~ N(0,I), argmax twin-min Q), Table 2 protocol.

Usage (repo root): .venv/bin/python results/inference_time/bench_bok.py
Same convention as bench_interleaved.py: numpy obs in, numpy action out (synchronizes), TrainStates
as arguments, PRNG split inside the one jitted call, compile + 300 warm-up calls excluded,
5 seeds x 30 interleaved blocks x 100 calls, mean ± std over seed means. The K candidates go through
the actor and the twin critic as one batch (one dispatch per action).
"""
import sys, time
from functools import partial
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gymnasium as gym
import jax, jax.numpy as jnp
import numpy as np
from time_inference import obs_stream
from train import compose_config
from ibolt.policy import IBOLTPolicy, actor_variables

SEEDS, BLOCKS, PER, WARM = (0, 1, 2, 3, 4), 30, 100, 300
KS = (4, 10, 16, 64)


@partial(jax.jit, static_argnames=["k"])
def bok(actor, critic, obs, key, k):
    d = actor.params['mu']['bias'].shape[0]
    key, zk, dk = jax.random.split(key, 3)
    o = jnp.broadcast_to(obs, (k, obs.shape[1]))
    mu, _ = actor.apply_fn(actor_variables(actor), o, jax.random.normal(zk, (k, d), obs.dtype))
    q = critic.apply_fn({'params': critic.params, 'batch_stats': critic.batch_stats}, o, mu,
                        rngs={'dropout': dk}, train=False)
    return mu[jnp.argmax(q.reshape(q.shape[0], k).min(0))][None], key


names = ["iBOLT zero z"] + [f"iBOLT best-of-{k}" for k in KS]
print(f"jax {jax.__version__}  {jax.devices()[0].device_kind}  Humanoid-v4  batch 1")
seed_means = {k: [] for k in names}; first = {}
rng = np.random.default_rng(0)
for seed in SEEDS:
    env = gym.make("Humanoid-v4"); stream = obs_stream(env, seed)
    low, high = env.action_space.low, env.action_space.high
    pol = IBOLTPolicy(env.observation_space, env.action_space, cfg=compose_config(["benchmark=humanoid"]))
    pol.build(jax.random.PRNGKey(seed), None, 3e-4)
    finish = lambda a: low + 0.5 * (np.clip(np.asarray(a), -1, 1) + 1.0) * (high - low)
    st = {"k": jax.random.PRNGKey(seed + 1000)}
    def mk(k):
        def f(o):
            a, st["k"] = bok(pol.actor_state, pol.qf_state, o.reshape(1, -1), st["k"], k)
            return finish(a)
        return f
    calls = {"iBOLT zero z": lambda o: finish(pol._predict(o.reshape(1, -1), deterministic=True))}
    calls.update({f"iBOLT best-of-{k}": mk(k) for k in KS})
    for n, f in calls.items():
        t0 = time.perf_counter(); f(stream[0]); first.setdefault(n, (time.perf_counter() - t0) * 1e3)
        for i in range(WARM):
            f(stream[1 + i % (len(stream) - 1)])
    blocks = {k: [] for k in names}
    for b in range(BLOCKS):
        for n in rng.permutation(names):
            f, t = calls[n], np.empty(PER)
            for i in range(PER):
                o = stream[(b * PER + i) % len(stream)]
                t0 = time.perf_counter(); f(o); t[i] = time.perf_counter() - t0
            blocks[n].append(t.mean() * 1e3)
    for n in names:
        seed_means[n].append(np.mean(blocks[n]))
    env.close()
print(f"\nlatency per action, {len(SEEDS)} seeds x {BLOCKS} interleaved blocks x {PER} calls (mean ± std over seeds)")
for n in names:
    v = np.array(seed_means[n]); print(f"{n:22s} {v.mean():.3f} ± {v.std(ddof=1):.3f} ms   (first call {first[n]:.0f} ms)")
