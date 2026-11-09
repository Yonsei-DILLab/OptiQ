"""Inference latency per action, measured so that mode-to-mode differences are trustworthy.

Usage (repo root): .venv/bin/python results/inference_time/bench_interleaved.py
Humanoid-v4, batch 1, one GPU, jax of the iBOLT venv for every method.

Same call convention for every method (as in SBX): the jitted action function receives the
actor TrainState and a numpy observation; stochastic modes split the PRNG key inside the same
jitted call (one dispatch per action); the action comes back as numpy (this synchronizes the
device), is clipped and unscaled to the env bounds. iBOLT uses the repo's IBOLTPolicy._predict.

[1] Latency: 5 seeds (network init + observation stream). Per seed, all methods/modes are
    compiled and warmed up (300 calls each; compile never enters the timing), then 30 blocks;
    each block visits every method/mode in a fresh random order and times 100 calls. This removes
    drift and order effects. Report: mean ± std across the 5 seed means, and the paired difference
    to iBOLT zero z (mean ± s.e. over all blocks).
[2] Device-only compute: 2,000 sequential actions inside one jitted lax.scan (one dispatch),
    i.e. GPU time per action without Python or dispatch overhead.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gymnasium as gym
import jax, jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from time_inference import SACActor, TD3Actor, obs_stream
from train import compose_config
from ibolt.policy import IBOLTPolicy

SEEDS, BLOCKS, PER, WARM = (0, 1, 2, 3, 4), 30, 100, 300
NAMES = ["iBOLT zero z", "iBOLT stochastic z", "iBOLT z+sigma",
         "SAC deterministic", "SAC stochastic", "TD3 deterministic"]


@jax.jit
def sac_det(s, o):
    return jnp.tanh(s.apply_fn({"params": s.params}, o)[0])


@jax.jit
def sac_sto(s, o, key):
    key, sub = jax.random.split(key)
    mu, ls = s.apply_fn({"params": s.params}, o)
    return jnp.tanh(mu + jnp.exp(ls) * jax.random.normal(sub, mu.shape)), key


@jax.jit
def td3_det(s, o):
    return s.apply_fn({"params": s.params}, o)


def build(env, seed):
    D, O = env.action_space.shape[0], env.observation_space.shape[0]
    low, high = env.action_space.low, env.action_space.high
    pol = IBOLTPolicy(env.observation_space, env.action_space, cfg=compose_config(["benchmark=humanoid"]))
    pol.build(jax.random.PRNGKey(seed), None, 3e-4)
    def ts(model):
        p = model.init(jax.random.PRNGKey(seed), jnp.zeros((1, O)))["params"]
        return TrainState.create(apply_fn=model.apply, params=p, tx=optax.adam(3e-4))
    sst, tst = ts(SACActor(D)), ts(TD3Actor(D))
    key = {"k": jax.random.PRNGKey(seed + 1000)}

    def finish(a):
        a = np.clip(np.asarray(a), -1, 1)
        return low + 0.5 * (a + 1.0) * (high - low)

    def ib(mu_only, det):
        def f(o):
            pol.evaluation_mu_only = mu_only
            return finish(pol._predict(o.reshape(1, -1), deterministic=det))
        return f

    def sac_s(o):
        a, key["k"] = sac_sto(sst, o.reshape(1, -1), key["k"])
        return finish(a)

    calls = dict(zip(NAMES, [ib(True, True), ib(True, False), ib(False, False),
                             lambda o: finish(sac_det(sst, o.reshape(1, -1))), sac_s,
                             lambda o: finish(td3_det(tst, o.reshape(1, -1)))]))
    return calls, pol.actor_state, sst, tst


print(f"jax {jax.__version__}  {jax.devices()[0].device_kind}  Humanoid-v4  batch 1")
seed_means = {k: [] for k in NAMES}
diffs = {k: [] for k in NAMES}
rng = np.random.default_rng(0)
for seed in SEEDS:
    env = gym.make("Humanoid-v4")
    stream = obs_stream(env, seed)
    calls, ist, sst, tst = build(env, seed)
    for f in calls.values():
        for i in range(WARM):
            f(stream[i])
    blocks = {k: [] for k in NAMES}
    for b in range(BLOCKS):
        for k in rng.permutation(NAMES):
            f, t = calls[k], np.empty(PER)
            for i in range(PER):
                o = stream[(b * PER + i) % len(stream)]
                t0 = time.perf_counter(); f(o); t[i] = time.perf_counter() - t0
            blocks[k].append(t.mean() * 1e3)
    for k in NAMES:
        seed_means[k].append(np.mean(blocks[k]))
        diffs[k].extend(np.array(blocks[k]) - np.array(blocks["iBOLT zero z"]))
    env.close()

print(f"\n[1] latency per action, {len(SEEDS)} seeds x {BLOCKS} interleaved blocks x {PER} calls")
print(f"{'':20s} {'ms (mean ± std over seeds)':>27s} {'vs iBOLT zero z (± s.e.)':>27s}")
for k in NAMES:
    m, d = np.array(seed_means[k]), np.array(diffs[k])
    print(f"{k:20s} {m.mean():14.3f} ± {m.std(ddof=1):.3f}      {d.mean():+.4f} ± {d.std(ddof=1) / np.sqrt(len(d)):.4f}")

K = 2000
obs_dev = jnp.asarray(np.stack(stream[:K]))[:, None, :]
DEV = {
    "iBOLT zero z": lambda o, k: (IBOLTPolicy.sample_action(ist, o, k, deterministic=True), k),
    "iBOLT stochastic z": lambda o, k: IBOLTPolicy._act_and_advance(ist, o, k, sample_conditional_noise=False),
    "iBOLT z+sigma": lambda o, k: IBOLTPolicy._act_and_advance(ist, o, k, sample_conditional_noise=True),
    "SAC deterministic": lambda o, k: (sac_det(sst, o), k),
    "SAC stochastic": lambda o, k: sac_sto(sst, o, k),
    "TD3 deterministic": lambda o, k: (td3_det(tst, o), k),
}
print(f"\n[2] device-only compute: {K} sequential actions in one jitted scan, 20 repeats")
for k, step in DEV.items():
    @jax.jit
    def run(obs, key, step=step):
        def body(c, o):
            a, c = step(o, c)
            return c, a.sum()
        return jax.lax.scan(body, key, obs)[1].sum()
    key = jax.random.PRNGKey(3)
    run(obs_dev, key).block_until_ready()
    ts = []
    for _ in range(20):
        t0 = time.perf_counter(); run(obs_dev, key).block_until_ready(); ts.append(time.perf_counter() - t0)
    ts = np.array(ts) / K * 1e6
    print(f"{k:20s} {ts.mean():6.2f} ± {ts.std(ddof=1):.2f} µs/action")
