"""Per-action inference latency (Table 2 protocol, made explicit for the rebuttal).

Usage (repo root):
  .venv/bin/python     results/inference_time/time_inference.py ibolt sac td3     # same JAX for all
  ~/sbx-venv/bin/python results/inference_time/time_inference.py sbx_sac sbx_td3  # SBX itself, cross-check

Protocol
  - Humanoid-v4, batch size 1, single GPU (CUDA_VISIBLE_DEVICES), nothing else running.
  - 5 seeds; a seed sets the network initialization and the observation stream (1,200 observations
    from a seeded rollout with uniform random actions). Latency does not depend on trained weights.
  - Compilation: every (method, mode) is compiled separately; its first call (trace + XLA compile,
    seed 0) is timed and reported on its own and never enters the latency, then 200 untimed
    warm-up calls, then 1,000 timed calls.
  - Synchronization: "predict" is the evaluation call (numpy obs in, numpy action out). The numpy
    conversion of the output forces the device to finish, so wall time includes host->device copy,
    the PRNG key split, the jitted actor, and device->host copy. "forward" times only the jitted
    action function on a device-resident observation, closed with block_until_ready().
  - Report: mean +/- std across the 5 seed means (Table 2 averages over 5 seeds), plus the pooled
    per-call std and 95th percentile.
"""
import json, sys, time
from functools import partial
from pathlib import Path
import gymnasium as gym
import jax, jax.numpy as jnp
import numpy as np

TASK, ENV_ID = "humanoid", "Humanoid-v4"
SEEDS, N_OBS, WARM, N = (0, 1, 2, 3, 4), 1200, 200, 1000
ROOT = Path(__file__).resolve().parents[2]


def obs_stream(env, seed):
    obs, out = env.reset(seed=seed)[0], []
    env.action_space.seed(seed)
    while len(out) < N_OBS:
        out.append(obs.astype(np.float32))
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            obs = env.reset()[0]
    return out


# ---------------------------------------------------------------- iBOLT (this repo)
def build_ibolt(env, seed):
    sys.path.insert(0, str(ROOT))
    from train import compose_config
    from ibolt.policy import IBOLTPolicy
    cfg = compose_config([f"benchmark={TASK}"])
    pol = IBOLTPolicy(env.observation_space, env.action_space, cfg=cfg)
    pol.build(jax.random.PRNGKey(seed), None, 3e-4)
    modes = {}
    for mode, (det, mu_only) in {"zero_z": (True, True), "stochastic_z": (False, True),
                                 "stochastic_z_sigma": (False, False)}.items():
        def predict(o, det=det, mu_only=mu_only):
            pol.evaluation_mu_only = mu_only
            return pol.predict(o, deterministic=det)[0]
        def forward(o, k, det=det, mu_only=mu_only):
            return IBOLTPolicy.sample_action(pol.actor_state, o, k, deterministic=det,
                                             sample_conditional_noise=not mu_only)
        modes[mode] = (predict, forward)
    return modes


# ------------------------------------------- SAC / TD3 actors as in SBX, in the same JAX as iBOLT
import flax.linen as nn


class SACActor(nn.Module):  # sbx.common.policies.Actor: relu MLP, mean / log_std heads, tanh squash
    action_dim: int
    @nn.compact
    def __call__(self, x):
        for w in (256, 256):
            x = nn.relu(nn.Dense(w)(x))
        return nn.Dense(self.action_dim)(x), jnp.clip(nn.Dense(self.action_dim)(x), -20, 2)


class TD3Actor(nn.Module):  # sbx.td3.policies.Actor
    action_dim: int
    @nn.compact
    def __call__(self, x):
        for w in (256, 256):
            x = nn.relu(nn.Dense(w)(x))
        return nn.tanh(nn.Dense(self.action_dim)(x))


def _wrap(env, seed, act_fn_det, act_fn_sto):
    """SBX-style predict: key split per call, jitted action, numpy out, clip, unscale."""
    low, high = env.action_space.low, env.action_space.high
    state = {"key": jax.random.PRNGKey(seed + 1000)}
    def make(fn, stochastic):
        def predict(o):
            batch = np.asarray(o).reshape(1, -1)
            if stochastic:
                state["key"], k = jax.random.split(state["key"])
                a = np.asarray(fn(batch, k))
            else:
                a = np.asarray(fn(batch))
            a = np.clip(a, -1, 1)
            return (low + 0.5 * (a + 1.0) * (high - low))[0]
        forward = (lambda o, k: fn(o, k)) if stochastic else (lambda o, k: fn(o))
        return predict, forward
    out = {"deterministic": make(act_fn_det, False)}
    if act_fn_sto is not None:
        out["stochastic"] = make(act_fn_sto, True)
    return out


def build_sac(env, seed):
    d = env.action_space.shape[0]
    m = SACActor(d)
    p = m.init(jax.random.PRNGKey(seed), jnp.zeros((1, env.observation_space.shape[0])))
    det = jax.jit(lambda o: jnp.tanh(m.apply(p, o)[0]))
    def _sto(o, k):
        mu, ls = m.apply(p, o)
        return jnp.tanh(mu + jnp.exp(ls) * jax.random.normal(k, mu.shape))
    return _wrap(env, seed, det, jax.jit(_sto))


def build_td3(env, seed):
    d = env.action_space.shape[0]
    m = TD3Actor(d)
    p = m.init(jax.random.PRNGKey(seed), jnp.zeros((1, env.observation_space.shape[0])))
    return _wrap(env, seed, jax.jit(lambda o: m.apply(p, o)), None)  # TD3 evaluates without noise


# ---------------------------------------------------------------- SBX itself (cross-check)
def _build_sbx(cls_name, env, seed):
    import sbx
    model = getattr(sbx, cls_name)("MlpPolicy", env, seed=seed, verbose=0)
    pol = model.policy
    modes = {"deterministic": (lambda o: pol.predict(o, deterministic=True)[0],
                               lambda o, k: type(pol).select_action(pol.actor_state, o))}
    if cls_name == "SAC":
        modes["stochastic"] = (lambda o: pol.predict(o, deterministic=False)[0],
                               lambda o, k: type(pol).sample_action(pol.actor_state, o, k))
    return modes


def _fused_predict(env, seed, step):
    """One device dispatch per action: the PRNG split happens inside the jitted step, the key
    stays on the device. Deterministic modes take no key at all (as SBX does for SAC/TD3)."""
    low, high = env.action_space.low, env.action_space.high
    state = {"key": jax.random.PRNGKey(seed + 1000)}
    def predict(o):
        a, state["key"] = step(np.asarray(o).reshape(1, -1), state["key"])
        a = np.clip(np.asarray(a), -1, 1)
        return (low + 0.5 * (a + 1.0) * (high - low))[0]
    return predict, (lambda o, k: step(o, k)[0])


def build_ibolt_fused(env, seed):
    sys.path.insert(0, str(ROOT))
    from train import compose_config
    from ibolt.policy import IBOLTPolicy
    cfg = compose_config([f"benchmark={TASK}"])
    pol = IBOLTPolicy(env.observation_space, env.action_space, cfg=cfg)
    pol.build(jax.random.PRNGKey(seed), None, 3e-4)
    st = pol.actor_state
    @jax.jit
    def zero(o, key):  # z = 0: no randomness, key passed through untouched
        return IBOLTPolicy.sample_action(st, o, key, deterministic=True), key
    def make(mu_only):
        @jax.jit
        def step(o, key):
            key, sub = jax.random.split(key)
            return IBOLTPolicy.sample_action(st, o, sub, deterministic=False,
                                             sample_conditional_noise=not mu_only), key
        return step
    return {"zero_z": _fused_predict(env, seed, zero),
            "stochastic_z": _fused_predict(env, seed, make(True)),
            "stochastic_z_sigma": _fused_predict(env, seed, make(False))}


def build_sac_fused(env, seed):
    d = env.action_space.shape[0]
    m = SACActor(d)
    p = m.init(jax.random.PRNGKey(seed), jnp.zeros((1, env.observation_space.shape[0])))
    @jax.jit
    def step(o, key):
        key, sub = jax.random.split(key)
        mu, ls = m.apply(p, o)
        return jnp.tanh(mu + jnp.exp(ls) * jax.random.normal(sub, mu.shape)), key
    return {"stochastic": _fused_predict(env, seed, step)}


BUILDERS = {"ibolt_fused": build_ibolt_fused, "sac_fused": build_sac_fused, "ibolt": build_ibolt, "sac": build_sac, "td3": build_td3,
            "sbx_sac": partial(_build_sbx, "SAC"), "sbx_td3": partial(_build_sbx, "TD3")}


def time_mode(predict, forward, stream, seed):
    t0 = time.perf_counter(); predict(stream[0]); compile_ms = (time.perf_counter() - t0) * 1e3
    for i in range(WARM):
        predict(stream[1 + i % (N_OBS - 1)])
    tp = np.empty(N)
    for i in range(N):
        o = stream[i % N_OBS]
        t0 = time.perf_counter(); predict(o); tp[i] = time.perf_counter() - t0
    dev = [jnp.asarray(o[None]) for o in stream[:64]]
    keys = list(jax.random.split(jax.random.PRNGKey(seed + 2000), WARM + N))
    jax.block_until_ready((dev, keys))
    for i in range(WARM):
        forward(dev[i % 64], keys[i]).block_until_ready()
    tf = np.empty(N)
    for i in range(N):
        o, k = dev[i % 64], keys[WARM + i]
        t0 = time.perf_counter(); forward(o, k).block_until_ready(); tf[i] = time.perf_counter() - t0
    return compile_ms, tp * 1e3, tf * 1e3


def main(methods):
    print(f"jax {jax.__version__}  backend={jax.default_backend()}  device={jax.devices()[0].device_kind}"
          f"  task={ENV_ID}  seeds={len(SEEDS)}  timed calls/seed={N}  warm-up={WARM}")
    hdr = f"{'method':9s} {'mode':19s} {'predict ms':>18s} {'call std':>8s} {'p95':>6s}   " \
          f"{'forward ms':>18s} {'call std':>8s}   {'compile ms':>12s}"
    print(hdr)
    rows = []
    for meth in methods:
        res = {}
        for seed in SEEDS:
            env = gym.make(ENV_ID)
            stream = obs_stream(env, seed)
            for mode, (pr, fw) in BUILDERS[meth](env, seed).items():
                res.setdefault(mode, []).append(time_mode(pr, fw, stream, seed))
            env.close()
        for mode, r in res.items():
            c = np.array([x[0] for x in r])
            pm = np.array([x[1].mean() for x in r]); fm = np.array([x[2].mean() for x in r])
            pall = np.concatenate([x[1] for x in r]); fall = np.concatenate([x[2] for x in r])
            row = dict(method=meth, mode=mode, predict_mean=pm.mean(), predict_seed_std=pm.std(ddof=1),
                       predict_call_std=pall.std(), predict_p95=np.percentile(pall, 95),
                       forward_mean=fm.mean(), forward_seed_std=fm.std(ddof=1), forward_call_std=fall.std(),
                       # first call of seed 0 = cold compile (later seeds may hit the jit cache)
                       compile_first=c[0], jax=jax.__version__)
            rows.append(row)
            print(f"{meth:9s} {mode:19s} {row['predict_mean']:9.3f} ± {row['predict_seed_std']:.3f}"
                  f" {row['predict_call_std']:8.3f} {row['predict_p95']:6.3f}   "
                  f"{row['forward_mean']:9.3f} ± {row['forward_seed_std']:.3f} {row['forward_call_std']:8.3f}"
                  f"   {row['compile_first']:12.0f}", flush=True)
    out = Path(__file__).with_name(f"latency_{'_'.join(methods)}.json")
    out.write_text(json.dumps(rows, indent=2, default=float))


if __name__ == "__main__":
    main(sys.argv[1:] or ["ibolt", "sac", "td3"])
