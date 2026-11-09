"""UTD1: when does iBOLT (crossqAC) fall behind SBX CrossQ, and which metrics move with it."""
import glob, os, numpy as np, csv
H = os.path.expanduser("~/OptiQ")
T = {"walker2d": ("0.1", [f"{H}/outputs/walker2d-crossq-q1024-s*", "/tmp/hw/4090/walker2d-crossq-q1024-s*"]),
     "humanoid": ("0.5", [f"{H}/results/live_refs/humanoid-crossq-q1024-s*"]),
     "halfcheetah": ("0.5", [f"{H}/results/live_refs/halfcheetah-crossq-q*"])}
B = 50_000
def ev(d, m):
    f = glob.glob(os.path.join(d, "**", f"evaluations_{m}.npz"), recursive=True)
    if not f: return None
    z = np.load(f[0]); return np.asarray(z["timesteps"]), np.asarray(z["results"]).mean(1)
def binned(runs, edges):
    out = np.full((len(runs), len(edges)), np.nan)
    for i, (t, y) in enumerate(runs):
        for j, e in enumerate(edges):
            k = (t > e - B) & (t <= e)
            if k.any(): out[i, j] = y[k].mean()
    return out
def prog(d, xkey, keys):
    f = glob.glob(os.path.join(d, "logs", "progress.csv"))
    if not f: return None
    rows = list(csv.DictReader(open(f[0])))
    x = np.array([float(r[xkey]) if r.get(xkey) not in (None, "") else np.nan for r in rows])
    return {k: (x, np.array([float(r[k]) if r.get(k) not in (None, "") else np.nan for r in rows])) for k in keys}
def pbin(series, edges):
    x, y = series; out = []
    for e in edges:
        k = (x > e - B) & (x <= e) & ~np.isnan(y)
        out.append(y[k].mean() if k.any() else np.nan)
    return np.array(out)
edges = np.arange(B, 1_000_001, B)
IBK = ["rollout/ep_rew_mean", "train/actor_std_mean", "train/actor_std_min", "exploration/noise_std", "exploration/entropy_proxy",
       "train/current_q_values", "train/critic_loss"]
SBK = ["rollout/ep_rew_mean", "train/ent_coef", "train/critic_loss"]
for task, (Tv, spats) in T.items():
    ib_dirs = sorted(glob.glob(f"{H}/outputs_tsweep/{task}-ibolt-crossqAC-T{Tv}-s*") + glob.glob(f"{H}/outputs_pc/{task}-ibolt-crossqAC-T{Tv}-s*"))
    sb_dirs = sorted(d for p in spats for d in glob.glob(p))
    E = {m: binned([r for r in (ev(d, m) for d in ib_dirs) if r], edges) for m in ["zero_z", "stochastic_z", "stochastic_z_sigma"]}
    S = {m: binned([r for r in (ev(d, m) for d in sb_dirs) if r], edges) for m in ["deterministic", "stochastic"]}
    P = [p for p in (prog(d, "env_steps", IBK) for d in ib_dirs) if p]
    Q = [p for p in (prog(d, "time/total_timesteps", SBK) for d in sb_dirs) if p]
    pi = {k: np.nanmean([pbin(p[k], edges) for p in P], 0) for k in IBK}
    ps = {k: (np.nanmean([pbin(p[k], edges) for p in Q], 0) if Q else np.full(len(edges), np.nan)) for k in SBK}
    print(f"\n=== {task}  iBOLT T={Tv} n={len(ib_dirs)} (train logs {len(P)})  vs  SBX n={len(sb_dirs)} (train logs {len(Q)})   [50K bins]")
    print(f"{'step':>5s} | {'SBXdet':>6s} {'z0':>6s} {'t':>5s} | {'stoch z':>7s} | {'SBXsmp':>6s} {'z+σ':>6s} | {'roll SBX':>8s} {'roll iB':>7s} | "
          f"{'σ mean':>6s} {'σ min':>6s} {'dacer':>6s} {'H prx':>6s} | {'Q iB':>6s} {'Lc iB':>6s} {'Lc SBX':>6s} {'α SBX':>6s}")
    for j, e in enumerate(edges):
        a, b = E["zero_z"][:, j], S["deterministic"][:, j]
        se = np.sqrt(np.nanvar(a, ddof=1) / np.sum(~np.isnan(a)) + np.nanvar(b, ddof=1) / np.sum(~np.isnan(b)))
        t = (np.nanmean(a) - np.nanmean(b)) / se if se > 0 else np.nan
        print(f"{e//1000:4d}K | {np.nanmean(b):6.0f} {np.nanmean(a):6.0f} {t:5.1f} | {np.nanmean(E['stochastic_z'][:, j]):7.0f} | "
              f"{np.nanmean(S['stochastic'][:, j]):6.0f} {np.nanmean(E['stochastic_z_sigma'][:, j]):6.0f} | "
              f"{ps['rollout/ep_rew_mean'][j]:8.0f} {pi['rollout/ep_rew_mean'][j]:7.0f} | "
              f"{pi['train/actor_std_mean'][j]:6.3f} {pi['train/actor_std_min'][j]:6.3f} {pi['exploration/noise_std'][j]:6.3f} {pi['exploration/entropy_proxy'][j]:6.1f} | "
              f"{pi['train/current_q_values'][j]:6.0f} {pi['train/critic_loss'][j]:6.2f} {ps['train/critic_loss'][j]:6.2f} {ps['train/ent_coef'][j]:6.3f}")
