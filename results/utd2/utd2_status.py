"""UTD=2 status: per group mean over seeds at the current step (last-25K window), vs UTD=1 at the same step."""
import glob, os, re, numpy as np
from collections import defaultdict
H = os.path.expanduser("~/OptiQ")

def ev(d, m):
    f = glob.glob(os.path.join(d, "**", f"evaluations_{m}.npz"), recursive=True)
    if not f: return None
    try:
        z = np.load(f[0]); t, y = np.asarray(z["timesteps"]), np.asarray(z["results"]).mean(1)
    except Exception: return None
    return (t, y) if len(t) else None

def at(r, step):  # mean over the 25K window ending at step
    t, y = r; k = (t <= step) & (t > step - 25_000)
    return y[k].mean() if k.any() and t[-1] >= step - 5000 else np.nan

G = defaultdict(list)
for d in glob.glob(f"{H}/outputs_utd2/*"):
    G[re.sub(r"-s\d+_.*$", "", os.path.basename(d))].append(d)
REF1 = {  # UTD=1 counterparts
    "humanoid-crossq-q1024-utd2": [f"{H}/results/live_refs/humanoid-crossq-q1024-s*"],
    "walker2d-crossq-q1024-utd2": [f"{H}/outputs/walker2d-crossq-q1024-s*", "/tmp/hw/4090/walker2d-crossq-q1024-s*"],
    "ant-crossq-q1024-utd2": [f"{H}/outputs/ant-crossq-q1024-s*"],
    "humanoid-ibolt-crossqAC-utd2-T0.5": [f"{H}/outputs_tsweep/humanoid-ibolt-crossqAC-T0.5-s*", f"{H}/outputs_pc/humanoid-ibolt-crossqAC-T0.5-s*"],
    "humanoid-ibolt-crossqAC-utd2-nodacer-T0.5": [f"{H}/outputs_tsweep/humanoid-ibolt-crossqAC-T0.5-s*", f"{H}/outputs_pc/humanoid-ibolt-crossqAC-T0.5-s*"],
    "walker2d-ibolt-crossqAC-utd2-T0.1": [f"{H}/outputs_tsweep/walker2d-ibolt-crossqAC-T0.1-s*"],
    "walker2d-ibolt-crossqAC-utd2-nodacer-T0.1": [f"{H}/outputs_tsweep/walker2d-ibolt-crossqAC-T0.1-s*"],
}
print(f"{'group':44s} {'n':>2s} {'step':>6s}  {'z0/det':>7s} {'stoch z':>7s} {'z+σ/smp':>7s}   | UTD1 same step z0/det (n)")
for g in sorted(G, key=lambda s: (s.split("-")[0], "crossq-q" not in s, s)):
    sbx = "-crossq-q" in g
    ms = ["deterministic", None, "stochastic"] if sbx else ["zero_z", "stochastic_z", "stochastic_z_sigma"]
    runs = [[ev(d, m) if m else None for m in ms] for d in G[g]]
    runs = [r for r in runs if r[0] is not None]
    if not runs: print(f"{g:44s} no eval yet"); continue
    step = min(int(r[0][0][-1]) for r in runs)
    cols = []
    for j in range(3):
        v = [at(r[j], step) for r in runs if r[j] is not None]
        cols.append(f"{np.nanmean(v):7.0f}" if v else "      —")
    ref = ""
    if g in REF1:
        rr = [ev(d, ms[0]) for p in REF1[g] for d in glob.glob(p)]
        v = [at(r, step) for r in rr if r is not None]
        v = [x for x in v if not np.isnan(x)]
        if v: ref = f"{np.mean(v):7.0f} ({len(v)})"
    print(f"{g:44s} {len(runs):2d} {step//1000:5d}K  {' '.join(cols)}   | {ref}")
