"""Pair checkpoint ESS with evaluation return (Humanoid, iBOLT+CrossQ critic vs paper critic).

Usage: python ess_vs_return.py <ess_json> [<ess_json> ...]   (run on the 5090 server)
Return at step s = mean of zero_z (and z+sigma) evals in [s-10K, s+10K] for the same run directory
that the checkpoint came from. Prints per-step group means and the per-seed rows.
"""
import glob, json, os, sys
import numpy as np

rows = [r for f in sys.argv[1:] for r in json.load(open(f))]
EVAL_ROOTS = [os.path.expanduser("~/OptiQ/outputs"), "/tmp/cqplot/cq"]


def ret(run, step, mode):
    for root in EVAL_ROOTS:
        f = glob.glob(f"{root}/{run}_*/eval/**/evaluations_{mode}.npz", recursive=True)
        if f:
            z = np.load(f[0]); t = np.asarray(z["timesteps"]); y = np.asarray(z["results"]).mean(1)
            m = (t >= step - 10_000) & (t <= step + 10_000)
            return float(y[m].mean()) if m.any() else np.nan
    return np.nan


for r in rows:
    r["ret_z0"] = ret(r["run"], r["step"], "zero_z")
    r["ret_sig"] = ret(r["run"], r["step"], "stochastic_z_sigma")

print(f"{'step':>5s} {'critic':7s} {'n':>2s} {'return z=0':>13s} {'return z+σ':>13s} "
      f"{'ESS/M comb':>11s} {'Q only':>13s} {'dens only':>13s} {'Q std/T':>11s}")
for s in range(100_000, 600_001, 100_000):
    for cq, lab in ((True, "CrossQ"), (False, "paper")):
        g = [r for r in rows if r["step"] == s and r["crossq"] == cq and "humanoid" in r["run"]]
        if not g:
            continue
        f = lambda k: (np.mean([r[k] for r in g]), np.std([r[k] for r in g]))
        out = [f(k) for k in ("ret_z0", "ret_sig", "ess", "ess_q", "ess_dens", "q_std_over_T")]
        print(f"{s//1000:4d}K {lab:7s} {len(g):2d} {out[0][0]:6.0f} ± {out[0][1]:4.0f} {out[1][0]:6.0f} ± {out[1][1]:4.0f} "
              f"{out[2][0]:5.3f} ± {out[2][1]:.3f} {out[3][0]:5.3f} ± {out[3][1]:.3f} {out[4][0]:5.3f} ± {out[4][1]:.3f} "
              f"{out[5][0]:5.1f} ± {out[5][1]:4.1f}")
print("\nper seed (step, run, return z=0, ESS comb, Q only, dens only, Q std/T)")
for r in sorted(rows, key=lambda r: (r["step"], not r["crossq"], r["run"])):
    if r["step"] <= 500_000:
        print(f"{r['step']//1000:4d}K {r['run']:32s} {r['ret_z0']:6.0f} {r['ess']:.3f} {r['ess_q']:.3f} "
              f"{r['ess_dens']:.3f} {r['q_std_over_T']:5.1f}")
