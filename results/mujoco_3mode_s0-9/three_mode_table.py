"""Final-performance table of the three iBOLT evaluation rules over ALL runs (no selection).
Usage: python three_mode_table.py <eval_dir>   (e.g. results/mujoco_3mode_s0-9/eval)
Per run: mean evaluation return over the last 100K steps (paper metric); then mean ± std over seeds.
"""
import glob, sys
import numpy as np

ROOT = sys.argv[1]
MODES = [("zero_z", "z=0 (paper)"), ("stochastic_z", "z~N(0,I)"), ("stochastic_z_sigma", "z~N(0,I)+sigma")]
print("mean over the last 100K steps per run, mean ± std (ddof=1) over seeds; 10 evaluation episodes per point")
print(f"{'task':12s} {'n':>2s} " + " ".join(f"{lab:>16s}" for _, lab in MODES))
for t in ["hopper", "walker2d", "halfcheetah", "ant", "humanoid"]:
    res = {m: [] for m, _ in MODES}
    for d in sorted(glob.glob(f"{ROOT}/{t}-ibolt-*")):
        for m in res:
            z = np.load(glob.glob(f"{d}/**/evaluations_{m}.npz", recursive=True)[0])
            ts, y = np.asarray(z["timesteps"]), np.asarray(z["results"]).mean(1)
            res[m].append(y[ts > ts[-1] - 100_000].mean())
    cells = [f"{np.mean(v):8.0f} ± {np.std(v, ddof=1):4.0f}" for v in res.values()]
    print(f"{t:12s} {len(res['zero_z']):2d} " + " ".join(f"{c:>16s}" for c in cells))
