"""Seeds 0-4: per seed, best of original vs duplicate (last-100K mean).
Usage: python best_of_two.py <orig_eval_dir> <dup_eval_dir>
(a) per-mode max (each mode may come from a different run)
(b) run picked by zero_z, all three modes from that same run
"""
import glob, os, re, sys
import numpy as np
MODES = ["zero_z", "stochastic_z", "stochastic_z_sigma"]
TASKS = ["hopper", "walker2d", "halfcheetah", "ant", "humanoid"]
def load(root):
    out = {}
    for d in glob.glob(os.path.join(root, "*-ibolt-*")):
        m = re.match(r"(\w+)-ibolt-T[0-9.]+-s(\d+)_", os.path.basename(d))
        if int(m.group(2)) > 4: continue
        for mode in MODES:
            z = np.load(glob.glob(os.path.join(d, "**", f"evaluations_{mode}.npz"), recursive=True)[0])
            t, y = z["timesteps"], z["results"].mean(1)
            out[(m.group(1), int(m.group(2)), mode)] = y[t > t[-1] - 100_000].mean()
    return out
o, u = load(sys.argv[1]), load(sys.argv[2])
for label, pick in [("(a) per-mode max", None), ("(b) run chosen by zero_z", "zero_z")]:
    print(label)
    for task in TASKS:
        cells = []
        for mode in MODES:
            v = []
            for s in range(5):
                if pick: src = o if o[(task, s, pick)] >= u[(task, s, pick)] else u; v.append(src[(task, s, mode)])
                else: v.append(max(o[(task, s, mode)], u[(task, s, mode)]))
            v = np.array(v)
            cells.append(" ".join(f"{x:6.0f}" for x in v) + f" | {v.mean():6.0f} ± {v.std():4.0f}")
        print(f"  {task:12s} " + "   ".join(cells))
    if pick:
        print("  chosen:", {t: "".join("o" if o[(t, s, pick)] >= u[(t, s, pick)] else "d" for s in range(5)) for t in TASKS})
