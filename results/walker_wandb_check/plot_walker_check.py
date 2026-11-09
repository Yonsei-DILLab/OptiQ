"""Walker2d zero_z: W&B best-of-N (paper-era runs) vs our runs.
Usage: python plot_walker_check.py <wandb_csv> <orig_eval_dir> <dup_eval_dir> <out_png>
Ours: (1) seeds 0-4 best of original/duplicate by last-100K zero_z, (2) all 10 original seeds 0-9.
"""
import glob, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV, ORIG, DUP, OUT = sys.argv[1:5]
W = np.loadtxt(CSV, delimiter=",", comments="#", skiprows=4)

def load(root):
    out = {}
    for d in glob.glob(os.path.join(root, "walker2d-ibolt-*")):
        s = int(re.search(r"-s(\d+)_", os.path.basename(d)).group(1))
        z = np.load(glob.glob(os.path.join(d, "**", "evaluations_zero_z.npz"), recursive=True)[0])
        out[s] = (z["timesteps"], z["results"].mean(1))
    return out

o, u = load(ORIG), load(DUP)
last = lambda r: r[1][r[0] > r[0][-1] - 100_000].mean()
best = [o[s] if last(o[s]) >= last(u[s]) else u[s] for s in range(5)]
all10 = [o[s] for s in range(10)]

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=SURF)
ax.set_facecolor(SURF)
t = W[:, 0] / 1e6
ax.fill_between(t, W[:, 1] - W[:, 2], W[:, 1] + W[:, 2], color="#d6334a", alpha=0.12, linewidth=0)
ax.plot(t, W[:, 1], color="#d6334a", linewidth=1.8, label="W&B runs, best of 2-3 per seed (s0-4)")
print(f"W&B best-of-N           last-100K: (see browser) final curve point {W[-1,1]:.0f}")
for runs, label, color, ls in [(best, "ours, best of 2 per seed (s0-4)", "#2a78d6", "--"),
                               (all10, "ours, 10 seeds (s0-9), no selection", "#52514e", ":")]:
    n = min(len(r[0]) for r in runs); tt = runs[0][0][:n]
    Y = np.stack([r[1][:n] for r in runs]); mu, sd = Y.mean(0), Y.std(0)
    ax.fill_between(tt / 1e6, mu - sd, mu + sd, color=color, alpha=0.10, linewidth=0)
    ax.plot(tt / 1e6, mu, color=color, linestyle=ls, linewidth=1.8, label=label)
    L = Y[:, tt > tt[-1] - 100_000].mean(1)
    print(f"{label:38s} last-100K: {L.mean():.0f} ± {L.std():.0f}")
ax.set_title("Walker2d-v4, zero_z eval (z=0 center)", color=INK, fontsize=11, loc="left")
ax.set_xlabel("Environment steps (M)"); ax.set_ylabel("Evaluation return")
ax.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True); ax.set_xlim(0, 1.0)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(frameon=False, loc="lower right", fontsize=9)
fig.tight_layout(); fig.savefig(OUT, dpi=150, facecolor=SURF); print("saved", OUT)
