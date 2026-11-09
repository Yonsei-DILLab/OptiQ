"""Original vs duplicate (same seed, other server's GPU model) for seeds 0-4.

Usage: python compare_dup.py <orig_eval_dir> <dup_eval_dir> <out_png>
Prints per-run last-100K mean (paper metric) for each mode, the first eval step at
which the two curves differ, and the per-task mean over seeds. Plots original
(solid) vs duplicate (dashed) mean +/- 1 std over seeds for zero_z and z+sigma.
"""
import glob, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORIG, DUP, OUT = sys.argv[1:4]
TASKS = [("hopper", "Hopper-v4"), ("walker2d", "Walker2d-v4"), ("halfcheetah", "HalfCheetah-v4"),
         ("ant", "Ant-v4"), ("humanoid", "Humanoid-v4")]
MODES = ["zero_z", "stochastic_z", "stochastic_z_sigma"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

def load(root):
    out = {}
    for d in glob.glob(os.path.join(root, "*-ibolt-*")):
        m = re.match(r"(\w+)-ibolt-T[0-9.]+-s(\d+)_", os.path.basename(d))
        task, seed = m.group(1), int(m.group(2))
        if seed > 4:
            continue
        for mode in MODES:
            f = glob.glob(os.path.join(d, "**", f"evaluations_{mode}.npz"), recursive=True)[0]
            z = np.load(f)
            out[(task, seed, mode)] = (z["timesteps"], z["results"].mean(1))
    return out

o, u = load(ORIG), load(DUP)
last = lambda t, y: y[t > t[-1] - 100_000].mean()
print(f"{'run':16s} {'first diff step':>15s}  " + "  ".join(f"{m+' orig→dup':>28s}" for m in MODES))
summary = {}
for task, _ in TASKS:
    for s in range(5):
        t1, y1 = o[(task, s, "zero_z")]; t2, y2 = u[(task, s, "zero_z")]
        n = min(len(t1), len(t2)); diff = np.nonzero(~np.isclose(y1[:n], y2[:n]))[0]
        first = int(t1[diff[0]]) if len(diff) else -1
        cells = []
        for mode in MODES:
            a, b = last(*o[(task, s, mode)]), last(*u[(task, s, mode)])
            summary.setdefault((task, mode), []).append((a, b))
            cells.append(f"{a:8.0f} → {b:8.0f} ({b-a:+6.0f})")
        print(f"{task+'-s'+str(s):16s} {first:>15d}  " + "  ".join(f"{c:>28s}" for c in cells))
print("\nper-task mean ± std over seeds 0-4 (last-100K):  original | duplicate | mean |dup−orig|")
for task, _ in TASKS:
    row = []
    for mode in MODES:
        A = np.array(summary[(task, mode)])
        row.append(f"{mode}: {A[:,0].mean():6.0f}±{A[:,0].std():4.0f} | {A[:,1].mean():6.0f}±{A[:,1].std():4.0f} | {np.abs(A[:,1]-A[:,0]).mean():5.0f}")
    print(f"{task:12s} " + "   ".join(row))

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
fig, axes = plt.subplots(1, 5, figsize=(20, 4.2), facecolor=SURF)
for ax, (task, title) in zip(axes, TASKS):
    ax.set_facecolor(SURF)
    for mode, label, color in [("zero_z", "z = 0", "#2a78d6"), ("stochastic_z_sigma", "z + σ", "#1baf7a")]:
        for src, tag, ls in [(o, "original", "-"), (u, "duplicate (other GPU)", "--")]:
            runs = [src[(task, s, mode)] for s in range(5)]
            n = min(len(t) for t, _ in runs); t = runs[0][0][:n] / 1e6
            Y = np.stack([y[:n] for _, y in runs]); mu, sd = Y.mean(0), Y.std(0)
            ax.fill_between(t, mu - sd, mu + sd, color=color, alpha=0.08, linewidth=0)
            ax.plot(t, mu, color=color, linestyle=ls, linewidth=1.6, label=f"{label}, {tag}")
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel("Environment steps (M)"); ax.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_xlim(0, 1.0)
axes[0].set_ylabel("Evaluation return")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
fig.text(0.995, 0.01, "iBOLT seeds 0-4: original vs duplicate on the other server (RTX 4090 ↔ RTX 5090); mean ± 1 std over seeds",
         ha="right", va="bottom", color=INK2, fontsize=8.5)
fig.tight_layout(rect=(0, 0.02, 1, 0.93)); fig.savefig(OUT, dpi=150, facecolor=SURF); print("saved", OUT)
