"""Seeds 0-4 learning curves for stochastic_z and stochastic_z_sigma only.
For each seed and mode, keep the run (original or duplicate) with the higher
last-100K mean for that mode, then plot mean +/- 1 std over the 5 kept runs.
Usage: python plot_best.py <orig_eval_dir> <dup_eval_dir> <out_png>
"""
import glob, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ORIG, DUP, OUT = sys.argv[1:4]
TASKS = [("hopper", "Hopper-v4"), ("walker2d", "Walker2d-v4"), ("halfcheetah", "HalfCheetah-v4"),
         ("ant", "Ant-v4"), ("humanoid", "Humanoid-v4")]
MODES = [("stochastic_z", "z ~ N(0,I) (center)", "#eb6834", "--"),
         ("stochastic_z_sigma", "z ~ N(0,I) + σ (full policy)", "#1baf7a", ":")]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
def load(root):
    out = {}
    for d in glob.glob(os.path.join(root, "*-ibolt-*")):
        m = re.match(r"(\w+)-ibolt-T[0-9.]+-s(\d+)_", os.path.basename(d))
        if int(m.group(2)) > 4: continue
        for mode, *_ in MODES:
            z = np.load(glob.glob(os.path.join(d, "**", f"evaluations_{mode}.npz"), recursive=True)[0])
            out[(m.group(1), int(m.group(2)), mode)] = (z["timesteps"], z["results"].mean(1))
    return out
o, u = load(ORIG), load(DUP)
last = lambda r: r[1][r[0] > r[0][-1] - 100_000].mean()
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
fig, axes = plt.subplots(1, 5, figsize=(20, 4.2), facecolor=SURF)
for ax, (task, title) in zip(axes, TASKS):
    ax.set_facecolor(SURF); row = []
    for mode, label, color, ls in MODES:
        runs, tag = [], ""
        for s in range(5):
            a, b = o[(task, s, mode)], u[(task, s, mode)]
            r = a if last(a) >= last(b) else b; runs.append(r); tag += "o" if r is a else "d"
        n = min(len(t) for t, _ in runs); t = runs[0][0][:n]
        Y = np.stack([y[:n] for _, y in runs]); mu, sd = Y.mean(0), Y.std(0)
        ax.fill_between(t / 1e6, mu - sd, mu + sd, color=color, alpha=0.15, linewidth=0)
        ax.plot(t / 1e6, mu, color=color, linestyle=ls, linewidth=1.8, label=label)
        L = Y[:, t > t[-1] - 100_000].mean(1); row.append(f"{mode}: {L.mean():6.0f} ± {L.std():4.0f} [{tag}]")
    print(f"{task:12s} " + "   ".join(row))
    ax.set_title(title, color=INK, fontsize=11, loc="left"); ax.set_xlabel("Environment steps (M)")
    ax.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_xlim(0, 1.0)
axes[0].set_ylabel("Evaluation return")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
fig.text(0.995, 0.01, "iBOLT seeds 0-4, better of two runs per seed (by last-100K mean, per mode); mean ± 1 std over seeds",
         ha="right", va="bottom", color=INK2, fontsize=8.5)
fig.tight_layout(rect=(0, 0.02, 1, 0.93)); fig.savefig(OUT, dpi=150, facecolor=SURF); print("saved", OUT)
