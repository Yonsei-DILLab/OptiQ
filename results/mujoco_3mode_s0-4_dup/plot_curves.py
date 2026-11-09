"""Learning curves for the 3-mode MuJoCo eval (seeds 5-9).

Usage: python plot_curves.py [eval_dir] [out_png] [seeds, e.g. 5,6,7,8,9 (default)]
Mean +/- 1 std across seeds (paper convention), truncated per task to the
shortest seed so the mean never mixes different seed counts.
Also prints the paper's final metric: mean over the last 100K steps, then over seeds.
"""
import glob, os, re, sys
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "eval")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "learning_curves.png")
SEEDS = {int(s) for s in (sys.argv[3] if len(sys.argv) > 3 else "5,6,7,8,9").split(",")}
TASKS = [("hopper", "Hopper-v4"), ("walker2d", "Walker2d-v4"), ("halfcheetah", "HalfCheetah-v4"),
         ("ant", "Ant-v4"), ("humanoid", "Humanoid-v4")]
# Reference categorical palette, slots 1-3 in fixed order + line style as secondary encoding.
MODES = [("zero_z", "z = 0 (center)", "#2a78d6", "-"),
         ("stochastic_z", "z ~ N(0,I) (center)", "#eb6834", "--"),
         ("stochastic_z_sigma", "z ~ N(0,I) + σ (full policy)", "#1baf7a", ":")]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

data = defaultdict(dict)  # task -> mode -> {seed: (t, mean_return)}
for d in glob.glob(os.path.join(EVAL, "*-ibolt-*")):
    m = re.match(r"(\w+)-ibolt-T[0-9.]+-s(\d+)_", os.path.basename(d))
    task, seed = m.group(1), int(m.group(2))
    if seed not in SEEDS:
        continue
    for mode, *_ in MODES:
        f = glob.glob(os.path.join(d, "**", f"evaluations_{mode}.npz"), recursive=True)
        if f:
            z = np.load(f[0])
            data[task].setdefault(mode, {})[seed] = (z["timesteps"], z["results"].mean(1))

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
fig, axes = plt.subplots(1, 5, figsize=(20, 4.2), facecolor=SURF)
print(f"{'task':12s} {'seeds':>5s} {'steps':>7s}  " + "  ".join(f"{m:>20s}" for m, *_ in MODES)
      + "   (mean ± std over seeds of last-100K mean)")
for ax, (task, title) in zip(axes, TASKS):
    ax.set_facecolor(SURF)
    runs = data.get(task, {})
    n = min(len(t) for seeds in runs.values() for t, _ in seeds.values())
    nseeds = len(runs.get("zero_z", {}))
    row = []
    for mode, label, color, ls in MODES:
        seeds = runs[mode]
        t = next(iter(seeds.values()))[0][:n]
        Y = np.stack([y[:n] for _, y in seeds.values()])
        mu, sd = Y.mean(0), Y.std(0)
        ax.fill_between(t / 1e6, mu - sd, mu + sd, color=color, alpha=0.15, linewidth=0)
        ax.plot(t / 1e6, mu, color=color, linestyle=ls, linewidth=1.8, label=label)
        last = Y[:, t > t[-1] - 100_000].mean(1)
        row.append(f"{last.mean():9.0f} ± {last.std():6.0f}")
    print(f"{task:12s} {nseeds:5d} {int(t[-1]):7d}  " + "  ".join(f"{r:>20s}" for r in row))
    ax.set_title(f"{title}", color=INK, fontsize=11, loc="left")
    ax.set_xlabel("Environment steps (M)")
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlim(0, 1.0)
axes[0].set_ylabel("Evaluation return")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
fig.text(0.995, 0.01, f"iBOLT, seeds {','.join(map(str, sorted(SEEDS)))}: mean ± 1 std over seeds; 10 episodes per eval, every 5K steps",
         ha="right", va="bottom", color=INK2, fontsize=8.5)
fig.tight_layout(rect=(0, 0.02, 1, 0.93))
fig.savefig(OUT, dpi=150, facecolor=SURF)
print("saved", OUT)
