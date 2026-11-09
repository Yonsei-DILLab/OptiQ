"""CrossQ comparison curves (Walker2d, Ant, Humanoid), usable while runs are in progress.

Usage: python plot_crossq.py <crossq_eval_root> <ibolt_ref_eval_root> <out_png>
  crossq_eval_root: copies of outputs/*crossq*/ (iBOLT+CrossQ critic and SBX CrossQ runs)
  ibolt_ref_eval_root: original iBOLT 3-mode runs (paper critic), e.g. results/mujoco_3mode_s0-9/eval
Top row: deterministic evaluation (iBOLT zero_z, CrossQ mean action).
Bottom row: full stochastic policy (iBOLT z+sigma, CrossQ sampled action).
Mean +/- 1 std over seeds 0-4, truncated per method to its shortest seed so the
mean never mixes seed counts. Also prints a table of the latest 100K-step window.
"""
import glob, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CQ, REF, OUT = sys.argv[1:4]
TASKS = [("walker2d", "Walker2d-v4"), ("ant", "Ant-v4"), ("humanoid", "Humanoid-v4")]
METHODS = [  # key, label, color, linestyle, deterministic mode, stochastic mode, root, dir pattern
    ("ibolt_cq", "iBOLT + CrossQ critic", "#2a78d6", "-", "zero_z", "stochastic_z_sigma", CQ, "{t}-ibolt-crossq-*"),
    ("crossq", "CrossQ (SBX)", "#eb6834", "--", "deterministic", "stochastic", CQ, "{t}-crossq-q*"),
    ("ibolt", "iBOLT (paper critic)", "#52514e", ":", "zero_z", "stochastic_z_sigma", REF, "{t}-ibolt-T*"),
]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})


def load(root, pattern, mode):
    runs = {}
    for d in glob.glob(os.path.join(root, pattern)):
        s = int(re.search(r"-s(\d+)_", os.path.basename(d)).group(1))
        if s > 4:
            continue
        f = glob.glob(os.path.join(d, "**", f"evaluations_{mode}.npz"), recursive=True)
        if f:
            z = np.load(f[0])
            if len(z["timesteps"]):
                runs[s] = (np.asarray(z["timesteps"]), np.asarray(z["results"]).mean(1))
    return runs


fig, axes = plt.subplots(2, len(TASKS), figsize=(6 * len(TASKS), 7.6), facecolor=SURF, squeeze=False)
table = []
for col, (task, title) in enumerate(TASKS):
    for row, which in enumerate(("det", "sto")):
        ax = axes[row, col]
        ax.set_facecolor(SURF)
        for key, label, color, ls, dmode, smode, root, pat in METHODS:
            runs = load(root, pat.format(t=task), dmode if which == "det" else smode)
            if not runs:
                continue
            # Seeds in progress have different lengths: average over the seeds that have
            # reached each eval step (masked), and draw every seed as a faint line.
            L = max(len(t) for t, _ in runs.values())
            t = max(runs.values(), key=lambda r: len(r[0]))[0]
            Y = np.full((len(runs), L), np.nan)
            for i, (ts, y) in enumerate(runs.values()):
                Y[i, :len(y)] = y
                ax.plot(ts / 1e6, y, color=color, linewidth=0.5, alpha=0.25)
            count = (~np.isnan(Y)).sum(0)
            keep = count >= 2
            mu, sd = np.nanmean(Y[:, keep], 0), np.nanstd(Y[:, keep], 0)
            ax.fill_between(t[keep] / 1e6, mu - sd, mu + sd, color=color, alpha=0.13, linewidth=0)
            ax.plot(t[keep] / 1e6, mu, color=color, linestyle=ls, linewidth=1.8, label=label)
            # Latest window per seed: mean over that seed's own last 100K steps.
            win = np.array([y[ts > ts[-1] - 100_000].mean() for ts, y in runs.values()])
            steps = sorted(int(ts[-1]) for ts, _ in runs.values())
            table.append((task, which, label, len(runs), steps, win.mean(), win.std()))
        mode_txt = "deterministic eval (iBOLT z=0, CrossQ mean action)" if which == "det" else \
                   "stochastic eval (iBOLT z+σ, CrossQ sampled action)"
        ax.set_title(f"{title} — {mode_txt}", color=INK, fontsize=10, loc="left")
        ax.set_xlim(0, 1.0); ax.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if row == 1:
            ax.set_xlabel("Environment steps (M)")
        if col == 0:
            ax.set_ylabel("Evaluation return")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.text(0.995, 0.005, "seeds 0-4; thick: mean ± 1 std over seeds that reached each step (≥2), "
         "faint: individual seeds; critic width 1024x2 for both CrossQ variants",
         ha="right", va="bottom", color=INK2, fontsize=8)
fig.tight_layout(rect=(0, 0.02, 1, 0.95))
fig.savefig(OUT, dpi=140, facecolor=SURF)

print(f"{'task':9s} {'eval':4s} {'method':24s} {'n':>2s}  {'steps per seed (K)':28s} latest-100K mean ± std")
for task, which, label, n, steps, m, s in table:
    st = ",".join(str(x // 1000) for x in steps)
    print(f"{task:9s} {which:4s} {label:24s} {n:2d}  {st:28s} {m:7.0f} ± {s:4.0f}")
print("saved", OUT)
