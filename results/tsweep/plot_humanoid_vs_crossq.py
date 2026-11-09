"""Humanoid-v4: iBOLT (CrossQ critic + CrossQ-style actor training, T=0.5) vs SBX CrossQ, 5 seeds each.

Run on the 5090:  ~/OptiQ/.venv/bin/python plot_humanoid_vs_crossq.py <out_png>
  iBOLT: ~/OptiQ/outputs_tsweep/humanoid-ibolt-crossqAC-T0.5-s* (+ outputs_pc for the seed run on the PC)
  CrossQ: ~/OptiQ/results/live_refs/humanoid-crossq-q*
Mean over seeds, shaded ±1 std, 25K-step moving average.
"""
import glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HOME = os.path.expanduser("~/OptiQ")
OUT = sys.argv[1] if len(sys.argv) > 1 else "humanoid_vs_crossq.png"
TASK = sys.argv[2] if len(sys.argv) > 2 else "humanoid"
INK, INK2, GRID, SURF = "#1f1e1c", "#52514e", "#e4e3df", "#fcfcfb"
# task -> (title, iBOLT temperature, iBOLT run dirs, SBX CrossQ run dirs)
# walker2d SBX: 5090 outputs/ + 4090 runs copied into /tmp/hw/4090
CFG = {
    "humanoid": ("Humanoid-v4", "0.5",
                 [os.path.join(HOME, "outputs_tsweep/humanoid-ibolt-crossqAC-T0.5-s*"),
                  os.path.join(HOME, "outputs_pc/humanoid-ibolt-crossqAC-T0.5-s*")],
                 [os.path.join(HOME, "results/live_refs/humanoid-crossq-q*")]),
    "walker2d": ("Walker2d-v4", "0.1",
                 [os.path.join(HOME, "outputs_tsweep/walker2d-ibolt-crossqAC-T0.1-s*")],
                 [os.path.join(HOME, "outputs/walker2d-crossq-q1024-s*"), "/tmp/hw/4090/walker2d-crossq-q1024-s*"]),
}
TITLE, TEMP, IBOLT, SBX = CFG[TASK]


def load(patterns, mode):
    runs = []
    for p in patterns:
        for d in sorted(glob.glob(p)):
            f = glob.glob(os.path.join(d, "**", f"evaluations_{mode}.npz"), recursive=True)
            if f:
                z = np.load(f[0])
                runs.append((np.asarray(z["timesteps"]), np.asarray(z["results"]).mean(1)))
    return runs


def smooth(y, k=5):
    return np.convolve(np.pad(y, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")


def band(ax, runs, color, ls, label):
    L = min(len(t) for t, _ in runs)
    t = runs[0][0][:L]
    Y = np.stack([smooth(y[:L]) for _, y in runs])
    mu, sd = Y.mean(0), Y.std(0)
    ax.fill_between(t / 1e6, mu - sd, mu + sd, color=color, alpha=0.15, linewidth=0)
    ax.plot(t / 1e6, mu, color=color, linestyle=ls, linewidth=2, label=label)


def last100(runs):
    v = [y[t > t[-1] - 100_000].mean() for t, y in runs]
    return np.mean(v), np.std(v)


fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURF, sharey=True)
# (source, eval mode, color, linestyle, label) per curve
panels = [("Mean action (no action noise)", [
              (SBX, "deterministic", "#eb6834", "--", "CrossQ, tanh(μ)"),
              (IBOLT, "zero_z", "#2a78d6", "-", "iBOLT, z = 0"),
              (IBOLT, "stochastic_z", "#1a9e3f", "-", "iBOLT, z ~ N(0, I)")]),
          ("Full stochastic policy (μ + σ)", [
              (SBX, "stochastic", "#eb6834", "--", "CrossQ, sampled"),
              (IBOLT, "stochastic_z_sigma", "#2a78d6", "-", "iBOLT, z ~ N(0, I) + σ")])]
for ax, (title, curves) in zip(axes, panels):
    ax.set_facecolor(SURF)
    for src, mode, color, ls, lab in curves:
        runs = load(src, mode)
        band(ax, runs, color, ls, f"{lab} ({len(runs)} seeds)")
        a, sa = last100(runs)
        print(f"{title:32s} {lab:24s} {a:6.0f} ± {sa:4.0f}   (latest 100K)")
    ax.set_title(title, fontsize=11, loc="left", color=INK)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Environment steps (M)", color=INK2)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(colors=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
axes[0].set_ylabel("Evaluation return", color=INK2)
fig.suptitle(f"{TITLE} — iBOLT (CrossQ critic + actor training, T = {TEMP}) vs CrossQ",
             fontsize=12, x=0.01, ha="left", color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUT, dpi=150, facecolor=SURF)
fig.savefig(os.path.splitext(OUT)[0] + ".pdf", facecolor=SURF)
