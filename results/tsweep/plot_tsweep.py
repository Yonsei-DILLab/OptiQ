"""Humanoid temperature sweep (iBOLT + CrossQ critic 1024): curves per T, mean of seeds.

Usage: python plot_tsweep.py <tsweep_eval_root> <ref_eval_root> <out_png> [task=humanoid]
  tsweep_eval_root: copies of outputs_tsweep/*/eval ; ref_eval_root: SBX CrossQ evals (+ humanoid: old T=0.1
  CrossQ-critic runs; halfcheetah: paper-critic iBOLT T=0.25 runs)
"""
import glob, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TS, CQ, OUT = sys.argv[1:4]
TASK = sys.argv[4] if len(sys.argv) > 4 else "humanoid"
TITLE = dict(humanoid="Humanoid-v4", halfcheetah="HalfCheetah-v4")[TASK]
INK2, GRID, SURF = "#52514e", "#e4e3df", "#fcfcfb"


def load(pattern, mode):
    runs = []
    for d in glob.glob(pattern):
        f = glob.glob(os.path.join(d, "**", f"evaluations_{mode}.npz"), recursive=True)
        if f:
            z = np.load(f[0])
            if len(z["timesteps"]):
                runs.append((np.asarray(z["timesteps"]), np.asarray(z["results"]).mean(1)))
    return runs


def curve(ax, runs, color, ls, label, lw=1.8):
    if not runs:
        return
    L = max(len(t) for t, _ in runs)
    t = max(runs, key=lambda r: len(r[0]))[0]
    Y = np.full((len(runs), L), np.nan)
    for i, (_, y) in enumerate(runs):
        Y[i, :len(y)] = y
    keep = (~np.isnan(Y)).sum(0) >= min(2, len(runs))
    mu = np.nanmean(Y[:, keep], 0)
    k = 5  # 25K-step moving average for readability
    mu_s = np.convolve(np.pad(mu, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")
    ax.plot(t[keep] / 1e6, mu_s, color=color, linestyle=ls, linewidth=lw, label=label)


temps = sorted({float(re.search(r"-T([0-9.]+)-s", os.path.basename(d)).group(1))
                for d in glob.glob(os.path.join(TS, f"{TASK}-ibolt-crossq-T*"))})
cmap = plt.get_cmap("viridis")
COLORS = {0.05: "#9b59b6", 0.1: "#2a78d6", 0.25: "#17becf", 0.5: "#1a9e3f", 0.75: "#d4a017",
          1.0: "#c0392b", 5.0: "#8c564b"}
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), facecolor=SURF)
for ax, (mode, sbx_mode, title) in zip(axes, [("zero_z", "deterministic", "deterministic eval (z = 0)"),
                                               ("stochastic_z_sigma", "stochastic", "full stochastic policy (z + σ)")]):
    ax.set_facecolor(SURF)
    r = load(os.path.join(CQ, f"{TASK}-crossq-q*"), sbx_mode)
    curve(ax, r, "#eb6834", "--", f"CrossQ (SBX), {len(r)} seeds")
    if TASK == "humanoid":
        curve(ax, load(os.path.join(CQ, "humanoid-ibolt-crossq-T0.1-s*"), mode), "#8a8985", ":", "earlier T=0.1, 5 seeds")
    else:
        r = load(os.path.join(CQ, f"{TASK}-ibolt-T0.25-s*"), mode)
        curve(ax, r, "#8a8985", ":", f"paper critic T=0.25, {len(r)} runs")
    for i, T in enumerate(temps):
        runs = load(os.path.join(TS, f"{TASK}-ibolt-crossq-T{T}-s*"), mode)
        curve(ax, runs, COLORS.get(T, cmap(i / max(1, len(temps) - 1))), "-", f"T={T:g} ({len(runs)} seeds)")
    r = load(os.path.join(TS, f"{TASK}-ibolt-crossqAC-T0.5-s*"), mode)
    curve(ax, r, "#111111", "-", f"T=0.5 + CrossQ actor training ({len(r)} seeds)", lw=2.4)
    ax.axvline(0.5, color=GRID, linewidth=1.2)
    ax.set_title(f"{TITLE}, iBOLT + CrossQ critic — {title}", fontsize=10, loc="left")
    ax.set_xlim(0, 1); ax.set_xlabel("Environment steps (M)"); ax.grid(True, color=GRID, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("Evaluation return (25K moving average)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=5, frameon=False, fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(OUT, dpi=130, facecolor=SURF)

def last100(rs):
    return [y[t > t[-1] - 100_000].mean() for t, y in rs]


for lab, pat, m0, m1 in [("SBX", f"{TASK}-crossq-q*", "deterministic", "stochastic")] + (
        [("paper", f"{TASK}-ibolt-T0.25-s*", "zero_z", "stochastic_z_sigma")] if TASK != "humanoid" else []):
    a, b = load(os.path.join(CQ, pat), m0), load(os.path.join(CQ, pat), m1)
    if a:
        print(f"{lab:>5s} {len(a):5d}  latest-100K {np.mean(last100(a)):7.0f} ± {np.std(last100(a)):5.0f}"
              f"  {np.mean(last100(b)):7.0f} ± {np.std(last100(b)):5.0f}")
r0 = load(os.path.join(TS, f"{TASK}-ibolt-crossqAC-T0.5-s*"), "zero_z")
r1 = load(os.path.join(TS, f"{TASK}-ibolt-crossqAC-T0.5-s*"), "stochastic_z_sigma")
if r0:
    print(f"  0.5+CrossQ actor {len(r0)} seeds, steps {','.join(str(int(t[-1]) // 1000) for t, _ in r0)}K  "
          f"latest-100K {np.mean(last100(r0)):7.0f} ± {np.std(last100(r0)):5.0f}  {np.mean(last100(r1)):7.0f} ± {np.std(last100(r1)):5.0f}")
print(f"{'T':>5s} {'seeds':>5s} {'steps (K)':>12s}  {'latest-100K z=0':>16s}  {'latest-100K z+σ':>16s}")
for T in temps:
    r0 = load(os.path.join(TS, f"{TASK}-ibolt-crossq-T{T}-s*"), "zero_z")
    r1 = load(os.path.join(TS, f"{TASK}-ibolt-crossq-T{T}-s*"), "stochastic_z_sigma")
    w = lambda rs: [y[t > t[-1] - 100_000].mean() for t, y in rs]
    print(f"{T:5g} {len(r0):5d} {','.join(str(int(t[-1]) // 1000) for t, _ in r0):>12s}  "
          f"{np.mean(w(r0)):7.0f} ± {np.std(w(r0)):5.0f}  {np.mean(w(r1)):7.0f} ± {np.std(w(r1)):5.0f}")
