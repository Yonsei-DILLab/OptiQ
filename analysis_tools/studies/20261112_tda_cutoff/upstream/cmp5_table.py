"""5-seed table: iBOLT 1M (N=32768) vs SAC 1M, mean +- std over training seeds. Run in results/modal_em (iBOLT venv).
iBOLT seed i: its own 64 (Hopper) / 32 (Humanoid) evaluation states. SAC seed j: pooled over all 5 iBOLT state sets
(Hopper 320, Humanoid 160 states). Same policy-sample count M=32768, kNN k=15, basin-graph H0 persistence.
Writes figs/cmp5_summary.txt, figs/cmp5_Ktau.png/.pdf
"""
import glob, pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dddcd7"
COL = {"iBOLT": "#2a78d6", "SAC": "#eb6834"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#b4b2aa", "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "pdf.fonttype": 42})
TG = np.logspace(-2, np.log10(4), 60)
HF = np.log(100.0)


def load(pat):
    R = []
    for f in sorted(glob.glob(pat)):
        R += pickle.load(open(f, "rb"))["res"]
    return R


def Kt(R):
    return np.array([[1 + ((r["pers"] >= t) & (r["pairs"][:, 0] >= r["fmax"] - HF)).sum() + (r["n_components"] - 1)
                      for t in TG] for r in R])


at = lambda K, t: K[:, np.argmin(abs(TG - t))]
METRICS = ["K@0.5", "K@1", "K@2", "K>=2@0.5", "K>=2@1", "K5@0.5", "Meff@0.5", "K5@1", "Meff@1"]


def seed_metrics(R):
    K = Kt(R); tm = lambda t, k: float(np.mean([r["tomato"][t][k] for r in R]))
    return dict(zip(METRICS, [at(K, .5).mean(), at(K, 1).mean(), at(K, 2).mean(), (at(K, .5) >= 2).mean(), (at(K, 1) >= 2).mean(),
                              tm(.5, "K5"), tm(.5, "meff"), tm(1.0, "K5"), tm(1.0, "meff")])), K.mean(0), len(R)


ENVS = {"Hopper": "hopper", "Humanoid": "humanoid"}
POL = {"iBOLT": lambda e, i: f"bp32k/ibolt_{e}_s{i}.pkl", "SAC": lambda e, i: f"bp32k/sac1m{i}_{e}_s[0-9].pkl"}
L = ["iBOLT 1M (mixture N=32768) vs SAC 1M (exact tanh-Gaussian), mean +- std over training seeds.",
     "iBOLT seed i on its own evaluation states (Hopper 64, Humanoid 32); SAC seed j on all 5 iBOLT state sets (Hopper 320, Humanoid 160).",
     "Same policy-sample count M=32768, kNN k=15. K(tau) = 1 + #maxima with persistence >= tau nats and peak >= 1% of max;",
     "K>=2 = fraction of states with >= 2 such modes; K5 = ToMATo clusters with >= 5% mass; M_eff = exp(entropy of cluster masses).",
     "", f"{'env':9s} {'policy':6s} {'seeds':>5s} " + " ".join(f"{m:>13s}" for m in METRICS)]
curves, per = {}, []
for env, e in ENVS.items():
    for name, pf in POL.items():
        S = [seed_metrics(R) for R in (load(pf(e, i)) for i in range(5)) if R]
        if not S:
            continue
        curves[(env, name)] = np.array([c for _, c, _ in S])
        L.append(f"{env:9s} {name:6s} {len(S):5d} " + " ".join(
            f"{np.mean([m[k] for m, _, _ in S]):6.2f}±{np.std([m[k] for m, _, _ in S], ddof=1) if len(S) > 1 else 0:<6.2f}" for k in METRICS))
        for i, (m, _, n) in enumerate(S):
            per.append(f"{env:9s} {name:6s} s{i} ({n:3d} st) " + " ".join(f"{m[k]:13.2f}" for k in METRICS))
    L.append("")
L += ["per seed:", f"{'env':9s} {'policy':6s} {'seed':12s} " + " ".join(f"{m:>13s}" for m in METRICS)] + per

fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
for ax, env in zip(axes, ENVS):
    for name in POL:
        if (env, name) not in curves:
            continue
        C = curves[(env, name)]; mu = C.mean(0); sd = C.std(0, ddof=1) if len(C) > 1 else 0 * mu
        ax.plot(TG, mu, color=COL[name], lw=2, label=f"{name} ({len(C)} seeds)")
        ax.fill_between(TG, mu - sd, mu + sd, color=COL[name], alpha=0.15, lw=0)
    ax.set_xscale("log"); ax.grid(True, color=GRID, lw=0.6); ax.set_ylim(bottom=0.8)
    ax.set_xlabel("persistence threshold τ (nats)"); ax.set_title(env, loc="left")
    sec = ax.secondary_xaxis("top", functions=(np.exp, lambda r: np.log(np.maximum(r, 1e-9))))
    sec.set_xticks([1.05, 1.5, 2, 5, 20]); sec.set_xticklabels(["1.05", "1.5", "2", "5", "20"])
    sec.set_xlabel("peak / saddle density ratio", fontsize=8)
    ax.legend(frameon=False, fontsize=8)
axes[0].set_ylabel("modes per state K(τ), mean ± std over seeds")
fig.tight_layout(); fig.savefig("figs/cmp5_Ktau.png", dpi=180); fig.savefig("figs/cmp5_Ktau.pdf"); plt.close(fig)
Path("figs/cmp5_summary.txt").write_text("\n".join(L))
print("\n".join(L))
