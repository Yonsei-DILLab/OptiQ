"""Matched iBOLT (N=32768) vs SAC 1M table + K(tau) figure. Same states, same policy-sample count M=32768, k=15.
Run in results/modal_em (iBOLT venv). Writes figs/cmp32k_summary.txt, figs/cmp32k_Ktau.png/.pdf
"""
import glob, pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dddcd7"
COL = {"iBOLT": "#2a78d6", "SAC seed0": "#eb6834", "SAC seed1": "#1baf7a"}
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
ENVS = {"Hopper": "hopper", "Humanoid": "humanoid"}
POL = {"iBOLT": "bp32k/ibolt_{e}_s{s}.pkl", "SAC seed0": "bp32k/sac1m0_{e}_s{s}.pkl", "SAC seed1": "bp32k/sac1m1_{e}_s{s}.pkl"}
H = (f"{'env':9s} {'policy':22s} {'states':>6s} {'maxima':>6s} " + " ".join(f"{'K@' + str(t):>6s}" for t in [0.25, 0.5, 1, 2])
     + f" {'K>=2@.5':>8s} {'K>=2@1':>7s} | {'K5@.5':>6s} {'Meff@.5':>7s} {'K5@1':>5s} {'Meff@1':>6s}")


def row(env, name, R):
    K = Kt(R); tm = lambda t, k: np.mean([r["tomato"][t][k] for r in R])
    return K, (f"{env:9s} {name:22s} {len(R):6d} {np.mean([r['n_modes'] for r in R]):6.2f} "
               + " ".join(f"{at(K, t).mean():6.2f}" for t in [0.25, 0.5, 1, 2])
               + f" {(at(K, .5) >= 2).mean():8.3f} {(at(K, 1) >= 2).mean():7.3f} | {tm(.5, 'K5'):6.2f} {tm(.5, 'meff'):7.2f}"
               + f" {tm(1.0, 'K5'):5.2f} {tm(1.0, 'meff'):6.2f}")


L = ["Matched comparison: iBOLT 1M (mixture N=32768 latents) vs SAC 1M (exact tanh-Gaussian density), same states",
     "(iBOLT zero-z evaluation rollouts; Hopper 5 seeds x 64, Humanoid 5 seeds x 32), same policy-sample count M=32768, kNN k=15.",
     "Basin-graph H0 persistence. K(tau) = 1 + #maxima with persistence >= tau nats (peak/saddle >= e^tau) and peak >= 1% of max.",
     "ToMATo@tau: K5 = clusters with >= 5% of policy samples; M_eff = exp(entropy of cluster masses).", "", H]
data = {}
for env, e in ENVS.items():
    for name, pat in POL.items():
        R = load(pat.format(e=e, s="[0-9]"))
        if R:
            K, s = row(env, name, R); data[(env, name)] = K; L.append(s)
    L.append("")
L += ["per seed (iBOLT):", H]
for env, e in ENVS.items():
    for s in range(5):
        R = load(POL["iBOLT"].format(e=e, s=s))
        if R:
            L.append(row(env, f"iBOLT s{s}", R)[1])
L += ["", "checks: extra graph components / points whose ascent ended > 1e-2 from a verified mode"]
for env, e in ENVS.items():
    for name, pat in POL.items():
        R = load(pat.format(e=e, s="[0-9]"))
        if R:
            L.append(f"   {env:9s} {name:12s} extra comps/state {np.mean([r['n_components'] - 1 for r in R]):.3f}  "
                     f"far points/state {np.mean([r['far'] for r in R]):.1f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
for ax, env in zip(axes, ENVS):
    for name in POL:
        if (env, name) not in data:
            continue
        K = data[(env, name)]
        ax.plot(TG, K.mean(0), color=COL[name], lw=2.2 if name == "iBOLT" else 1.6, label=name)
        if name == "iBOLT":
            ax.fill_between(TG, np.percentile(K, 25, 0), np.percentile(K, 75, 0), color=COL[name], alpha=0.12, lw=0)
    ax.set_xscale("log"); ax.grid(True, color=GRID, lw=0.6); ax.set_ylim(bottom=0.8)
    ax.set_xlabel("persistence threshold τ (nats)"); ax.set_title(env, loc="left")
    sec = ax.secondary_xaxis("top", functions=(np.exp, lambda r: np.log(np.maximum(r, 1e-9))))
    sec.set_xticks([1.05, 1.5, 2, 5, 20]); sec.set_xticklabels(["1.05", "1.5", "2", "5", "20"])
    sec.set_xlabel("peak / saddle density ratio", fontsize=8)
    ax.legend(frameon=False, fontsize=8)
axes[0].set_ylabel("modes per state K(τ)")
fig.tight_layout(); fig.savefig("figs/cmp32k_Ktau.png", dpi=180); fig.savefig("figs/cmp32k_Ktau.pdf"); plt.close(fig)
Path("figs/cmp32k_summary.txt").write_text("\n".join(L))
print("\n".join(L))
