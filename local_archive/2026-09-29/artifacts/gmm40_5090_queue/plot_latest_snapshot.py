"""Render immutable 100k samples without filtering or mixing source settings."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("snapshot", type=Path)
root = parser.parse_args().snapshot
data = json.loads((root / "snapshot.json").read_text())
target = json.loads((root / "target.json").read_text())
means, std = np.asarray(target["means"]), np.asarray(target["std"])
runs = {(r["job"]["method"], r["job"]["seed"]): r for r in data["runs"]}
methods = data["manifest"]["plan"]["methods"]
labels = dict(optiq_trg="OptiQ Direct GMM / TRG", sac="SAC", dipo="DIPO",
              meow="MEOW", mfpo="MFPO", sql="SQL (JAX SVGD)")
out = root / "figures"
out.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
blue, orange, ink = "#2381b4", "#e9884f", "#17242e"
theta = np.linspace(0, 2 * np.pi, 100)


def panel(ax, method, seed):
    run = runs[method, seed]
    assert run["queue"]["status"] == "completed"
    metric = next(row for row in run["history"] if row["step"] == 100000)
    samples = np.load(root / "samples" / run["job"]["name"] / "100000/samples.npy")
    assert samples.shape == (10000, 2) and np.isfinite(samples).all()
    distance = (((samples[:, None, :] - means) / std[None, :, None]) ** 2).sum(-1)
    near = distance.min(-1) <= 9
    assert np.isclose(near.mean(), metric["high_density_fraction"])
    for selected, color in ((~near, orange), (near, blue)):
        ax.scatter(*samples[selected].T, s=1.3, c=color, alpha=.40, rasterized=True)
    for center, sigma in zip(means, std):
        ax.plot(center[0] + 3 * sigma * np.cos(theta),
                center[1] + 3 * sigma * np.sin(theta),
                color="#818990", ls="--", lw=.65)
    ax.scatter(*means.T, marker="+", s=28, c=ink, lw=1.15)
    ax.set(xlim=(-42, 42), ylim=(-42, 42), aspect="equal", xlabel="Action x₁",
           ylabel="Action x₂", xticks=[-40, -20, 0, 20, 40], yticks=[-40, -20, 0, 20, 40])
    ax.grid(alpha=.12)
    ax.set_title(f"{labels[method]} · seed {seed}\n"
                 f"coverage {metric['mode_coverage']}/40 · near {near.mean():.2%}\n"
                 f"MMD² {metric['mmd2']:.4f} · mass TV {metric['mode_mass_tv']:.3f}",
                 fontsize=11.5, pad=10)
    return {**metric, "method": method, "seed": seed}


def figure(items, columns, title, subtitle, filename):
    fig, axes = plt.subplots(2, columns, figsize=(5.1 * columns, 11.3))
    rows = [panel(ax, *item) for ax, item in zip(axes.flat, items)]
    fig.suptitle(title, fontsize=19, fontweight="bold", y=.982)
    fig.text(.5, .947, subtitle, ha="center", fontsize=11, color="#495866")
    fig.subplots_adjust(top=.865, bottom=.12, left=.065, right=.98,
                        wspace=.24, hspace=.43)
    handles = [Line2D([], [], marker="o", linestyle="", color=blue, label="Within any GT 3σ"),
               Line2D([], [], marker="o", linestyle="", color=orange, label="Outside all GT 3σ"),
               Line2D([], [], marker="+", linestyle="", color=ink, label="GT component centers")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .055),
               ncol=3, frameon=False, fontsize=10)
    fig.text(.5, .035, "All 10,000 IID full-policy samples per panel · dashed circles: GT 3σ", ha="center", fontsize=9)
    fig.text(.5, .015, "Original frozen run 87d5d8f · OptiQ mean init=1e−4, log σ=[−5, −1], initial log σ=−1", ha="center", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{filename}.{ext}", dpi=160, facecolor="white")
    plt.close(fig)
    return rows


seed0 = figure([(m, 0) for m in methods], 3, "GMM40 · all six algorithms · seed 0 · 100k updates",
               "Fixed Q · T=1 · batch=256 · native baseline architectures and compute differ",
               "all_six_seed0_100k")
optiq = figure([("optiq_trg", s) for s in range(4)], 2, "OptiQ Direct GMM / TRG · four seeds · 100k updates",
               "N=M=64 · random latent · 256×2 network · T=1 · batch=256",
               "optiq_four_seeds_100k")
keys = ("mode_coverage", "high_density_fraction", "mmd2", "mode_mass_tv")
summary = {
    "source_commit": data["manifest"]["source_commit"], "captured_at": data["captured_at"],
    "controller_counts_at_snapshot_start": {k: len(data["status"][k]) for k in ("completed", "running", "queued", "failed")},
    "seed0_100k": [{k: row[k] for k in ("method", "seed", *keys)} for row in seed0],
    "optiq_four_seed_100k": {key: {"mean": float(np.mean([r[key] for r in optiq])),
                                   "sample_sd": float(np.std([r[key] for r in optiq], ddof=1))} for key in keys},
    "completed_100k_seeds": {m: [r["job"]["seed"] for r in data["runs"] if r["job"]["method"] == m
                                  and r["queue"]["status"] == "completed" and r["latest"]["step"] == 100000] for m in methods}
}
(out / "summary_latest.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
print(out)
