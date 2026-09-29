"""Plot the latest available control trajectories for v1, v3, and v4."""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from antmaze_experiments.progress_reward import maze_geometry


ROOT = Path(__file__).resolve().parent
MODE = sys.argv[1] if len(sys.argv) > 1 else "control"
assert MODE in ("control", "native")
RUNS = (
    ("v1", 100096, ROOT / f"v1_{MODE}_100096.npz", ((-12, 4), (-8, 8))),
    ("v3", 50176, ROOT / f"v3_{MODE}_50176.npz", ((-18, 18), (-18, 18))),
    ("v4", 100096, ROOT / f"v4_{MODE}_100096.npz", ((-22, 6), (-14, 14))),
)


fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.5), constrained_layout=True)
for ax, (task, step, path, limits) in zip(axes, RUNS):
    archive = np.load(path, allow_pickle=False)
    assert int(archive["env_steps"]) == step and archive["xy"].shape[0] == 40
    trajectories = [archive["xy"][i, :n + 1] for i, n in enumerate(archive["lengths"])]
    walls, goals, _ = maze_geometry(task)
    for x0, y0, x1, y1 in walls:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                               facecolor="#c1c8ce", edgecolor="#87919a", lw=.45))
    for xy in trajectories:
        ax.plot(xy[:, 0], xy[:, 1], color="#17689c", lw=.9, alpha=.34)
    starts = np.array([xy[0] for xy in trajectories])
    ends = np.array([xy[-1] for xy in trajectories])
    ax.scatter(starts[:, 0], starts[:, 1], s=10, color="black", alpha=.7)
    ax.scatter(ends[:, 0], ends[:, 1], s=19, marker="x", color="#ba2630", lw=.75)
    ax.scatter(goals[:, 0], goals[:, 1], s=150, marker="*", color="#f1bd36",
               edgecolor="black", lw=.5)
    if task == "v1":
        ax.add_patch(Rectangle((-2, -2), 4, 4, fill=False, ls="--", lw=1.1,
                               edgecolor="#555"))
        upper = sum(((xy[:, 0] < -2) & (xy[:, 1] > 2)).any() for xy in trajectories)
        lower = sum(((xy[:, 0] < -2) & (xy[:, 1] < -2)).any() for xy in trajectories)
        label = f"upper/lower entry {upper}/{lower}"
    elif task == "v3":
        left = sum((xy[:, 0] < -2).any() for xy in trajectories)
        right = sum((xy[:, 0] > 2).any() for xy in trajectories)
        label = f"left/right entry {left}/{right}"
    else:
        upper = sum(((xy[:, 0] < -2) & (xy[:, 1] > 2)).any() for xy in trajectories)
        lower = sum(((xy[:, 0] < -2) & (xy[:, 1] < -2)).any() for xy in trajectories)
        label = f"upper/lower entry {upper}/{lower}"
    successes = int(np.sum(archive["goals"] > 0))
    ax.text(.02, .03, f"{successes}/40 success · {label}", transform=ax.transAxes,
            va="bottom", fontsize=9,
            bbox=dict(facecolor="white", alpha=.86, edgecolor="none", pad=2))
    ax.set_title(f"{task} · {step:,} steps", fontsize=12)
    ax.set_xlim(*limits[0]); ax.set_ylim(*limits[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=.1)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")

description = ("random z + conditional σ" if MODE == "control" else "random z, μ-only (no conditional σ)")
fig.suptitle("OptiQ · 100 × distance decrease · latest available trajectories\n"
             f"{description} · 40 rollouts per maze · v1 random start; v3/v4 fixed start",
             fontsize=12.5, fontweight="bold")
stem = "v1_v3_v4_latest_control_trajectories" if MODE == "control" else "v1_v3_v4_latest_mu_only_trajectories"
fig.savefig(ROOT / f"{stem}.png", dpi=180, bbox_inches="tight")
fig.savefig(ROOT / f"{stem}.pdf", bbox_inches="tight")
