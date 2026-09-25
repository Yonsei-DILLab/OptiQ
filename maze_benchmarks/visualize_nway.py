"""Automatic N-Way policy, trajectory, value and goal-use figure.

The contour is the task's position-reward term (negative nearest-goal squared
distance), not the full action-dependent reward or learned policy density.
Arrows and trajectories are sampled from the learned policy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .nway import STATE_LIMIT, goal_positions
from .visualize_4way import _visited_cells


def render(task: str, rollout_path: Path, probe_path: Path, output: Path,
           *, title: str = "") -> Path:
    if not task.endswith("way") or not task[:-3].isdigit():
        raise ValueError(task)
    goals_xy = goal_positions(int(task[:-3]))
    with np.load(rollout_path) as data:
        paths = data["xy"]
        goal_ids = data["goal_ids"]
        returns = data["returns"]
    with np.load(probe_path) as data:
        x, y, valid = data["x"], data["y"], data["valid"]
        actions, values = data["actions"], data["q"]
        critic_label = str(data["critic_label"])
    if paths.ndim != 3 or paths.shape[-1] != 2 or goal_ids.shape != (len(paths),):
        raise ValueError("invalid rollout array")
    if np.any((goal_ids >= len(goals_xy)) | (goal_ids < -1)):
        raise ValueError("goal ids outside task range")
    if actions.shape[:2] != valid.shape or values.shape[:2] != valid.shape:
        raise ValueError("invalid policy/Q probe")
    if not np.isfinite(returns).all() or not np.isfinite(values[valid]).all():
        raise ValueError("nonfinite return or Q")
    counts = np.bincount(goal_ids[goal_ids >= 0], minlength=len(goals_xy))
    failures = int((goal_ids < 0).sum())
    colors = plt.get_cmap("hsv")(np.arange(len(goals_xy)) / len(goals_xy))
    grid = np.linspace(-STATE_LIMIT, STATE_LIMIT, 321)
    xx, yy = np.meshgrid(grid, grid)
    reference = -np.square(np.stack((xx, yy), axis=-1)[:, :, None] - goals_xy).sum(axis=-1).min(axis=-1)

    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    layout = fig.add_gridspec(2, 2, height_ratios=(1., .9))
    policy_ax = fig.add_subplot(layout[0, 0])
    trajectory_ax = fig.add_subplot(layout[0, 1])
    q_ax = fig.add_subplot(layout[1, 0], projection="3d")
    histogram_ax = fig.add_subplot(layout[1, 1])

    levels = np.linspace(max(-60., float(reference.min())), 0., 20)
    policy_ax.contour(xx, yy, reference, levels=levels, cmap="viridis", linewidths=.7)
    visited = _visited_cells(paths, x, y) & valid
    rows, cols = np.nonzero(visited)
    choices = (rows * 131 + cols * 17) % actions.shape[-2]
    arrows = actions[rows, cols, choices]
    policy_ax.quiver(x[cols], y[rows], arrows[:, 0], arrows[:, 1], color="#d8392b",
                     angles="xy", scale_units="xy", scale=1.7, width=.004,
                     headwidth=3.5, headlength=5.)
    center = (len(y) // 2, len(x) // 2)
    origins = np.array([[.17, .17], [-.17, .17], [-.17, -.17], [.17, -.17]])
    origin_actions = actions[center[0], center[1], :4]
    policy_ax.quiver(origins[:, 0], origins[:, 1], origin_actions[:, 0], origin_actions[:, 1],
                     color="#d8392b", angles="xy", scale_units="xy", scale=1.7,
                     width=.005, headwidth=3.5, headlength=5.)
    policy_ax.set_title("Sampled policy actions · goal-distance contours")

    # Plot a deterministic subset so dense 32-goal rollouts remain readable.
    shown = np.unique(np.linspace(0, len(paths) - 1, min(len(paths), 256), dtype=int))
    for index in shown:
        route = paths[index]
        route = route[np.isfinite(route).all(axis=-1)]
        color = colors[goal_ids[index]] if goal_ids[index] >= 0 else "#7b7b7b"
        trajectory_ax.plot(route[:, 0], route[:, 1], color=color, alpha=.25, lw=.7)
    trajectory_ax.set_title(f"Direct-policy trajectories · {len(shown)}/{len(paths)} shown")

    for ax in (policy_ax, trajectory_ax):
        ax.scatter(goals_xy[:, 0], goals_xy[:, 1], color=colors, s=24, edgecolors="black",
                   linewidths=.35, zorder=5)
        ax.scatter([0], [0], marker="+", color="black", s=65, zorder=6)
        ax.set(xlim=(-STATE_LIMIT, STATE_LIMIT), ylim=(-STATE_LIMIT, STATE_LIMIT),
               xlabel="x", ylabel="y", aspect="equal")
        ax.grid(alpha=.12)

    qgrid = np.full(valid.shape, np.nan, np.float32)
    qgrid[valid] = values[valid].mean(axis=-1)
    qxx, qyy = np.meshgrid(x, y)
    q_ax.plot_surface(qxx, qyy, np.ma.masked_where(~valid, qgrid),
                      cmap="viridis", linewidth=0, antialiased=True)
    q_ax.set(xlabel="x", ylabel="y", zlabel="Value function",
             title=f"Learned EπQ(s,a) · {critic_label}")
    q_ax.view_init(elev=28, azim=-58)
    q_ax.set_box_aspect((1, 1, .65))

    histogram_ax.bar(np.arange(len(counts)), counts, color=colors, width=.84)
    histogram_ax.set(xlabel="Goal index (counter-clockwise from east)",
                     ylabel="Episodes", title=f"Goals reached: {(counts > 0).sum()}/{len(counts)} · failures {failures}")
    histogram_ax.set_xticks(np.arange(len(counts)))
    histogram_ax.tick_params(axis="x", labelsize=7)
    histogram_ax.grid(axis="y", alpha=.2)
    fig.suptitle(title or f"{task.upper()} · one sampled policy · {len(paths)} rollouts",
                 fontsize=16)
    fig.text(.02, .008, "Contours show negative nearest-goal squared distance, not full reward or policy density. "
             "Q is the policy-averaged learned critic; absolute scales differ across methods.", fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--task", required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="")
    args = parser.parse_args()
    print(render(args.task, args.rollout, args.probe, args.output, title=args.title))


if __name__ == "__main__":
    main()
