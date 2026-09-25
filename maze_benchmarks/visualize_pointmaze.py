"""Actual sampled-policy trajectories and the original DrAC maze geometry."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from pointmaze.drac_paper import MAP_NAMES, upstream_modules


def plot_map(ax, maze: str, obstacle: bool = False):
    if maze not in MAP_NAMES:
        raise ValueError(maze)
    get_map, _ = upstream_modules()
    cells = get_map(maze)
    height, width = len(cells), len(cells[0])
    for row, values in enumerate(cells):
        for col, value in enumerate(values):
            x, y = col - width / 2, height / 2 - row - 1
            if value == 1:
                color = "#777777"
            elif value == 2:
                color = "#c9c9c9" if obstacle else "#eeeeee"
            elif value == "g":
                color = "#39c947"
            else:
                continue
            ax.add_patch(Rectangle((x, y), 1, 1, color=color, linewidth=0, zorder=0))
            if value == "g":
                ax.plot(x + .5, y + .5, "kx", ms=4, zorder=4)
            if value == 2 and not obstacle:
                ax.add_patch(Rectangle((x, y), 1, 1, fill=False, edgecolor="#cccccc",
                                       linestyle=":", linewidth=.4, zorder=1))
            if value == 2 and obstacle:
                ax.plot(x + .5, y + .5, "kx", ms=3, zorder=3)

    for row, values in enumerate(cells):
        for col, value in enumerate(values):
            if value == "r":
                ax.plot(col + .5 - width / 2, height / 2 - row - .5,
                        "o", color="red", markersize=6, zorder=5)
    ax.set(xlim=(-width / 2, width / 2), ylim=(-height / 2, height / 2),
           aspect="equal", xlabel="x", ylabel="y")
    ax.grid(False)


def plot_rollouts(ax, data: Path, max_trajectories: int = 100):
    with np.load(data) as values:
        tracks = values["xy"]
        goals = values["goal_ids"]
        returns = values["returns"]
    for index in range(min(max_trajectories, len(tracks))):
        valid = np.isfinite(tracks[index]).all(axis=-1)
        track = tracks[index, valid]
        if len(track):
            ax.plot(track[:, 0], track[:, 1], color="#da22d3", linewidth=.6,
                    alpha=.28, zorder=2)
    return dict(episodes=len(goals), success=float(np.mean(goals >= 0)),
                goals=np.bincount(goals[goals >= 0], minlength=int(goals.max() + 1)).tolist()
                if np.any(goals >= 0) else [], mean_return=float(np.mean(returns)))


def render_trajectories(task: str, policy_npz: Path, destination: Path,
                        title: str, obstacle_npz: Path | None = None):
    maze = task.removeprefix("pm_")
    if maze not in MAP_NAMES:
        raise ValueError(task)
    panels = [(policy_npz, False, "Training maze")]
    if obstacle_npz is not None:
        panels.append((obstacle_npz, True, "Unseen obstacles"))
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5.4),
                             squeeze=False, constrained_layout=True)
    for ax, (source, obstacle, subtitle) in zip(axes[0], panels):
        plot_map(ax, maze, obstacle)
        result = plot_rollouts(ax, source)
        ax.set_title(f"{subtitle} · success {result['success']:.1%}")
    fig.suptitle(title, fontsize=11)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160)
    plt.close(fig)
