"""Actual sampled-policy trajectories and the original DrAC maze geometry."""

from __future__ import annotations

from pathlib import Path
import importlib.util

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import numpy as np
from .plot_style import TRAJECTORY_ALPHA, TRAJECTORY_COLOR, TRAJECTORY_LINEWIDTH
from .plot_style import (SUCCESS_TRAJECTORY_COLOR, FAILURE_TRAJECTORY_COLOR,
                         VISITED_GOAL_COLOR, UNVISITED_GOAL_COLOR, UNKNOWN_GOAL_COLOR,
                         GOAL_BORDER_COLOR, GOAL_BORDER_LINEWIDTH)

MAP_NAMES = ("simple", "medium", "hard")


def get_map(name: str):
    # Reporting can run on a CPU-only machine without MuJoCo/Gymnasium.
    source = Path(__file__).resolve().parents[1] / "pointmaze/drac_upstream/envs/mgmaze/maps.py"
    spec = importlib.util.spec_from_file_location("drac_paper_maps_reporting", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_map(name)


def plot_map(ax, maze: str, obstacle: bool = False, *, goal_counts=None,
             outcome_colors: bool = False, visited_goal_color: str = VISITED_GOAL_COLOR,
             unvisited_goal_color: str = UNVISITED_GOAL_COLOR,
             show_inactive_obstacles: bool = False):
    if maze not in MAP_NAMES:
        raise ValueError(maze)
    cells = get_map(maze)
    height, width = len(cells), len(cells[0])
    goal_count = sum(value == "g" for row in cells for value in row)
    if goal_counts is not None:
        goal_counts = np.asarray(goal_counts)
        if (goal_counts.shape != (goal_count,) or not np.isfinite(goal_counts).all()
                or np.any(goal_counts < 0) or np.any(goal_counts != goal_counts.astype(int))):
            raise ValueError("goal counts must match the row-major map goal IDs")
    goal_index = 0
    for row, values in enumerate(cells):
        for col, value in enumerate(values):
            x, y = col - width / 2, height / 2 - row - 1
            if value == 1:
                color = "#777777"
            elif value == 2:
                if not obstacle and not show_inactive_obstacles:
                    continue
                color = "#c9c9c9" if obstacle else "#eeeeee"
            elif value == "g":
                color = (UNKNOWN_GOAL_COLOR if goal_counts is None else
                         visited_goal_color if goal_counts[goal_index] > 0 else
                         unvisited_goal_color) if outcome_colors else "#39c947"
                goal_index += 1
            else:
                continue
            outlined_goal = value == "g" and outcome_colors
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor=color,
                                   edgecolor=GOAL_BORDER_COLOR if outlined_goal else color,
                                   linewidth=GOAL_BORDER_LINEWIDTH if outlined_goal else 0,
                                   joinstyle="miter", zorder=3 if outlined_goal else 0))
            if value == "g":
                marker_color = ("white" if np.dot(to_rgb(color), [.299,.587,.114]) < .6
                                else "black") if outcome_colors and goal_counts is not None else "black"
                ax.plot(x + .5, y + .5, "x", ms=4, zorder=4,
                        color=marker_color)
            if value == 2 and not obstacle:
                ax.add_patch(Rectangle((x, y), 1, 1, fill=False, edgecolor="#cccccc",
                                       linestyle=":", linewidth=.4, zorder=1))
            if value == 2 and obstacle:
                ax.plot(x + .5, y + .5, "kx", ms=3, zorder=3)

    for row, values in enumerate(cells):
        for col, value in enumerate(values):
            if value == "r":
                ax.plot(col + .5 - width / 2, height / 2 - row - .5,
                        "o", color="#222222" if outcome_colors else "red", markersize=6, zorder=5)
    ax.set(xlim=(-width / 2, width / 2), ylim=(-height / 2, height / 2),
           aspect="equal", xlabel="x", ylabel="y")
    ax.grid(False)


def plot_rollouts(ax, data: Path, max_trajectories: int = 500,
                  alpha: float = TRAJECTORY_ALPHA, *, outcome_colors: bool = False,
                  success_color: str = SUCCESS_TRAJECTORY_COLOR,
                  failure_color: str = FAILURE_TRAJECTORY_COLOR,
                  trajectory_border_width: float = 0., hide_failed_trajectories: bool = False):
    if not np.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("trajectory alpha must lie in (0,1]")
    if not np.isfinite(trajectory_border_width) or trajectory_border_width < 0:
        raise ValueError("trajectory border width must be finite and nonnegative")
    with np.load(data) as values:
        tracks = values["xy"]
        goals = values["goal_ids"]
        returns = values["returns"]
    selected_indices = [i for i in range(min(max_trajectories,len(tracks)))
                        if not hide_failed_trajectories or goals[i]>=0]
    selected_tracks = [tracks[i,np.isfinite(tracks[i]).all(axis=-1)] for i in selected_indices]
    if trajectory_border_width:
        # All outlines are below all colored paths: a later white stroke must
        # never erase an earlier trajectory or alter its apparent frequency.
        ax.add_collection(LineCollection([track for track in selected_tracks if len(track)],
                          colors="white", linewidths=TRAJECTORY_LINEWIDTH+2*trajectory_border_width,
                          alpha=1., capstyle="round", joinstyle="round", zorder=1.9))
    for index, track in zip(selected_indices,selected_tracks):
        if len(track):
            color = (success_color if goals[index] >= 0 else
                     failure_color) if outcome_colors else TRAJECTORY_COLOR
            ax.plot(track[:, 0], track[:, 1], color=color,
                    linewidth=TRAJECTORY_LINEWIDTH, alpha=alpha,
                    solid_capstyle="round", zorder=2)
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
