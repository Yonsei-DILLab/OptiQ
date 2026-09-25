"""Render the two official-source four-goal maze layouts without a simulator."""

from __future__ import annotations

import ast
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def assignment(path: Path, name: str):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def fourway_grid():
    # Explicitly reproduce environment.native_grid while keeping this layout
    # renderer usable on a laptop without Gymnasium or MuJoCo installed.
    grid = np.zeros((27, 27), dtype=np.int8)
    mid = 13
    grid[0, :] = grid[-1, :] = 1
    grid[:, 0] = grid[:, -1] = 1
    grid[1:mid - 1, 1:mid - 1] = 1
    grid[mid + 2:-1, 1:mid - 1] = 1
    grid[1:mid - 1, mid + 2:-1] = 1
    grid[mid + 2:-1, mid + 2:-1] = 1
    grid[mid - 1:mid + 2, 1] = 2
    grid[mid - 1:mid + 2, -2] = 3
    grid[1, mid - 1:mid + 2] = 4
    grid[-2, mid - 1:mid + 2] = 5
    grid[mid - 1:mid + 2, 2:mid - 1] = 6
    grid[mid, mid - 1] = 6
    grid[mid - 1:mid + 2, mid + 2:-2] = 7
    grid[mid, mid + 1] = 7
    grid[2:mid - 1, mid - 1:mid + 2] = 8
    grid[mid - 1, mid] = 8
    grid[mid + 2:-2, mid - 1:mid + 2] = 9
    grid[mid + 1, mid] = 9
    return grid


def pointmaze_grid():
    values = assignment(ROOT / "pointmaze/environment.py", "MAZE_MAP")
    return np.asarray([[1 if v == 1 else 2 if v == "g" else 3 if v == "r" else 0
                        for v in row] for row in values], dtype=np.int8)


def plot(out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    grids = (fourway_grid(), pointmaze_grid())
    titles = ("ICML 2019 4-Way geometry (27×27)", "Farama PointMaze dynamics + four-goal map (7×7)")
    for ax, grid, title in zip(axes, grids, titles):
        display = np.where(grid == 1, 1, 0)
        ax.imshow(display, cmap=ListedColormap(["#f8fafc", "#334155"]), interpolation="nearest")
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        if grid.shape[0] == 27:
            goals = [(12.5, 1, "W"), (12.5, 25, "E"), (1, 12.5, "N"), (25, 12.5, "S")]
            start = (13, 13)
        else:
            goals = [(3, 1, "W"), (3, 5, "E"), (1, 3, "N"), (5, 3, "S")]
            start = (3, 3)
        for row, col, label in goals:
            ax.scatter(col, row, marker="*", s=290, color="#f97316", edgecolor="black", linewidth=.6, zorder=3)
            ax.annotate(label, (col, row), xytext=(0, -20), textcoords="offset points", ha="center", fontsize=10)
        ax.scatter(start[1], start[0], marker="o", s=110, color="#14b8a6", edgecolor="black", linewidth=.6, zorder=3)
        ax.annotate("start", (start[1], start[0]), xytext=(5, 8), textcoords="offset points", fontsize=9)
    fig.suptitle("Four-goal environments: geometry preview (no trained trajectories)", fontsize=14)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    plot(parser.parse_args().out)
