"""Paper-style PointMaze comparison from preserved raw rollout summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np

from .visualize_pointmaze import plot_map, plot_rollouts


METHODS = ("optiq", "sac", "sql", "meow", "mfpo", "dipo", "td3")
MAZES = ("simple", "medium", "hard")
PALETTE = dict(zip(METHODS, ("#1f77b4", "#ff7f0e", "#9467bd", "#2ca02c",
                             "#d62728", "#8c564b", "#7f7f7f")))
METRICS = (
    ("Success rate", lambda row: row["policy"]["success"], (0., 1.)),
    ("Reachable goals", lambda row: row["policy"]["reachable_goals"], None),
    ("Removal SR5", lambda row: row["policy"]["sr5_removal"], (0., 1.)),
    ("Obstacle SR5", lambda row: row["obstacle_policy"]["sr5_obstacle"], (0., 1.)),
)


def records(root: Path, maze: str, method: str):
    folder = root / "runs" / f"pm_{maze}-{method}-s0"
    rows = []
    for path in sorted((folder / "evaluations").glob("*_summary.json")):
        item = json.loads(path.read_text())
        if item.get("task") == f"pm_{maze}" and item.get("method") == method:
            rows.append(item)
    return rows


def render_curves(root: Path, destination: Path):
    fig, axes = plt.subplots(4, 3, figsize=(15, 12), constrained_layout=True)
    for column, maze in enumerate(MAZES):
        for row, (label, getter, limits) in enumerate(METRICS):
            ax = axes[row, column]
            for method in METHODS:
                results = records(root, maze, method)
                if not results:
                    continue
                ax.plot([item["step"] for item in results], [getter(item) for item in results],
                        marker="o", markersize=2.5, label=method.upper(),
                        color=PALETTE[method], linewidth=1.5)
            ax.set(title=maze.title() if row == 0 else None,
                   xlabel="Environment transitions" if row == 3 else None,
                   ylabel=label if column == 0 else None)
            ax.set_xlim(0, {"simple": 100_000, "medium": 200_000,
                            "hard": 300_000}[maze])
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.xaxis.set_major_formatter(
                FuncFormatter(lambda value, _position: f"{value / 1000:.0f}k"))
            ax.tick_params(axis="x", labelbottom=row == 3)
            if limits is not None:
                ax.set_ylim(*limits)
            if row == 1:
                ax.set_ylim(0, 8 if maze == "hard" else 4)
            ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=7,
                   bbox_to_anchor=(.5, 1.03), frameon=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)


def render_trajectories(root: Path, maze: str, destination: Path):
    fig, axes = plt.subplots(1, len(METHODS), figsize=(3.0 * len(METHODS), 3.8),
                             constrained_layout=True)
    for ax, method in zip(axes, METHODS):
        plot_map(ax, maze)
        folder = root / "runs" / f"pm_{maze}-{method}-s0" / "evaluations"
        summaries = sorted(folder.glob("*_summary.json"))
        if summaries:
            item = json.loads(summaries[-1].read_text())
            path = folder / f"{item['step']:09d}_policy.npz"
            if path.is_file():
                result = plot_rollouts(ax, path)
                ax.set_title(f"{method.upper()} · {item['step']//1000}k\n"
                             f"success {result['success']:.0%}")
            else:
                ax.set_title(f"{method.upper()} · raw data missing")
        else:
            ax.set_title(f"{method.upper()} · pending")
    fig.suptitle(f"{maze.title()} PointMaze · directly sampled policy trajectories · seed 0")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def render_medium_hard(root: Path, destination: Path):
    """Two-row, seven-method trajectory plate like paper Figures 10 and 11."""
    fig, axes = plt.subplots(2, len(METHODS), figsize=(21, 7.2),
                             constrained_layout=True)
    for row, maze in enumerate(("medium", "hard")):
        for column, method in enumerate(METHODS):
            ax = axes[row, column]
            plot_map(ax, maze)
            folder = root / "runs" / f"pm_{maze}-{method}-s0" / "evaluations"
            summaries = sorted(folder.glob("*_summary.json"))
            if summaries:
                item = json.loads(summaries[-1].read_text())
                path = folder / f"{item['step']:09d}_policy.npz"
                if path.is_file():
                    plot_rollouts(ax, path)
                    detail = (f"{item['policy']['success']:.0%} success · "
                              f"{item['policy']['reachable_goals']}/{8 if maze == 'hard' else 4} goals")
                else:
                    detail = "raw trajectories missing"
            else:
                detail = "pending"
            ax.set(title=f"{method.upper()}\n{detail}", xlabel="", ylabel="")
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row, 0].text(-.13, .5, maze.title(), transform=axes[row, 0].transAxes,
                          rotation=90, ha="center", va="center", fontweight="bold")
    fig.suptitle("Medium and Hard PointMaze · first 100 sampled-policy trajectories per panel · seed 0\n"
                 "Success and reachable-goal counts use all episodes at the latest checkpoint")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def render_optiq_modes(root: Path, maze: str, destination: Path):
    """Compare sampled policy and random-z means from one OptiQ checkpoint."""
    folder = root / "runs" / f"pm_{maze}-optiq-s0" / "evaluations"
    summaries = sorted(folder.glob("*_summary.json"))
    if not summaries:
        return False
    item = json.loads(summaries[-1].read_text())
    step = item["step"]
    sources = (("policy", "Direct policy, including conditional σ"),
               ("mu_only", "Random-z μ-only, no conditional σ"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.4), constrained_layout=True)
    for ax, (mode, label) in zip(axes, sources):
        source = folder / f"{step:09d}_{mode}.npz"
        if not source.is_file():
            raise FileNotFoundError(source)
        plot_map(ax, maze)
        actual = plot_rollouts(ax, source)
        summary = item[mode]
        counts = actual["goals"] + [0] * (len(summary["goals"]) - len(actual["goals"]))
        if (actual["episodes"] != summary["episodes"] or
                not np.isclose(actual["success"], summary["success"]) or
                counts != summary["goals"]):
            raise ValueError(f"OptiQ evaluation summary mismatch: {source}")
        ax.set_title(f"{label}\nSuccess {summary['success']:.1%} · "
                     f"goals {summary['goals']}")
    fig.suptitle(f"OptiQ {maze.title()} · seed 0 · {step:,} transitions · "
                 "first 100 trajectories per panel")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=170)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    render_curves(args.root, args.output / "learning_curves.png")
    for maze in MAZES:
        render_trajectories(args.root, maze,
                            args.output / f"trajectories_{maze}.png")
        render_optiq_modes(args.root, maze,
                           args.output / f"optiq_{maze}_policy_vs_mu_only.png")
    render_medium_hard(args.root, args.output / "trajectories_medium_hard.png")
    print(json.dumps({"output": str(args.output), "seed": 0,
                      "methods": METHODS, "mazes": MAZES}))


if __name__ == "__main__":
    main()
