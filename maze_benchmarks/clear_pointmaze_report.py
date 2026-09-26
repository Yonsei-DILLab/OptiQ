"""Readable post-hoc PointMaze trajectories from every preserved evaluation.

The goal-balanced atlas is illustrative, not a frequency plot. The companion
visitation map and goal counts use every episode of each saved policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
import numpy as np

from .report_pointmaze import MAZES, METHODS
from .visualize_pointmaze import get_map, plot_map


PALETTE = plt.get_cmap("tab10")


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def load_run(root: Path, maze: str, method: str) -> dict:
    folder = root / "runs" / f"pm_{maze}-{method}-s0"
    config = json.loads((folder / "config.json").read_text())
    summaries = sorted((folder / "evaluations").glob("*_summary.json"))
    if len(summaries) == 0:
        raise FileNotFoundError(folder)
    summary = json.loads(summaries[-1].read_text())
    source = folder / "evaluations" / f"{summary['step']:09d}_policy.npz"
    with np.load(source) as values:
        tracks = values["xy"].copy()
        lengths = values["lengths"].copy()
        goals = values["goal_ids"].copy()
    count = 8 if maze == "hard" else 4
    actual = np.bincount(goals[goals >= 0], minlength=count)
    episodes = int(summary["policy"]["episodes"])
    if (len(tracks) != episodes or len(lengths) != episodes or
            len(goals) != episodes or actual.tolist() != summary["policy"]["goals"] or
            not np.isclose(np.mean(goals >= 0), summary["policy"]["success"]) or
            not np.all((lengths > 0) & (lengths < tracks.shape[1])) or
            not np.isfinite(tracks[:, 0, :]).all()):
        raise ValueError(f"saved evaluation disagrees with summary: {source}")
    positions = np.asarray(config["geometry"]["goal_positions"])
    if positions.shape != (count, 2):
        raise ValueError(f"goal geometry mismatch: {source}")
    return dict(xy=tracks, lengths=lengths, goals=goals, counts=actual,
                episodes=episodes,
                positions=positions, summary=summary, source=source)


def tracks_for_goal(run: dict, goal: int, limit: int) -> np.ndarray:
    candidates = np.flatnonzero(run["goals"] == goal)
    if len(candidates) <= limit:
        return candidates
    # Evenly spaced evaluation indices provide a deterministic, neutral sample.
    return candidates[np.rint(np.linspace(0, len(candidates) - 1, limit)).astype(int)]


def draw_goals(ax, positions: np.ndarray):
    for goal, (x, y) in enumerate(positions):
        color = PALETTE(goal)
        ax.add_patch(Circle((x, y), .42, fill=False, edgecolor=color,
                            linewidth=1.8, zorder=6))
        ax.text(x, y + .68, str(goal + 1), fontsize=7, fontweight="bold",
                color=color, ha="center", va="bottom", zorder=7,
                bbox=dict(facecolor="white", alpha=.8, edgecolor="none", pad=.2))


def selected_tracks(ax, run: dict, *, per_goal: int, show_failures: bool):
    if show_failures and not np.any(run["goals"] >= 0):
        failures = np.flatnonzero(run["goals"] < 0)
        selected = failures[np.rint(np.linspace(0, len(failures) - 1,
                                               min(12, len(failures)))).astype(int)]
        for episode in selected:
            track = run["xy"][episode, :run["lengths"][episode] + 1]
            ax.plot(track[:, 0], track[:, 1], color="#777777", linewidth=.8,
                    alpha=.5, zorder=3)
        return
    for goal in range(len(run["counts"])):
        for episode in tracks_for_goal(run, goal, per_goal):
            track = run["xy"][episode, :run["lengths"][episode] + 1]
            ax.plot(track[:, 0], track[:, 1], color=PALETTE(goal),
                    linewidth=.85, alpha=.58, zorder=3)


def format_map(ax, maze: str, run: dict):
    plot_map(ax, maze)
    draw_goals(ax, run["positions"])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")


def atlas(root: Path, mazes: tuple[str, ...], methods: tuple[str, ...], output: Path):
    fig, axes = plt.subplots(len(mazes), len(methods),
                             figsize=(3.8 * len(methods), 4.25 * len(mazes)),
                             squeeze=False, constrained_layout=True)
    for row, maze in enumerate(mazes):
        for column, method in enumerate(methods):
            ax = axes[row, column]
            run = load_run(root, maze, method)
            format_map(ax, maze, run)
            selected_tracks(ax, run, per_goal=4, show_failures=True)
            reached = int(np.count_nonzero(run["counts"]))
            success = run["summary"]["policy"]["success"]
            ax.set_title(f"{method.upper()} · {success:.0%} success\n"
                         f"{reached}/{len(run['counts'])} goals", fontsize=9)
        axes[row, 0].text(-.12, .5, maze.title(), transform=axes[row, 0].transAxes,
                          rotation=90, ha="center", va="center", fontweight="bold")
    fig.suptitle(f"Goal-stratified paths from {run['episodes']:,} saved policy evaluations · seed 0\n"
                 "Up to 4 evenly spaced successful episodes per goal; grey paths show failures when no goal was reached",
                 fontsize=12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def visit_fraction(run: dict, maze: str, bins: int = 112) -> np.ndarray:
    cells = get_map(maze)
    height, width = len(cells), len(cells[0])
    counts = np.zeros((bins, bins), dtype=np.int32)
    for path, length in zip(run["xy"], run["lengths"]):
        xy = path[:length + 1]
        x = np.clip(((xy[:, 0] + width / 2) / width * bins).astype(int), 0, bins - 1)
        y = np.clip(((xy[:, 1] + height / 2) / height * bins).astype(int), 0, bins - 1)
        flat = np.unique(y * bins + x)
        counts.ravel()[flat] += 1
    return counts / run["episodes"]


def density(root: Path, mazes: tuple[str, ...], methods: tuple[str, ...], output: Path):
    fig, axes = plt.subplots(len(mazes), len(methods),
                             figsize=(3.8 * len(methods), 4.25 * len(mazes)),
                             squeeze=False, constrained_layout=True)
    image = None
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(alpha=0)
    for row, maze in enumerate(mazes):
        cells = get_map(maze)
        height, width = len(cells), len(cells[0])
        for column, method in enumerate(methods):
            ax = axes[row, column]
            run = load_run(root, maze, method)
            values = visit_fraction(run, maze)
            image = ax.imshow(np.ma.masked_equal(values, 0), origin="lower",
                              extent=(-width / 2, width / 2, -height / 2, height / 2),
                              cmap=cmap, norm=LogNorm(vmin=1 / run["episodes"], vmax=1),
                              interpolation="nearest", zorder=-1)
            format_map(ax, maze, run)
            ax.set_title(f"{method.upper()} · all {run['episodes']:,} episodes", fontsize=9)
        axes[row, 0].text(-.12, .5, maze.title(), transform=axes[row, 0].transAxes,
                          rotation=90, ha="center", va="center", fontweight="bold")
    fig.colorbar(image, ax=axes.ravel().tolist(), location="bottom", fraction=.025,
                 pad=.04, label="Fraction of episodes visiting each location (log scale)")
    fig.suptitle("All-episode visitation density · one policy per method · seed 0",
                 fontsize=12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def goal_facets(root: Path, maze: str, method: str, output: Path):
    run = load_run(root, maze, method)
    count = len(run["counts"])
    columns = 4
    rows = (count + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4.2 * rows),
                             squeeze=False, constrained_layout=True)
    for goal, ax in enumerate(axes.ravel()):
        if goal >= count:
            ax.axis("off")
            continue
        format_map(ax, maze, run)
        for episode in tracks_for_goal(run, goal, 12):
            track = run["xy"][episode, :run["lengths"][episode] + 1]
            ax.plot(track[:, 0], track[:, 1], color=PALETTE(goal),
                    linewidth=.9, alpha=.55, zorder=3)
        ax.set_title(f"Goal {goal + 1}: {run['counts'][goal]}/{run['episodes']} episodes",
                     fontsize=10)
    fig.suptitle(f"{maze.title()} {method.upper()} · routes to each goal · seed 0\n"
                 "Up to 12 evenly spaced successful episodes per goal from the same saved policy",
                 fontsize=12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-source", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "archive-manifest.json").read_text())
    if (manifest["completed"], manifest["expected"]) != (21, 21):
        raise ValueError("all 21 runs must be verified before the clear report")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = {}
    episode_counts = set()
    for maze in MAZES:
        for method in METHODS:
            run = load_run(root, maze, method)
            sources[f"{maze}:{method}"] = digest(run["source"])
            episode_counts.add(run["episodes"])
    if len(episode_counts) != 1:
        raise ValueError(f"inconsistent final episode counts: {episode_counts}")
    first_group = tuple(METHODS[:4])
    second_group = tuple(METHODS[4:])
    figures = {
        "simple_goal_paths.png": lambda p: atlas(root, ("simple",), tuple(METHODS), p),
        "medium_hard_goal_paths.png": lambda p: atlas(root, ("medium", "hard"), tuple(METHODS), p),
        "medium_hard_goal_paths_a.png": lambda p: atlas(root, ("medium", "hard"), first_group, p),
        "medium_hard_goal_paths_b.png": lambda p: atlas(root, ("medium", "hard"), second_group, p),
        "simple_visitation.png": lambda p: density(root, ("simple",), tuple(METHODS), p),
        "medium_hard_visitation.png": lambda p: density(root, ("medium", "hard"), tuple(METHODS), p),
        "medium_hard_visitation_a.png": lambda p: density(root, ("medium", "hard"), first_group, p),
        "medium_hard_visitation_b.png": lambda p: density(root, ("medium", "hard"), second_group, p),
        "medium_optiq_goal_facets.png": lambda p: goal_facets(root, "medium", "optiq", p),
        "hard_optiq_goal_facets.png": lambda p: goal_facets(root, "hard", "optiq", p),
        "hard_sql_goal_facets.png": lambda p: goal_facets(root, "hard", "sql", p),
    }
    hashes = {}
    for name, render in figures.items():
        destination = output / name
        render(destination)
        hashes[name] = digest(destination)
    record = dict(training_source_commit=manifest["training_source_commit"],
                  archive_source_commit=manifest["reporting_source_commit"],
                  report_source_commit=args.report_source,
                  policy="directly sampled final policy including conditional sigma for OptiQ",
                  seed=0, episodes_per_policy=episode_counts.pop(), raw_evaluation_sha256=sources,
                  figure_sha256=hashes,
                  atlas="up to 4 evenly spaced success episodes per goal; illustrative, not frequency",
                  density="all final episodes; fraction of episodes visiting each grid cell")
    (output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(dict(verified_runs=len(sources), figures=list(figures), output=str(output))))


if __name__ == "__main__":
    main()
