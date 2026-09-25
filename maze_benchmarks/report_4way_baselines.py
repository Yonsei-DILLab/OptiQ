"""Validate and render finished wall-free 4-Way baseline runs.

This is post-hoc reporting code; it does not replace any frozen training source.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .visualize_4way import GOAL_XY, _goal_proximity, render


ORDER = ("optiq", "sac", "sql", "meow", "mfpo", "dipo", "td3")
LABEL = dict(optiq="OptiQ T=1", sac="SAC", sql="SVGD SQL", meow="MEOW",
             mfpo="MFPO", dipo="DIPO", td3="TD3")
COLORS = ("#da492e", "#3487ba", "#269c55", "#9a58ac")


def validate(method: str, folder: Path):
    config = json.loads((folder / "config.json").read_text())
    progress = json.loads((folder / "progress.json").read_text())
    if config["task"] != "4way" or config["method"] != method or config["seed"] != 0:
        raise ValueError(f"{folder}: method, task or seed mismatch")
    if (config["steps"], config["num_envs"], config["batch_size"], config["utd"]) != (100000, 16, 256, 1.):
        raise ValueError(f"{folder}: data/update profile mismatch")
    if progress["status"] != "complete" or progress["steps"] != 100000 or progress["updates"] != 98976:
        raise ValueError(f"{folder}: 100k training is incomplete")
    record = progress["latest_evaluation"]
    if record["source_commit"] != config["source_commit"] or record["step"] != 100000:
        raise ValueError(f"{folder}: source or evaluation step mismatch")
    raw = folder / "evaluations/000100000_policy.npz"
    probe = folder / "evaluations/000100000_probe.npz"
    if not raw.is_file() or not probe.is_file():
        raise FileNotFoundError(f"{folder}: raw rollout or Q/action probe missing")
    with np.load(raw) as data:
        paths = data["xy"]
        goals = data["goal_ids"]
        returns = data["returns"]
        lengths = data["lengths"]
    if paths.shape != (100, 21, 2) or goals.shape != (100,) or not np.isfinite(returns).all():
        raise ValueError(f"{folder}: malformed raw policy rollouts")
    counts = np.bincount(goals[goals >= 0], minlength=4)
    summary = record["policy"]
    if summary["episodes"] != 100 or counts.tolist() != summary["goals"] or int((goals < 0).sum()) != summary["failure"]:
        raise ValueError(f"{folder}: reported goal counts differ from raw trajectories")
    with np.load(probe) as data:
        if not np.isfinite(data["actions"]).all() or not np.isfinite(data["q"]).all():
            raise ValueError(f"{folder}: nonfinite action or Q probe")
        critic_label = str(data["critic_label"])
    checkpoint_dir = folder / "checkpoints"
    if not list(checkpoint_dir.glob("policy_000100000.*")) or not (checkpoint_dir / "replay_000100000.npz").is_file():
        raise FileNotFoundError(f"{folder}: missing final policy/critic or replay")
    first = paths[:, 1] - paths[:, 0]
    dominant_x = np.abs(first[:, 0]) >= np.abs(first[:, 1])
    first_direction = np.where(dominant_x, np.where(first[:, 0] >= 0, 0, 1),
                               np.where(first[:, 1] >= 0, 2, 3))
    item = dict(method=method, label=LABEL[method], source_commit=config["source_commit"],
                steps=100000, updates=progress["updates"], seed=0,
                success=summary["success"], goal_counts=counts.tolist(),
                failures=summary["failure"], first_direction_counts=np.bincount(first_direction, minlength=4).tolist(),
                minimum_goal_count=int(counts.min()), mean_return=float(returns.mean()),
                mean_length=float(lengths.mean()), critic_label=critic_label)
    return item, raw, probe, paths, goals


def trajectories(rows, destination):
    _, _, proximity = _goal_proximity()
    axis = np.linspace(-7.7, 7.7, proximity.shape[0])
    xx, yy = np.meshgrid(axis, axis)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    for ax, (item, _, _, paths, goals) in zip(axes.ravel(), rows):
        ax.contour(xx, yy, proximity, levels=14, cmap="viridis", linewidths=.4, alpha=.6)
        for xy, goal in zip(paths, goals):
            xy = xy[np.isfinite(xy).all(axis=1)]
            ax.plot(xy[:, 0], xy[:, 1], color=COLORS[int(goal)] if goal >= 0 else "#666666",
                    lw=.8, alpha=.19)
        ax.scatter(GOAL_XY[:, 0], GOAL_XY[:, 1], color=COLORS, s=28, zorder=5)
        ax.scatter([0], [0], color="black", marker="+", s=48, zorder=6)
        ax.set(xlim=(-7.5, 7.5), ylim=(-7.5, 7.5), aspect="equal",
               title=f"{item['label']}: E/W/N/S {item['goal_counts']}")
        ax.grid(alpha=.15)
    for ax in axes.ravel()[len(rows):]:
        ax.axis("off")
    fig.suptitle("One sampled policy per method · 100 center-start rollouts · wall-free 4-Way")
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--run", nargs=2, action="append", metavar=("METHOD", "FOLDER"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    locations = {method: Path(folder) for method, folder in args.run}
    if set(locations) != set(ORDER) or len(args.run) != len(ORDER):
        raise ValueError("provide exactly one run for each of: " + ", ".join(ORDER))
    rows = [validate(method, locations[method]) for method in ORDER]
    args.output.mkdir(parents=True, exist_ok=True)
    summary = [row[0] for row in rows]
    (args.output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    for index, subset in enumerate((rows[:4], rows[4:]), start=1):
        render([(item["label"], raw, probe) for item, raw, probe, _, _ in subset],
               args.output / f"policy_and_q_{index}.png")
    trajectories(rows, args.output / "trajectories.png")
    print(args.output / "results.json")


if __name__ == "__main__":
    main()
