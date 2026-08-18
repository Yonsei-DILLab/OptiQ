"""Compare four-way policy rollouts with and without density correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from environment import MultiGoalEnv


def reward_surface(points: np.ndarray, goals: np.ndarray) -> np.ndarray:
    differences = points[..., None, :] - goals
    return -np.square(differences).sum(axis=-1).min(axis=-1)


def load_result(run_dir: Path) -> tuple[list[np.ndarray], dict]:
    with np.load(run_dir / "trajectories.npz") as archive:
        trajectories = [archive[key] for key in sorted(archive.files)]
    summary = json.loads((run_dir / "summary.json").read_text())
    return trajectories, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uncorrected-run", type=Path, required=True)
    parser.add_argument("--corrected-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    results = (
        ("Without density correction", *load_result(args.uncorrected_run)),
        ("With density correction", *load_result(args.corrected_run)),
    )
    environment = MultiGoalEnv()
    state_axis = np.linspace(-7.5, 7.5, 301)
    state_x, state_y = np.meshgrid(state_axis, state_axis)
    rewards = reward_surface(
        np.stack([state_x, state_y], axis=-1), environment.goal_positions
    )

    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.35), constrained_layout=True)
    for panel, (title, trajectories, summary) in zip(axes, results, strict=True):
        panel.contour(
            state_x,
            state_y,
            rewards,
            levels=18,
            cmap="turbo",
            linewidths=0.8,
            alpha=0.68,
        )
        for trajectory in trajectories:
            panel.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                color="#477db3",
                alpha=0.2,
                linewidth=0.75,
            )
        panel.scatter(
            environment.goal_positions[:, 0],
            environment.goal_positions[:, 1],
            color="#c9342d",
            edgecolor="white",
            linewidth=0.6,
            s=45,
            zorder=5,
        )
        panel.scatter([0.0], [0.0], color="black", s=18, zorder=5)
        entropy = summary["trajectory_goal_entropy"]
        counts = summary["trajectory_goal_counts"]
        panel.set(
            title=f"{title}\nGoal counts {counts}, entropy {entropy:.3f}",
            xlabel="State x",
            ylabel="State y",
            xlim=(-7.5, 7.5),
            ylim=(-7.5, 7.5),
            aspect="equal",
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=250, bbox_inches="tight")
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
