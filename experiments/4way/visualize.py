"""Plot four-way paths, goal frequencies, action modes and teacher diagnostics."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

NAMES = ("east", "west", "north", "south")
COLORS = ("#d55e00", "#0072b2", "#009e73", "#cc79a7")
GOALS = np.asarray([[5., 0.], [-5., 0.], [0., 5.], [0., -5.]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records = json.loads((args.run / "evaluation_summary.json").read_text())
    steps = sorted({int(row["step"]) for row in records})
    full = [row for row in records if row["mode"] == "policy"]
    mu = [row for row in records if row["mode"] == "mu_only"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    final_step = max(steps)
    path_data = np.load(args.run / "evaluations" / f"eval_{final_step:05d}_policy.npz")
    ax = axes[0, 0]
    trajectories, goals, lengths = path_data["trajectories"], path_data["goals"], path_data["lengths"]
    for i in range(len(trajectories)):
        route = int(goals[i])
        color = COLORS[route] if route >= 0 else "#999999"
        ax.plot(trajectories[i, :lengths[i]+1, 0], trajectories[i, :lengths[i]+1, 1],
                color=color, alpha=.34 if route >= 0 else .13, lw=.8)
    ax.scatter([0], [0], marker="*", s=140, color="black", zorder=5, label="fixed start")
    for i, (goal, name) in enumerate(zip(GOALS, NAMES)):
        ax.scatter(*goal, marker="X", s=110, color=COLORS[i], label=f"{name} goal")
        ax.add_patch(plt.Circle(goal, 1.0, fill=False, color=COLORS[i], alpha=.6))
    ax.set(xlim=(-7, 7), ylim=(-7, 7), xlabel="x", ylabel="y",
           title=f"{final_step:,} policy rollouts ({len(trajectories)} episodes)")
    ax.set_aspect("equal")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8, ncol=2)

    ax = axes[0, 1]
    x = np.arange(len(full))
    bottom = np.zeros(len(full))
    for goal_id, (name, color) in enumerate(zip(NAMES, COLORS)):
        vals = np.asarray([row["goal_counts"][goal_id] for row in full])
        ax.bar(x, vals, bottom=bottom, color=color, label=name)
        bottom += vals
    failures = np.asarray([row["episodes"] - sum(row["goal_counts"]) for row in full])
    ax.bar(x, failures, bottom=bottom, color="#bbbbbb", label="no goal")
    ax.set_xticks(x, [str(row["step"]) for row in full])
    ax.set(xlabel="environment transitions", ylabel="episodes", title="Full-policy goal counts")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=.2)

    ax = axes[1, 0]
    slot = 0
    tick_positions, tick_labels = [], []
    for mode_rows, mode_label in ((full, "policy"), (mu, "μ-only")):
        for row in mode_rows:
            counts = list(row["direction_counts"])
            counts.append(row["uncommitted_first_actions"])
            fractions = np.asarray(counts, dtype=float) / row["episodes"]
            bottom = 0.0
            for d, color in enumerate((*COLORS, "#999999")):
                ax.bar(slot, fractions[d], bottom=bottom, width=.72, color=color,
                       alpha=.92 if mode_label == "policy" else .48)
                bottom += fractions[d]
            tick_positions.append(slot)
            tick_labels.append(f"{row['step']}\n{mode_label}")
            slot += 1
        slot += .45
    ax.set_xticks(tick_positions, tick_labels)
    ax.set_ylim(0, 1)
    ax.set(xlabel="environment transitions", ylabel="fraction", title="First-action direction mass")
    ax.grid(axis="y", alpha=.2)
    ax.legend(handles=[Patch(facecolor=color, label=name)
                       for name, color in zip((*NAMES, "uncommitted"), (*COLORS, "#999999"))],
              fontsize=7, ncol=2)

    ax = axes[1, 1]
    q_steps = []
    q_values = []
    for step in steps:
        q_path = args.run / "evaluations" / f"origin_action_cloud_{step:05d}.npz"
        if q_path.exists():
            q_steps.append(step)
            q_values.append(np.load(q_path)["directional_q"])
    if q_steps:
        q_values = np.asarray(q_values)  # checkpoint × twin critic × direction
        for d, (name, color) in enumerate(zip(NAMES, COLORS)):
            q_low = q_values[:, :, d].min(axis=1)
            q_high = q_values[:, :, d].max(axis=1)
            ax.plot(q_steps, (q_low + q_high) / 2, color=color, marker="o", label=name)
            ax.fill_between(q_steps, q_low, q_high, color=color, alpha=.16)
        ax.set(xlabel="environment transitions", ylabel="Q at origin for unit-axis action",
               title="Twin-critic directional Q probes")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=.2)
    else:
        ax.text(.5, .5, "No directional Q probes were recorded", ha="center", va="center")
        ax.set_axis_off()

    fig.suptitle("Direct GMM OptiQ / iBOLT — symmetric four-goal diagnostic", fontsize=15)
    png = args.out / "fourway_diagnostic.png"
    fig.savefig(png, dpi=180)
    plt.close(fig)
    report = ["# 4way OptiQ diagnostic", "",
              "Four goals are symmetric and every evaluation episode starts at the same origin.",
              "Goal reach frequencies are separate from first-action directional mass.", "",
              "| step | mode | success | east | west | north | south | no goal | mean return |",
              "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in records:
        gc = row["goal_counts"]
        report.append(f"| {row['step']} | {row['mode']} | {row['success_rate']:.3f} | "
                      f"{gc[0]} | {gc[1]} | {gc[2]} | {gc[3]} | "
                      f"{row['episodes']-sum(gc)} | {row['mean_return']:.3f} |")
    report += ["", f"One seed and {final_step:,} transitions are a short diagnostic, not evidence of asymptotic multimodality.", ""]
    (args.out / "REPORT.md").write_text("\n".join(report))
    print(png)


if __name__ == "__main__":
    main()
