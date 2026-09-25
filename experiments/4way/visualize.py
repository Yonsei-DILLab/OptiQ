"""Plot four-way paths, goal frequencies, action modes and teacher diagnostics."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
           title=f"10k policy rollouts ({len(trajectories)} episodes)")
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
    width = .36
    for mode_rows, offset, label in ((full, -width/2, "policy: z + conditional σ"),
                                     (mu, width/2, "μ-only: random z")):
        for d, (name, color) in enumerate(zip(NAMES, COLORS)):
            vals = np.asarray([row["direction_counts"][d] / row["episodes"] for row in mode_rows])
            ax.bar(x + offset, vals, width/4, color=color, alpha=.9 if offset < 0 else .48,
                   label=f"{name} ({label})" if d == 0 else None)
        other = np.asarray([row["uncommitted_first_actions"] / row["episodes"] for row in mode_rows])
        ax.plot(x + offset, other, "k--" if offset < 0 else "k:", marker="o", ms=3,
                label=f"uncommitted ({label})")
    ax.set_xticks(x, [str(row["step"]) for row in full])
    ax.set_ylim(0, 1)
    ax.set(xlabel="environment transitions", ylabel="fraction", title="First-action direction mass")
    ax.grid(axis="y", alpha=.2)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1, 1]
    training_path = args.run / "training.csv"
    with training_path.open() as handle:
        rows = list(csv.DictReader(handle))
    if rows:
        tr_steps = np.asarray([int(r["step"]) for r in rows])
        for key, label, style in (
            ("train/source_ess_absolute", "teacher source ESS", "-"),
            ("train/source_q_std", "candidate Q std", "--"),
            ("train/q_to_density_logit_std_ratio", "Q/density logit std", ":"),
        ):
            values = np.asarray([float(r[key]) if r.get(key) else np.nan for r in rows])
            if np.isfinite(values).any():
                ax.plot(tr_steps, values, style, label=label, alpha=.9)
        ax.set(xlabel="environment transitions", title="Learner responsibility diagnostics")
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    else:
        ax.text(.5, .5, "No learner diagnostics were recorded", ha="center", va="center")
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
    report += ["", "One seed and 10k transitions are a short diagnostic, not evidence of asymptotic multimodality.", ""]
    (args.out / "REPORT.md").write_text("\n".join(report))
    print(png)


if __name__ == "__main__":
    main()
