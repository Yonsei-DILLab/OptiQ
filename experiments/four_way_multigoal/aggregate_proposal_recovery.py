"""Aggregate the four-way proposal recovery experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .proposal_recovery import METHODS


METRICS = (
    "s0_action_covered_modes",
    "s0_action_mode_entropy_normalized",
    "s0_action_mode_tv_to_uniform",
    "s0_action_mode_max_abs_bias",
    "trajectory_success_rate",
    "trajectory_goal_covered_modes",
    "trajectory_goal_mode_entropy_normalized",
    "trajectory_goal_mode_tv_to_uniform",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/four_way_multigoal/proposal_recovery"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=(1, 2, 3))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = {}
    missing = []
    for method in METHODS:
        records[method] = []
        for seed in args.seeds:
            path = args.input_dir / method / f"seed_{seed}" / "summary.json"
            if not path.exists():
                missing.append(str(path))
                continue
            records[method].append(json.loads(path.read_text()))
    if missing:
        raise FileNotFoundError("missing result files:\n" + "\n".join(missing))

    aggregate = {
        "seeds": args.seeds,
        "formal_unbiasedness_note": (
            "include_anchor=True adds deterministic samples that are not draws from "
            "the continuous proposal, so exact finite-sample unbiasedness is not "
            "guaranteed. Reported TV/bias metrics measure empirical recovery."
        ),
        "methods": {},
    }
    rows = []
    for method, method_records in records.items():
        summary = {}
        for metric in METRICS:
            values = np.asarray([record[metric] for record in method_records])
            summary[metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "values": values.tolist(),
            }
        probabilities = np.asarray(
            [record["s0_action_mode_probabilities"] for record in method_records]
        )
        summary["s0_action_mode_probabilities"] = {
            "mean": probabilities.mean(axis=0).tolist(),
            "std": probabilities.std(axis=0, ddof=1).tolist(),
        }
        summary["all_seeds_cover_four_s0_modes"] = bool(
            all(record["s0_action_covered_modes"] == 4 for record in method_records)
        )
        summary["all_seeds_pass_s0_uniform_chi_square_5pct"] = bool(
            all(
                record["s0_action_uniform_chi_square_df3"] <= 7.8147
                for record in method_records
            )
        )
        summary["mean_wall_time_seconds"] = float(
            np.mean([record["wall_time_seconds"] for record in method_records])
        )
        summary["mean_proposal_out_of_bounds_fraction"] = float(
            np.mean(
                [
                    record["final_training_metrics"][
                        "proposal_out_of_bounds_fraction"
                    ]
                    for record in method_records
                ]
            )
        )
        summary["mean_proposal_unsupported_fraction"] = float(
            np.mean(
                [
                    record["final_training_metrics"][
                        "proposal_unsupported_fraction"
                    ]
                    for record in method_records
                ]
            )
        )
        aggregate["methods"][method] = summary
        rows.append(
            {
                "method": method,
                **{
                    f"{metric}_{statistic}": summary[metric][statistic]
                    for metric in METRICS
                    for statistic in ("mean", "std")
                },
                "all_seeds_cover_four_s0_modes": summary[
                    "all_seeds_cover_four_s0_modes"
                ],
                "all_seeds_pass_s0_uniform_chi_square_5pct": summary[
                    "all_seeds_pass_s0_uniform_chi_square_5pct"
                ],
                "mean_wall_time_seconds": summary["mean_wall_time_seconds"],
                "mean_proposal_out_of_bounds_fraction": summary[
                    "mean_proposal_out_of_bounds_fraction"
                ],
                "mean_proposal_unsupported_fraction": summary[
                    "mean_proposal_unsupported_fraction"
                ],
            }
        )

    json_path = args.input_dir / "aggregate.json"
    csv_path = args.input_dir / "aggregate.csv"
    figure_path = args.input_dir / "aggregate.png"
    json_path.write_text(json.dumps(aggregate, indent=2) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=rows[0].keys(), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    labels = [method.replace("_", "\n") for method in METHODS]
    x = np.arange(len(labels))
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    for axis, metric, title, limits in (
        (axes[0, 0], "s0_action_covered_modes", "S0 covered modes", (0, 4.2)),
        (
            axes[0, 1],
            "s0_action_mode_tv_to_uniform",
            "S0 total variation to uniform",
            (0, None),
        ),
        (
            axes[1, 0],
            "trajectory_success_rate",
            "Trajectory success rate",
            (0, 1.0),
        ),
        (
            axes[1, 1],
            "trajectory_goal_mode_tv_to_uniform",
            "Successful-goal TV to uniform",
            (0, None),
        ),
    ):
        means = [aggregate["methods"][method][metric]["mean"] for method in METHODS]
        errors = [aggregate["methods"][method][metric]["std"] for method in METHODS]
        axis.bar(x, means, yerr=errors, capsize=3, color="#4477AA")
        axis.set_title(title)
        axis.set_xticks(x, labels, fontsize=7)
        axis.set_ylim(*limits)
        axis.grid(axis="y", alpha=0.2)
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    print(json.dumps(aggregate, indent=2))
    print(f"saved {json_path}")
    print(f"saved {csv_path}")
    print(f"saved {figure_path}")


if __name__ == "__main__":
    main()
