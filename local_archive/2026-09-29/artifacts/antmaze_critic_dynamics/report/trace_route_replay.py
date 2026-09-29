"""Plot route use and replay occupancy from the saved v3/v4 control records."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RUNS = ROOT.parent / "results"
HISTORY = json.loads((ROOT / "results.json").read_text())["history"]
FIGURE, AXES = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
EXPORT = {}

for column, (task, host, run, minority, majority, replay_minor, replay_major) in enumerate((
    ("v3", "vast-heechan-180", "v3-optiq-control-500k-s0", "left", "right",
     "left_of_x_minus8", "right_of_x_plus8"),
    ("v4", "vast-heechan-199", "v4-optiq-control-500k-s0", "lower", "upper",
     "lower_after_x_minus4", "upper_after_x_minus4"),
)):
    run_root = RUNS / host / "runs" / run
    evals = sorted(
        (r for r in HISTORY if r["task"] == task and r["condition"] == "control"
         and r["episodes"] == 40),
        key=lambda r: r["total_step"],
    )
    replay = [json.loads(p.read_text()) for p in
              sorted((run_root / "replay-diagnostics").glob("*/summary.json"))]
    replay = [r for r in replay if r["step"] <= 500224]
    EXPORT[task] = {"evaluations": [
        {"step": r["total_step"], "minority": r[minority], "majority": r[majority],
         "uncommitted": r["uncommitted"]} for r in evals],
        "replay": [{"step": r["step"], "minority_region_count": r["spatial_counts"][replay_minor],
                    "majority_region_count": r["spatial_counts"][replay_major],
                    "sample_size": r["samples"],
                    "td_fit_rms": r["metrics"]["td_fit_residual"]["raw"]["rms"],
                    "online_target_rms": r["metrics"]["online_minus_target"]["raw"]["rms"]}
                   for r in replay]}
    ax = AXES[0, column]
    ax.plot([r["total_step"] / 1000 for r in evals], [r[minority] for r in evals],
            "o-", color="#d95f02", label=f"evaluation {minority} / 40")
    ax.plot([r["total_step"] / 1000 for r in evals], [r[majority] for r in evals],
            "o-", color="#1b9e77", label=f"evaluation {majority} / 40")
    ax.set_ylim(0, 43)
    ax.set_ylabel("Route entries / 40 rollouts")
    ax.set_title(f"{task}: fixed-start policy rollouts")
    ax.grid(alpha=.2)
    ax.legend(loc="upper left", fontsize=8)

    ax = AXES[1, column]
    xs = [r["step"] / 1000 for r in replay]
    ax.plot(xs, [r["spatial_counts"][replay_minor] for r in replay],
            "o-", color="#d95f02", label=f"replay {minority} region")
    ax.plot(xs, [r["spatial_counts"][replay_major] for r in replay],
            "o-", color="#1b9e77", label=f"replay {majority} region")
    ax.set_ylabel("Region observations / 1,024 replay samples")
    ax.set_xlabel("Environment transitions (thousands)")
    ax.set_title(f"{task}: replay occupancy (not full paths)")
    ax.grid(alpha=.2)
    ax.legend(loc="upper left", fontsize=8)

FIGURE.suptitle("Route concentration precedes replay erasure in the 500k control runs")
FIGURE.tight_layout()
FIGURE.savefig(ROOT / "route_vs_replay_trace.png", dpi=160)
(ROOT / "route_vs_replay_trace.json").write_text(json.dumps({
    "source": "saved critic-dynamics control evals and 1,024-transition replay samples",
    "training_source": "23603a7e7696aa64e8e49a38b986d6b430b6d6f3",
    "reward": "100*(d_current-d_next)",
    "scope": "one training seed and early 500k only; replay regions are not complete routes",
    "tasks": EXPORT,
}, indent=2) + "\n")
