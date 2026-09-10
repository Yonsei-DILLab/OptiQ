"""Read-only paired-seed reporting and early-stop evidence for confirmation.

The manifest fixes run identities. Comparisons intersect evaluation steps rather
than comparing the latest (asynchronous) records. Stopped/slow seeds are never
dropped from the four-seed aggregate. This script does not control processes.
"""
from pathlib import Path
import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def aligned_pair(v2_steps, v2_returns, reference_steps, reference_returns, tail=3):
    common, vi, ri = np.intersect1d(v2_steps, reference_steps, return_indices=True)
    if not len(common):
        raise ValueError("No common evaluation step")
    v2, reference = np.asarray(v2_returns)[vi], np.asarray(reference_returns)[ri]
    vmean, rmean = float(v2[-tail:].mean()), float(reference[-tail:].mean())
    return {"steps": common, "v2": v2, "optiq": reference,
            "step": int(common[-1]), "v2_tail_mean": vmean, "optiq_tail_mean": rmean,
            "ratio": vmean/rmean if rmean > 0 else None}


def recent_training_rate(rows, window=20000):
    samples = [(float(row["time/total_timesteps"]), float(row["time/time_elapsed"]))
               for row in rows if row.get("time/total_timesteps") and row.get("time/time_elapsed")
               and float(row["time/total_timesteps"]) >= 10000]
    if len(samples) < 2:
        return None
    n, t = samples[-1]
    n0, t0 = next(((x, y) for x, y in samples if x >= n-window), samples[0])
    return (n-n0)/(t-t0) if t-t0 >= 60 else None


def read_run(item, expected_episodes):
    directory = Path(item["directory"])
    cfg = json.loads((directory / "config.json").read_text())
    assert cfg["env_name"] == "Humanoid-v4"
    assert cfg["seed"] == item["seed"]
    assert cfg["runtime"]["git_commit"] == item["commit"]
    assert cfg["num_eval_episodes"] == expected_episodes
    assert cfg["alg"].get("behavior_uniform_probability", 0.) == 0.
    with np.load(next(directory.glob("eval/*/evaluations.npz"))) as data:
        steps, results = data["timesteps"].copy(), data["results"].copy()
    assert results.shape == (len(steps), expected_episodes)
    if not np.isfinite(results).all():
        raise ValueError(f"Nonfinite evaluation results in {directory}")
    with (directory / "logs/progress.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    def last(key):
        return next((float(row[key]) for row in reversed(rows) if row.get(key)), None)
    return {"item": item, "steps": steps, "returns": results.mean(axis=1),
            "logged_steps": last("time/total_timesteps"),
            "completed_marker": (directory / "completed.json").exists(),
            "training_rate": recent_training_rate(rows),
            "acceptance": last("train/soft_guard_acceptance_cumulative"),
            "ess": last("train/source_ess_absolute"),
            "entropy": last("train/backup_entropy_lower"),
            "actor_std": last("train/actor_std_mean")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/v2_improvement/confirmation_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/v2_improvement/confirmation_report")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    assert len(manifest["runs"]) == 8
    items = {(r["seed"], r["method"]): r for r in manifest["runs"]}
    assert set(items) == {(s, m) for s in range(4) for m in ("v2", "optiq")}
    # Supervisor is authoritative for liveness, independently of file markers.
    status_result = subprocess.run(["supervisorctl", "status", *[r["supervisor"] for r in items.values()]],
                                   text=True, capture_output=True)
    statuses = {line.split()[0]: line.split()[1] for line in status_result.stdout.splitlines() if len(line.split()) >= 2}
    runs = {key: read_run(item, manifest["eval_episodes"]) for key, item in items.items()}
    pairs = {}
    for seed in range(4):
        a, b = runs[seed, "v2"], runs[seed, "optiq"]
        pairs[seed] = aligned_pair(a["steps"], a["returns"], b["steps"], b["returns"])
    common = pairs[0]["steps"]
    for pair in pairs.values():
        common = np.intersect1d(common, pair["steps"])
    curves = {method: np.stack([pair[method][np.searchsorted(pair["steps"], common)]
                               for pair in pairs.values()]) for method in ("v2", "optiq")}
    scores = {method: values[:, -3:].mean(axis=1) for method, values in curves.items()}
    gap = scores["v2"] - scores["optiq"]
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(),
              "common_step_all_four_seeds": int(common[-1]),
              "aggregate": {method: {"mean": float(values.mean()), "seed_sd": float(values.std(ddof=1)),
                                      "seed_scores": values.tolist()} for method, values in scores.items()},
              "paired_mean_gap": float(gap.mean()), "paired_gap_seed_sd": float(gap.std(ddof=1)),
              "positive_seed_gaps": int((gap > 0).sum()), "pairs": [], "runs": []}
    for seed, pair in pairs.items():
        # A resource rule, not a significance test or proof of inferiority.
        recent = min(7, len(pair["steps"]))
        slope = (float(np.polyfit(pair["steps"][-recent:]/1000, pair["v2"][-recent:], 1)[0])
                 if recent >= 2 else None)
        eligible = (pair["step"] >= 100000 and pair["ratio"] is not None
                    and pair["ratio"] < .5 and slope is not None and slope <= 0)
        report["pairs"].append({"seed": seed, "matched_step": pair["step"],
            "v2_tail_mean": pair["v2_tail_mean"], "optiq_tail_mean": pair["optiq_tail_mean"],
            "ratio": pair["ratio"], "v2_trend_per_1000_steps": slope,
            "eligible_for_early_stop_review": bool(eligible)})
    for run in runs.values():
        item = run["item"]
        row = {**item, **{k: v for k, v in run.items() if k not in {"item", "steps", "returns"}}}
        row["process_state"] = statuses.get(item["supervisor"], "UNKNOWN")
        report["runs"].append(row)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "latest.json").write_text(json.dumps(report, indent=2))
    (output / f"step_{int(common[-1]):07d}.json").write_text(json.dumps(report, indent=2))
    lines = ["# Humanoid v2 paired confirmation", "",
        f"Common evaluation step across all eight runs: {int(common[-1]):,}.",
        "Scores average the last three matching evaluation checkpoints; ten stochastic episodes per checkpoint.", "",
        "| Training seed | v2 | OptiQ | v2 / OptiQ |",
        "|---|---:|---:|---:|"]
    for seed in range(4):
        a, b = scores["v2"][seed], scores["optiq"][seed]
        lines.append(f"| {seed} | {a:.1f} | {b:.1f} | {a/b:.3f} |")
    lines += ["", f"Paired seed-mean gap: {gap.mean():+.1f}; SD across four seed gaps: {gap.std(ddof=1):.1f}.",
        "Four seeds are a small sample. Episode counts do not increase the number of independent training seeds.",
        "Stopped and slower seeds remain in the aggregate; it cannot silently become a survivors-only result.", "",
        "Early-stop flags are review evidence only. This read-only script does not stop or restart any process.",
        "The sample guard and these return plots do not certify actual statewise policy improvement."]
    (output / "latest.md").write_text("\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {"v2": "#176ab0", "optiq": "#303030"}
    for method, values in curves.items():
        mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
        axes[0].plot(common/1000, mean, color=colors[method], label=method)
        axes[0].fill_between(common/1000, mean-sd, mean+sd, color=colors[method], alpha=.15)
    for seed, pair in pairs.items():
        axes[1].plot(pair["steps"]/1000, pair["v2"] / np.maximum(pair["optiq"], 1e-8), label=f"seed {seed}")
    axes[1].axhline(1., color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Evaluation return"); axes[1].set_ylabel("v2 / OptiQ at matching steps")
    axes[0].set_title("Four-seed mean and SD"); axes[1].set_title("Individual paired seeds")
    for ax in axes:
        ax.set_xlabel("Environment steps (thousands)"); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle("Humanoid-v4: checked K64 v2 vs OptiQ; no extra uniform exploration")
    fig.tight_layout()
    fig.savefig(output / "latest.png", dpi=160); fig.savefig(output / "latest.pdf")
    plt.close(fig)
    print(json.dumps({key: report[key] for key in ("common_step_all_four_seeds", "aggregate", "paired_mean_gap", "pairs")}, indent=2))


if __name__ == "__main__":
    main()
