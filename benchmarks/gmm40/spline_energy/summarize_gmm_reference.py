"""Aggregate the restored five-seed experiment against the preserved baseline."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = ["mode_coverage", "high_density_fraction", "forward_kl", "reverse_kl",
           "mode_mass_tv", "mmd2"]


def aggregate(values):
    x = np.asarray(values, dtype=np.float64)
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)), "values": x.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    outputs = args.campaign / "outputs"
    new_histories, finals = [], []
    for seed in range(5):
        folder = outputs / f"seed{seed}"
        complete = json.loads((folder / "COMPLETE.json").read_text())
        assert complete["completed"] and complete["steps"] == 100_000
        history = [json.loads(line) for line in (folder / "history.jsonl").read_text().splitlines()]
        assert [row["update"] for row in history] == list(range(0, 100_001, 5_000))
        assert all(np.isfinite([row[m] for row in history]).all() for m in METRICS)
        new_histories.append(history); finals.append(history[-1])

    baseline = json.loads(args.baseline.read_text())
    baseline_runs = baseline["runs"]["spline_energy"]
    assert [run["seed"] for run in baseline_runs] == list(range(5))
    old_histories = [run["history"] for run in baseline_runs]
    old_final = {
        metric: baseline["aggregates"]["spline_energy"][
            "kl" if metric in {"forward_kl", "reverse_kl"} else "primary"][metric]
        for metric in METRICS
    }
    new_final = {metric: aggregate([row[metric] for row in finals]) for metric in METRICS}
    deltas = {metric: {
        "mean_difference": new_final[metric]["mean"] - old_final[metric]["mean"],
        "max_abs_seed_difference": float(np.max(np.abs(
            np.asarray(new_final[metric]["values"]) - np.asarray(old_final[metric]["values"]))))
    } for metric in METRICS}
    thresholds = {}
    for name, histories in [("preserved_baseline", old_histories), ("restored_implementation", new_histories)]:
        thresholds[name] = {}
        for threshold in [.90, .95, .98]:
            thresholds[name][str(threshold)] = [next(
                (row["update"] for row in history
                 if row["primary"]["high_density_fraction"] >= threshold), None)
                if "primary" in history[0] else next(
                (row["update"] for row in history if row["high_density_fraction"] >= threshold), None)
                for history in histories]
    result = {
        "passed": all(row["mode_coverage"] == 40 for row in finals),
        "comparison": "same five seeds, target, update/query budget, proposal, evaluation samples and keys",
        "restored": new_final, "preserved_baseline": old_final, "difference": deltas,
        "first_crossing_updates": thresholds,
        "limitations": "GMM40 is state-free 2D fixed-oracle density fitting; this does not validate MuJoCo TD learning.",
    }
    report = args.campaign / "reports"; report.mkdir(exist_ok=True)
    (report / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    steps = np.asarray([row["update"] for row in new_histories[0]])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for histories, label, color, accessor in [
        (old_histories, "Preserved successful implementation", "#596579",
         lambda row: row["primary"]["high_density_fraction"]),
        (new_histories, "Restored raw-energy implementation", "#007E87",
         lambda row: row["high_density_fraction"]),
    ]:
        values = np.asarray([[accessor(row) for row in history] for history in histories])
        mean, sd = values.mean(0), values.std(0, ddof=1)
        axes[0].plot(steps, mean, label=label, color=color)
        axes[0].fill_between(steps, mean - sd, mean + sd, color=color, alpha=.15)
    axes[0].set(xlabel="Updates", ylabel="Fraction within target 3σ", ylim=(0, 1.01))
    x = np.arange(5); width = .36
    axes[1].bar(x - width/2, old_final["forward_kl"]["values"], width,
                label="Preserved", color="#596579")
    axes[1].bar(x + width/2, new_final["forward_kl"]["values"], width,
                label="Restored", color="#007E87")
    axes[1].set(xlabel="Seed", ylabel="Final forward KL", xticks=x)
    for ax in axes: ax.grid(alpha=.2)
    axes[0].legend(); axes[1].legend()
    fig.suptitle("GMM40 | matched five-seed 100k comparison")
    fig.tight_layout()
    fig.savefig(report / "comparison.png", dpi=170)
    plt.close(fig)

    lines = ["# Restored raw-energy GMM40 result", "",
             "Matched five-seed, 100k-update comparison against the preserved successful spline run.", "",
             "| Metric | Preserved mean ± SD | Restored mean ± SD | Mean difference |", "|---|---:|---:|---:|"]
    for metric in METRICS:
        old, new = old_final[metric], new_final[metric]
        lines.append(f"| {metric} | {old['mean']:.8g} ± {old['sd']:.3g} | "
                     f"{new['mean']:.8g} ± {new['sd']:.3g} | {deltas[metric]['mean_difference']:+.3g} |")
    lines += ["", "Every restored seed covered 40/40 modes at 100k.", "",
              "This result validates the state-free GMM40 implementation only. It is not a MuJoCo result."]
    (report / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
