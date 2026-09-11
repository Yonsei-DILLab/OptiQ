"""Aligned four-seed finite-policy comparison, retaining all three references.

Read-only for training. Stopped seeds stay in the common-horizon aggregate;
historical reference provenance differs and is not an exploration-only ablation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_v2_confirmation import read_run
from scripts.assess_v2_confirmation import FINAL_STEPS, drawdown, seed_statistics

LABELS = {"finite": "Finite-policy candidate", "v2": "Completed continuous v2",
          "optiq": "Matched OptiQ (uniform 0%)", "historical": "Historical OptiQ (uniform 10%)"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through-step", type=int)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/v2_improvement/finite_report")
    args = parser.parse_args()
    base = ROOT/"outputs/v2_improvement"
    finite = json.loads((base/"finite_screen_manifest.json").read_text())
    confirmation = json.loads((base/"confirmation_manifest.json").read_text())
    historical = json.loads((base/"historical_behavior010_reference.json").read_text())
    assert len(finite["runs"]) == 4 and {x["seed"] for x in finite["runs"]} == set(range(4))
    runs = {(x["method"], x["seed"]): read_run(x, 10) for x in confirmation["runs"]}
    for item in finite["runs"]:
        runs["finite", item["seed"]] = read_run(item, 10)
    for item in historical["runs"]:
        path = next(Path(item["directory"]).glob("eval/*/evaluations.npz"))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["evaluation_sha256"]
        with np.load(path) as data:
            steps, results = data["timesteps"].copy(), data["results"].copy()
        assert results.shape == (len(steps), 10) and np.isfinite(results).all()
        runs["historical", item["seed"]] = {"steps": steps, "returns": results.mean(axis=1)}
    assert set(runs) == {(method, seed) for method in LABELS for seed in range(4)}
    common = runs["finite", 0]["steps"]
    for run in runs.values():
        assert np.all(np.diff(run["steps"]) > 0)
        common = np.intersect1d(common, run["steps"])
    if args.through_step is not None:
        if args.through_step < 1:
            parser.error("Positive cutoff required")
        common = common[common <= args.through_step]
    if len(common) < 3:
        raise ValueError("Need three shared evaluation checkpoints")
    curves = {m: np.stack([runs[m, s]["returns"][np.searchsorted(runs[m, s]["steps"], common)]
                           for s in range(4)]) for m in LABELS}
    response = subprocess.run(["supervisorctl", "status", *[x["service"] for x in finite["runs"]]],
                              capture_output=True, text=True)
    statuses = {x.split()[0]: x.split()[1] for x in response.stdout.splitlines() if len(x.split()) >= 2}
    primary = {"available": bool(np.isin(FINAL_STEPS, common).all()), "steps": FINAL_STEPS.tolist()}
    if primary["available"]:
        primary["methods"] = {m: seed_statistics(v[:, np.searchsorted(common, FINAL_STEPS)].mean(axis=1))
                               for m, v in curves.items()}
    runtime = []
    for item in finite["runs"]:
        run = runs["finite", item["seed"]]
        row = {"seed": item["seed"], "status": statuses.get(item["service"], "UNKNOWN"),
               **{k: run[k] for k in ("logged_steps", "completed_marker", "training_rate",
                                      "acceptance", "ess", "entropy", "actor_std")}}
        row["seconds_to_100k_at_recent_rate"] = ((100000-run["logged_steps"])/run["training_rate"]
            if run["training_rate"] and run["logged_steps"] < 100000 else None)
        runtime.append(row)
    record = {"generated_utc": datetime.now(timezone.utc).isoformat(),
              "common_step_all_16_runs": int(common[-1]), "common_steps": common.tolist(),
              "curves": {m: v.tolist() for m, v in curves.items()},
              "tail_checkpoints": common[-5:].tolist(),
              "tail5": {m: seed_statistics(v[:, -5:].mean(axis=1)) for m, v in curves.items()},
              "normalized_auc": {m: seed_statistics(np.sum(.5*(v[:, 1:]+v[:, :-1])*np.diff(common), axis=1)
                                                        /(common[-1]-common[0])) for m, v in curves.items()},
              "drawdowns": {m: [drawdown(common, v[s]) for s in range(4)] for m, v in curves.items()},
              "primary_900k_1m": primary, "runtime_diagnostics_current": runtime,
              "historical_scope": historical["scope"],
              "limitations": ["All four seeds retained at identical evaluation steps.",
                  "Shading is training-seed SD, not a confidence interval.",
                  "Runtime diagnostics may be newer than the aligned comparison cutoff.",
                  "Early curves and sampled guard acceptance do not establish policy improvement or final performance."]}
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output/f"step_{int(common[-1]):07d}"
    stem.with_suffix(".json").write_text(json.dumps(record, indent=2)+"\n")
    lines = ["# Humanoid finite-policy screen", "",
             f"Shared horizon of all 16 runs: {common[-1]:,} environment steps.",
             f"Recent scores average checkpoints: {common[-5:].tolist()}.", "",
             "| Method | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Mean ± seed SD |",
             "|---|---:|---:|---:|---:|---:|"]
    for method, values in record["tail5"].items():
        scores = " | ".join(f"{v:.1f}" for v in values["seed_scores"])
        lines.append(f"| {LABELS[method]} | {scores} | {values['mean']:.1f} ± {values['seed_sd']:.1f} |")
    lines += ["", f"Predeclared 900k–1M primary window available: {primary['available']}.",
              "", historical["scope"], ""]
    stem.with_suffix(".md").write_text("\n".join(lines))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"finite": "#18846d", "v2": "#176ab0", "optiq": "#555555", "historical": "#bd7405"}
    figure, axes = plt.subplots(1, 5, figsize=(17, 3.8), sharey=True)
    for method, values in curves.items():
        mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
        style = "--" if method == "historical" else "-"
        axes[0].plot(common/1000, mean, style, color=colors[method], label=LABELS[method], linewidth=1.4)
        axes[0].fill_between(common/1000, mean-sd, mean+sd, color=colors[method], alpha=.10)
        for seed in range(4):
            axes[seed+1].plot(common/1000, values[seed], style, color=colors[method], linewidth=1)
    axes[0].set_title("Four-seed mean ± SD")
    axes[0].set_ylabel("Evaluation return")
    for seed in range(4):
        axes[seed+1].set_title(f"Seed {seed}")
    for axis in axes:
        axis.set_xlabel("Environment steps (thousands)")
        axis.grid(alpha=.2)
    figure.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=8)
    figure.suptitle("Humanoid-v4 · Finite-policy screen · Historical collection/source differs", fontsize=12)
    figure.tight_layout(rect=(0, .09, 1, .96))
    figure.savefig(stem.with_suffix(".png"), dpi=160)
    figure.savefig(stem.with_suffix(".pdf"))
    plt.close(figure)
    print(json.dumps({k: record[k] for k in ("common_step_all_16_runs", "tail5", "runtime_diagnostics_current")}, indent=2))


if __name__ == "__main__":
    main()
