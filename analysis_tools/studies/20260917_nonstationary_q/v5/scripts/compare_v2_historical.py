"""Compare all fixed Humanoid seeds, retaining the stronger historical reference.

Read-only with respect to training. The historical collection/source differs
from the current paired experiment; its difference is not a causal ablation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_v2_confirmation import read_run
from scripts.assess_v2_confirmation import FINAL_STEPS, drawdown, seed_statistics

LABELS = {
    "v2": "v2 (no extra uniform)",
    "optiq": "OptiQ (no extra uniform)",
    "historical": "Historical OptiQ (10% uniform)",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through-step", type=int)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/v2_improvement/confirmation_historical_comparison")
    args = parser.parse_args()
    if args.through_step is not None and args.through_step < 1:
        parser.error("--through-step must be positive")
    evidence = ROOT / "outputs/v2_improvement"
    manifest = json.loads((evidence / "confirmation_manifest.json").read_text())
    historical = json.loads((evidence / "historical_behavior010_reference.json").read_text())
    runs = {(r["method"], r["seed"]): read_run(r, 10) for r in manifest["runs"]}
    assert len(manifest["runs"]) == 8
    assert set(runs) == {(m, s) for m in ("v2", "optiq") for s in range(4)}
    assert len(historical["runs"]) == 4
    assert {r["seed"] for r in historical["runs"]} == set(range(4))
    for item in historical["runs"]:
        directory = Path(item["directory"])
        cfg = json.loads((directory / "config.json").read_text())
        assert cfg["seed"] == item["seed"] and cfg["env_name"] == "Humanoid-v4"
        assert cfg["alg"]["behavior_uniform_probability"] == .1
        path = next(directory.glob("eval/*/evaluations.npz"))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["evaluation_sha256"]
        with np.load(path) as data:
            steps, results = data["timesteps"].copy(), data["results"].copy()
        assert results.shape == (len(steps), 10) and np.isfinite(results).all()
        runs["historical", item["seed"]] = {"steps": steps, "returns": results.mean(axis=1)}
    common = next(iter(runs.values()))["steps"]
    for run in runs.values():
        assert np.all(np.diff(run["steps"]) > 0)
        common = np.intersect1d(common, run["steps"])
    if args.through_step is not None:
        common = common[common <= args.through_step]
    if len(common) < 3:
        raise ValueError("Three shared evaluation checkpoints are required")
    curves = {m: np.stack([runs[m, s]["returns"][np.searchsorted(runs[m, s]["steps"], common)]
                           for s in range(4)]) for m in LABELS}
    complete = bool(np.isin(FINAL_STEPS, common).all())
    final = {"available": complete, "steps": FINAL_STEPS.tolist()}
    if complete:
        final["methods"] = {m: seed_statistics(v[:, np.searchsorted(common, FINAL_STEPS)].mean(axis=1))
                            for m, v in curves.items()}
    record = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "requested_cutoff": args.through_step, "common_step_all_12_runs": int(common[-1]),
        "tail_checkpoints": common[-3:].tolist(),
        "tail3": {m: seed_statistics(v[:, -3:].mean(axis=1)) for m, v in curves.items()},
        "primary_900k_1m": final,
        "normalized_auc": {m: seed_statistics(np.sum(.5 * (v[:, 1:] + v[:, :-1]) * np.diff(common), axis=1)
                                                / (common[-1] - common[0])) for m, v in curves.items()},
        "drawdowns": {m: [drawdown(common, v[s]) for s in range(4)] for m, v in curves.items()},
        "historical_scope": historical["scope"],
        "limitations": ["All four training seeds are retained; shaded areas are seed SD, not confidence intervals.",
                        "Historical behavior and source provenance differ; this is not a causal exploration ablation.",
                        "Finite-sample return curves do not certify statewise soft policy improvement."]}
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.output / f"step_{int(common[-1]):07d}"
    stem.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    lines = ["# Humanoid: v2 and both OptiQ references", "",
             f"All 12 runs share evaluations through {common[-1]:,} environment steps.",
             f"Recent scores average checkpoints {', '.join(str(x) for x in common[-3:])}.", "",
             "| Method | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Mean ± seed SD |",
             "|---|---:|---:|---:|---:|---:|"]
    for m, values in record["tail3"].items():
        cells = " | ".join(f"{x:.1f}" for x in values["seed_scores"])
        lines.append(f"| {LABELS[m]} | {cells} | {values['mean']:.1f} ± {values['seed_sd']:.1f} |")
    lines += ["", f"Full predefined 900k–1M window available: {complete}."]
    if complete:
        lines += ["", "| Method | 900k–1M mean ± seed SD |", "|---|---:|"]
        for m, values in final["methods"].items():
            lines.append(f"| {LABELS[m]} | {values['mean']:.1f} ± {values['seed_sd']:.1f} |")
    lines += ["", historical["scope"], ""]
    stem.with_suffix(".md").write_text("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"v2": "#176ab0", "optiq": "#4b4b4b", "historical": "#c47b0a"}
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.8), sharey=True)
    for m, values in curves.items():
        mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
        style = "--" if m == "historical" else "-"
        axes[0].plot(common/1000, mean, style, color=colors[m], label=LABELS[m], linewidth=1.3)
        axes[0].fill_between(common/1000, mean-sd, mean+sd, color=colors[m], alpha=.12)
        for seed in range(4):
            axes[seed+1].plot(common/1000, values[seed], style, color=colors[m], linewidth=1)
    axes[0].set_title("Four-seed mean ± SD")
    axes[0].set_ylabel("Evaluation return")
    for seed in range(4):
        axes[seed+1].set_title(f"Training seed {seed}")
    for ax in axes:
        ax.set_xlabel("Environment steps (thousands)")
        ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=9)
    fig.suptitle("Humanoid-v4 · Historical reference has different collection/source provenance", fontsize=12)
    fig.tight_layout(rect=(0, .09, 1, .96))
    fig.savefig(stem.with_suffix(".png"), dpi=160)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    print(json.dumps({k: record[k] for k in ("common_step_all_12_runs", "tail3", "primary_900k_1m")}, indent=2))


if __name__ == "__main__":
    main()
