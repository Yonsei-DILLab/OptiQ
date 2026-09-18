"""Read-only fixed-window and drawdown assessment of the eight confirmation runs.

The primary window is fixed at 900k--1M. Missing seeds/checkpoints prevent a
final comparison; this module neither selects winners nor controls training.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_v2_confirmation import read_run

EXPECTED_KEYS = {(seed, method) for seed in range(4) for method in ("v2", "optiq")}
FINAL_STEPS = np.arange(900000, 1000001, 5000)


def seed_statistics(values):
    values = np.asarray(values, dtype=float)
    return {"seed_scores": values.tolist(), "mean": float(values.mean()),
            "seed_sd": float(values.std(ddof=1)), "minimum_seed": float(values.min()),
            "maximum_seed": float(values.max())}


def final_window(runs):
    if set(runs) != EXPECTED_KEYS:
        raise ValueError("All four prescribed seeds for both methods are required")
    missing = {f"{method}_seed{seed}": np.setdiff1d(FINAL_STEPS, run["steps"]).tolist()
               for (seed, method), run in runs.items()}
    missing = {name: steps for name, steps in missing.items() if steps}
    result = {"start": 900000, "end": 1000000, "interval": 5000,
              "available": not missing, "missing_checkpoints": missing}
    if missing:
        return result
    scores, final_scores = {}, {}
    for method in ("v2", "optiq"):
        scores[method], final_scores[method] = [], []
        for seed in range(4):
            run = runs[seed, method]
            values = run["returns"][np.searchsorted(run["steps"], FINAL_STEPS)]
            scores[method].append(float(values.mean()))
            final_scores[method].append(float(values[-1]))
    gaps = np.array(scores["v2"]) - np.array(scores["optiq"])
    result.update(methods={m: seed_statistics(v) for m, v in scores.items()},
                  final_checkpoint={m: seed_statistics(v) for m, v in final_scores.items()},
                  paired_gaps=seed_statistics(gaps), positive_seed_gaps=int((gaps > 0).sum()))
    return result


def drawdown(steps, returns, start=100000, width=3):
    """Largest relative drop of the rolling mean, with observed recovery only.

    Durations use environment steps. An unfinished recovery is censored at the
    latest common step, not counted as a completed recovery. This is descriptive,
    not a stability significance test or a policy-improvement certificate.
    """
    steps, returns = np.asarray(steps), np.asarray(returns, dtype=float)
    if len(returns) < width:
        return {"available": False}
    smooth = np.convolve(returns, np.ones(width) / width, mode="valid")
    ends = steps[width-1:]
    selected = ends >= start
    smooth, ends = smooth[selected], ends[selected]
    if not len(smooth) or not np.any(smooth > 0):
        return {"available": False}
    peaks = np.maximum.accumulate(smooth)
    relative = np.full_like(smooth, np.nan)
    np.divide(peaks-smooth, peaks, out=relative, where=peaks > 0)
    trough = int(np.nanargmax(relative))
    peak = int(np.argmax(smooth[:trough+1]))
    recovered = np.flatnonzero(smooth[trough:] >= smooth[peak])
    recovery = trough + int(recovered[0]) if len(recovered) else None
    return {"available": True, "rolling_checkpoints": width, "start_step": int(ends[0]),
            "end_step": int(ends[-1]), "maximum_drawdown_fraction": float(relative[trough]),
            "current_drawdown_fraction": float(relative[-1]),
            "peak_step": int(ends[peak]), "trough_step": int(ends[trough]),
            "peak_return": float(smooth[peak]), "trough_return": float(smooth[trough]),
            "recovery_step": int(ends[recovery]) if recovery is not None else None,
            "peak_to_recovery_steps": int(ends[recovery]-ends[peak]) if recovery is not None else None,
            "unrecovered_elapsed_steps": int(ends[-1]-ends[peak]) if recovery is None else 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/v2_improvement/confirmation_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/v2_improvement/confirmation_assessment")
    parser.add_argument("--through-step", type=int,
                        help="Freeze evaluation metrics at this step; process states remain current.")
    args = parser.parse_args()
    if args.through_step is not None and args.through_step < 1:
        parser.error("--through-step must be positive")
    manifest = json.loads(args.manifest.read_text())
    assert manifest["cap"] == 1000000 and manifest["eval_episodes"] == 10
    items = {(r["seed"], r["method"]): r for r in manifest["runs"]}
    assert len(manifest["runs"]) == 8 and set(items) == EXPECTED_KEYS
    runs = {key: read_run(item, manifest["eval_episodes"]) for key, item in items.items()}
    if args.through_step is not None:
        for run in runs.values():
            selected = run["steps"] <= args.through_step
            run["steps"], run["returns"] = run["steps"][selected], run["returns"][selected]
    status = subprocess.run(["supervisorctl", "status", *[i["supervisor"] for i in items.values()]],
                            capture_output=True, text=True)
    states = {s.split()[0]: s.split()[1] for s in status.stdout.splitlines() if len(s.split()) >= 2}
    common = next(iter(runs.values()))["steps"]
    for run in runs.values():
        common = np.intersect1d(common, run["steps"])
    if len(common) < 3:
        raise ValueError("At least three shared evaluations are needed")
    stability, auc = {}, {"v2": [], "optiq": []}
    for (seed, method), run in sorted(runs.items()):
        values = run["returns"][np.searchsorted(run["steps"], common)]
        stability[f"{method}_seed{seed}"] = drawdown(common, values)
        area = np.sum(.5 * (values[:-1] + values[1:]) * np.diff(common))
        auc[method].append(float(area / (common[-1]-common[0])))
    primary = final_window(runs)
    completed = all(run["completed_marker"] and states.get(items[key]["supervisor"]) == "EXITED"
                    for key, run in runs.items())
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(),
              "requested_evaluation_cutoff": args.through_step,
              "common_evaluation_step": int(common[-1]), "primary_final_window": primary,
              "all_runs_completed": completed, "final_results_ready_for_review": completed and primary["available"],
              "normalized_common_step_auc": {m: seed_statistics(v) for m, v in auc.items()},
              "stability": stability,
              "processes": [{"seed": seed, "method": method, "state": states.get(item["supervisor"], "UNKNOWN"),
                             "completed_marker": runs[seed, method]["completed_marker"]}
                            for (seed, method), item in sorted(items.items())],
              "limitations": ["Four training seeds; evaluation episodes are not independent training seeds.",
                              "AUC and drawdowns use only the horizon shared by all eight runs.",
                              "No significance or theoretical policy-improvement claim follows from these metrics."]}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "latest.json").write_text(json.dumps(report, indent=2)+"\n")
    (args.output / f"step_{int(common[-1]):07d}.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
