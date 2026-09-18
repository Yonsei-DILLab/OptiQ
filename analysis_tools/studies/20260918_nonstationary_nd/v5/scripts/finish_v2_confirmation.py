"""Wait for the fixed confirmation, then run its predefined final evaluation.

This service never starts/stops training or chooses a winner. It fails rather
than treating an incomplete/stopped experiment as a completed comparison.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_v2_confirmation import read_run
from scripts.assess_v2_confirmation import EXPECTED_KEYS, FINAL_STEPS, seed_statistics


def completion_ready(items, runs, states):
    if set(items) != EXPECTED_KEYS or set(runs) != EXPECTED_KEYS:
        raise ValueError("All eight prescribed runs are required")
    ready = True
    for key, item in items.items():
        state = states[item["supervisor"]]
        if state == "RUNNING":
            ready = False
            continue
        if state != "EXITED":
            raise RuntimeError(f"Unexpected training state: {key}: {state}")
        run = runs[key]
        if not run["completed_marker"] or not np.isin(FINAL_STEPS, run["steps"]).all():
            raise RuntimeError(f"Incomplete training exit: {key}")
        for kind in ("actor", "critic"):
            files = list(Path(item["directory"]).glob(f"checkpoints/*/{kind}_state_1000000.msgpack"))
            if len(files) != 1 or files[0].stat().st_size == 0:
                raise RuntimeError(f"Missing/nonunique final {kind} checkpoint: {key}")
    return ready


def verify_independent(record, manifest, protocol):
    if not record.get("summary", {}).get("complete"):
        raise ValueError("Incomplete independent evaluation; preserve it for manual review")
    assert record["checkpoint_step"] == protocol["checkpoint_step"] == 1000000
    assert record["seed_base"] == protocol["environment_seeds"][0] == 1100000
    assert record["episodes_per_run"] == protocol["episodes_per_model"] == 50
    expected = {(i["seed"], i["method"]): i for i in manifest["runs"]}
    results = {(r["training_seed"], r["method"]): r for r in record["results"]}
    assert len(record["results"]) == 8 and set(results) == set(expected) == EXPECTED_KEYS
    for key, result in results.items():
        assert result["source_commit"] == expected[key]["commit"]
        assert len(result["episodes"]) == 50
        assert [e["env_seed"] for e in result["episodes"]] == protocol["environment_seeds"]
        assert [e["policy_seed"] for e in result["episodes"]] == protocol["policy_seeds"]
        assert hashlib.sha256(Path(result["checkpoint"]).read_bytes()).hexdigest() == result["checkpoint_sha256"]
        for episode in result["episodes"]:
            assert np.isfinite(episode["return"])
            components = episode["environment_metrics"]
            reconstructed = sum(components[k]["sum"] for k in ("reward_linvel", "reward_alive", "reward_quadctrl"))
            assert abs(reconstructed-episode["return"]) < 1e-7


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true", help="Inspect readiness only; do not wait or evaluate")
    args = p.parse_args()
    evidence = ROOT / "outputs/v2_improvement"
    manifest = json.loads((evidence / "confirmation_manifest.json").read_text())
    protocol = json.loads((evidence / "confirmation_final_independent_protocol.json").read_text())
    items = {(i["seed"], i["method"]): i for i in manifest["runs"]}
    assert len(manifest["runs"]) == 8 and set(items) == EXPECTED_KEYS
    assert hashlib.sha256((ROOT / "scripts/evaluate_v2_confirmation.py").read_bytes()).hexdigest() == protocol["common_evaluator_sha256"]
    core_diff = subprocess.check_output(["git", "diff", manifest["runs"][0]["commit"], "--", "optiq_dime", "configs"], cwd=ROOT)
    if core_diff:
        raise RuntimeError("Training/config source changed since launch; inspect evaluation compatibility first")
    lock = (evidence / "confirmation_finisher.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = evidence / "confirmation_finisher_status.json"

    def record(stage, **fields):
        row = {"utc": datetime.now(timezone.utc).isoformat(), "stage": stage, **fields}
        status_path.write_text(json.dumps(row, indent=2) + "\n")
        print(json.dumps(row), flush=True)

    while True:
        result = subprocess.run(["supervisorctl", "status", *[i["supervisor"] for i in items.values()]],
                                capture_output=True, text=True)
        states = {line.split()[0]: line.split()[1] for line in result.stdout.splitlines() if len(line.split()) >= 2}
        if set(states) != {i["supervisor"] for i in items.values()}:
            if args.check:
                raise RuntimeError("Could not observe all supervisor handles")
            record("observation_retry", reason="Could not observe all supervisor handles")
            time.sleep(45)
            continue
        try:
            runs = {key: read_run(item, manifest["eval_episodes"]) for key, item in items.items()}
        except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
            if args.check:
                raise
            record("observation_retry", reason=type(exc).__name__)
            time.sleep(45)
            continue
        ready = completion_ready(items, runs, states)
        record("ready" if ready else "waiting", ready=ready,
               last_evaluation_steps={f"{m}_s{s}": int(r["steps"][-1]) for (s, m), r in runs.items()},
               process_states=states)
        if args.check:
            return
        if ready:
            break
        time.sleep(45)

    # The service may have waited while other work happened in this checkout.
    assert hashlib.sha256((ROOT / "scripts/evaluate_v2_confirmation.py").read_bytes()).hexdigest() == protocol["common_evaluator_sha256"]
    if subprocess.check_output(["git", "diff", manifest["runs"][0]["commit"], "--", "optiq_dime", "configs"], cwd=ROOT):
        raise RuntimeError("Training/config source changed during wait; inspect before evaluation")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MPLBACKEND="Agg")

    def run_script(name, arguments, logfile):
        with (evidence / logfile).open("w") as stream:
            subprocess.run([sys.executable, "-u", str(ROOT / "scripts" / name), *arguments],
                           cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)

    record("reporting")
    for name, log in (("analyze_v2_confirmation.py", "confirmation_report_1000000.log"),
                      ("assess_v2_confirmation.py", "confirmation_assessment_1000000.log"),
                      ("compare_v2_historical.py", "confirmation_historical_comparison_1000000.log")):
        run_script(name, ["--through-step", "1000000"], log)
    for suffix in ("md", "png", "pdf"):
        shutil.copyfile(evidence / f"confirmation_report/latest.{suffix}",
                        evidence / f"confirmation_report/step_1000000.{suffix}")
    assessment = json.loads((evidence / "confirmation_assessment/step_1000000.json").read_text())
    comparison = json.loads((evidence / "confirmation_historical_comparison/step_1000000.json").read_text())
    assert assessment["final_results_ready_for_review"]
    assert comparison["primary_900k_1m"]["available"]
    independent_path = evidence / "confirmation_independent_1000000.json"
    if not independent_path.exists():
        record("evaluating", checkpoint_step=1000000, episodes_per_model=50, seed_base=1100000)
        run_script("evaluate_v2_confirmation.py", ["--step", "1000000", "--episodes", "50", "--seed-base", "1100000"],
                   "confirmation_independent_1000000.log")
    independent = json.loads(independent_path.read_text())
    verify_independent(independent, manifest, protocol)
    historical = json.loads((evidence / "historical_behavior010_independent_1000000.json").read_text())
    assert historical["complete"] and historical["seed_base"] == 1100000 and historical["episodes_per_run"] == 50
    assert historical["common_evaluator_sha256"] == protocol["common_evaluator_sha256"]
    scores = sorted(historical["results"], key=lambda r: r["seed"])
    assert [r["seed"] for r in scores] == [0, 1, 2, 3]
    combined = {"generated_utc": datetime.now(timezone.utc).isoformat(),
                "primary_900k_1m": assessment["primary_final_window"],
                "primary_all_three_methods": comparison["primary_900k_1m"],
                "independent_matched": independent["summary"],
                "independent_historical": seed_statistics([r["mean_return"] for r in scores]),
                "historical_scope": historical["scope"],
                "status": "Ready for human/agent review; no automatic success or policy-improvement claim."}
    (evidence / "confirmation_final_review_inputs.json").write_text(json.dumps(combined, indent=2) + "\n")
    record("complete", results_ready_for_review=True, success_claim=False)


if __name__ == "__main__":
    main()
