"""Wait for completed finite-policy training, then produce final evidence.

Never starts, stops or restarts a training run. A stopped/incomplete seed is
reported as an incomplete screen, never omitted or presented as a final win.
"""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_v2_finite import completion_gate, digest, load_references


def main():
    base = ROOT/"outputs/v2_improvement"
    lock = (base/"finite_finisher.lock").open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = json.loads((base/"finite_final_evaluation_protocol.json").read_text())
    manifest = json.loads(Path(protocol["candidate_manifest"]).read_text())
    assert digest(protocol["common_evaluator"]) == protocol["common_evaluator_sha256"]
    load_references(protocol)
    output = base/"finite_finisher_status.json"
    state = {"started_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
             "script_sha256": digest(__file__), "stage": "waiting_for_training", "goal_complete": False}

    def save(stage, **fields):
        state.update(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat(), **fields)
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2)); temporary.replace(output)

    while True:
        pending = completion_gate(manifest, protocol)
        save("waiting_for_training", pending=pending)
        if not pending:
            break
        if any(r["status"] in {"STOPPED", "EXITED", "FATAL"} for r in pending):
            save("screen_incomplete", reason="At least one seed ended without the complete final protocol.")
            print(json.dumps(state), flush=True)
            return
        time.sleep(30)
    save("reporting_primary_window")
    subprocess.run([sys.executable, str(ROOT/"scripts/report_v2_finite.py"), "--through-step", "1000000"], check=True, cwd=ROOT)
    save("independent_final_evaluation")
    subprocess.run([sys.executable, str(ROOT/"scripts/evaluate_v2_finite.py")], check=True, cwd=ROOT)
    result_path = base/"finite_independent_1000000.json"
    result = json.loads(result_path.read_text())
    assert result["summary"]["complete"] and len(result["results"]) == 4
    save("complete", result_path=str(result_path), result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest())
    print(json.dumps(state), flush=True)


if __name__ == "__main__":
    main()
