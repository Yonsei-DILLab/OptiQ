"""Bounded read-only observation of the existing campaign controllers."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--rounds", type=int, default=5)
args = parser.parse_args()
assert 1 <= args.rounds <= 60
base = Path(__file__).resolve().parent
for iteration in range(args.rounds):
    command = [sys.executable, "-m", "antmaze.multimodal.collect_dense_noveld", "--output", str(base)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    if result.returncode:
        print(result.stderr, flush=True)
        raise SystemExit(result.returncode)
    snapshot = json.loads((base / "latest.json").read_text())
    records, attention = [], []
    for host, data in snapshot["hosts"].items():
        if data.get("failure.json"):
            attention.append([host, data["failure.json"]])
        for job in data["status.json"]["jobs"]:
            run = data["runs"].get(job["id"], {})
            if "failure.json" in run:
                attention.append([job["id"], run["failure.json"]])
            if job["status"] == "running" and not data["processes"].get(job["id"]):
                attention.append([job["id"], "Process missing; inspect before taking any action"])
            records.append(dict(id=job["id"], status=job["status"], step=run.get("progress.json", {}).get("env_steps")))
    completed = [row["id"] for row in records if row["status"] == "completed"]
    ready_to_archive = [job for job in completed
                        if not (base / "runs" / job / "archive-verification.json").exists()]
    print(json.dumps(dict(checked_at=snapshot["collected_at"],
        live=sum(row["status"] == "running" for row in records),
        pending=sum(row["status"] == "pending" for row in records),
        completed=completed, ready_to_archive=ready_to_archive, attention=attention,
        steps={row["id"]: row["step"] for row in records if row["step"] is not None})), flush=True)
    if attention or ready_to_archive:
        break
    if iteration + 1 < args.rounds:
        time.sleep(55)
