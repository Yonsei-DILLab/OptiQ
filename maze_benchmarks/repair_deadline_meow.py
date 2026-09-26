"""Explicit, one-time recovery of MEOW's pre-model missing-tyro failure."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import time

from .rebalance_deadline import TRAINING, write


def repair(root: Path, readiness: Path, repair_commit: str):
    if len(repair_commit) != 40:
        raise ValueError("full committed repair SHA required")
    ready = json.loads(readiness.read_text())
    if (ready["training_source"] != TRAINING or
            not ready["cpu_model_actions_finite"] or not ready["cpu_model_q_finite"] or
            ready["optimizer_updates"] != 0 or ready["packages"]["tyro"] != "0.8.14"):
        raise ValueError("dependency/model import verification missing")
    name = "pm_medium-meow-s0"
    with (root / "queue.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        queue = json.loads((root / "queue.json").read_text())
        audit = root / "meow-dependency-repair.json"
        if audit.exists():
            raise FileExistsError("one-time repair already recorded; no automatic retry")
        if queue["source_commit"] != TRAINING or queue["state"] != "paused":
            raise ValueError("unexpected queue source/state")
        failed = [j for j in queue["jobs"] if j["state"] == "failed"]
        if len(failed) != 1 or failed[0]["name"] != name:
            raise ValueError("failure inventory changed")
        job = failed[0]
        if job.get("stage") != "preflights" or job["host"] != "vast-heechan-46":
            raise ValueError("only the known pre-model setup failure is recoverable")
        try:
            os.kill(job["pid"], 0)
        except ProcessLookupError:
            pass
        else:
            raise ValueError("failed worker is still alive")
        log = root / "logs" / f"{name}-preflights.log"
        if "ModuleNotFoundError: No module named 'tyro'" not in log.read_text():
            raise ValueError("failure cause differs from investigated dependency omission")
        if ((root / "runs" / name).exists() or
                (root / "preflights" / name / "progress.json").exists() or
                (root / "proofs" / f"{name}-preflights.json").exists()):
            raise ValueError("learning already started; cannot retry this way")
        archive = root / "preserved-attempts" / "meow-missing-tyro"
        archive.mkdir(parents=True, exist_ok=False)
        before = json.loads(json.dumps(queue))
        write(archive / "queue-before.json", before)
        write(archive / "readiness.json", ready)
        for relative in (f"preflights/{name}", f"jobs/{name}.json",
                         f"logs/{name}-preflights.log", f"logs/{name}-preflights-command.json"):
            path = root / relative
            if path.exists():
                destination = archive / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                path.rename(destination)
        for key in ("worker", "pid", "started", "stage", "updated", "failed", "error"):
            job.pop(key, None)
        job.update(state="pending", dependency_repair_commit=repair_commit,
                   preserved_attempt=str(archive))
        queue["recovered_error"] = queue.pop("error")
        queue.update(state="running", updated=time.time())
        write(audit, dict(before=before, after=queue, readiness=ready,
                          repair_commit=repair_commit, training_source=TRAINING,
                          reason="Dependency installed; preserve failed pre-model attempt and rerun original preflight"))
        write(root / "queue.json", queue)
    return dict(state=queue["state"], recovered_job=name, training_source=TRAINING)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--repair-commit", required=True)
    args = parser.parse_args()
    print(json.dumps(repair(args.root, args.readiness, args.repair_commit)))


if __name__ == "__main__":
    main()
