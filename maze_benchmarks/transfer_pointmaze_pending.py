"""Mark exactly named pending PointMaze jobs as transferred under the queue lock."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import time


def atomic_json(path: Path, payload: dict) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--training-source", required=True)
    parser.add_argument("--transfer-plan-commit", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source-host", choices=("180", "199", "vast1_repair"),
                        required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    names = plan[f"transfer_from_{args.source_host}"]
    if (len(set(names)) != len(names) or
            plan["training_source_commit"] != args.training_source or
            plan["new_host"] != args.destination):
        raise ValueError("invalid transfer plan")
    root = args.root.resolve()
    with (root / "queue.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        queue_path = root / "queue.json"
        state = json.loads(queue_path.read_text())
        if state["source_commit"] != args.training_source or state["state"] != "running":
            raise ValueError("unexpected source or queue state")
        entries = {entry["name"]: entry for entry in state["jobs"]}
        if any(name not in entries or entries[name]["state"] != "pending" for name in names):
            raise ValueError("a requested transfer job was claimed or is absent")
        sidecar = root / f"transfer-to-{args.destination}.json"
        if sidecar.exists():
            raise FileExistsError(sidecar)
        now = time.time()
        audit = dict(training_source_commit=args.training_source,
                     transfer_plan_commit=args.transfer_plan_commit,
                     destination=args.destination, names=names,
                     previous_queue=state, transferred_at=now)
        for name in names:
            entries[name].update(state="transferred", destination=args.destination,
                                 transferred_at=now)
        state["updated"] = now
        if all(entry["state"] == "transferred" for entry in state["jobs"]):
            state["state"] = "transferred"
        atomic_json(sidecar, audit)
        atomic_json(queue_path, state)
    print(json.dumps({"transferred": names, "destination": args.destination}))


if __name__ == "__main__":
    main()
