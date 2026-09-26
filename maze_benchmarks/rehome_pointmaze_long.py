"""Audit and rehome a paused long PointMaze queue without touching live learners."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import time


REQUIRED_SETTINGS = {
    "target_steps": None,
    "budget_multiplier": 10,
    "eval_every": None,
    "final_eval_episodes": 2000,
}
AUDIT_NAME = "gpu-failure-rehome-to-199.json"


def atomic_json(path: Path, payload: dict) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temp, path)


def check_settings(queue: dict, plan: dict) -> None:
    if queue.get("source_commit") != plan["training_source_commit"]:
        raise ValueError("training source differs")
    for key, expected in REQUIRED_SETTINGS.items():
        if queue.get(key) != expected:
            raise ValueError(f"{key} differs: {queue.get(key)}")


def destination(root: Path, plan: dict, plan_commit: str) -> Path:
    if str(root) != plan["destination_root"]:
        raise ValueError("wrong destination root")
    names = plan["pending_to_transfer"] + plan["failed_to_restart_fresh"]
    if len(names) != len(set(names)):
        raise ValueError("duplicate names in plan")
    with (root / "queue.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / "queue.json"
        queue = json.loads(path.read_text())
        check_settings(queue, plan)
        if queue["state"] != "running":
            raise ValueError("destination queue is not running")
        existing = {j["name"]: j for j in queue["jobs"]}
        if any(n in existing or (root / "runs" / n).exists() for n in names):
            raise ValueError("destination already has one of the planned jobs")
        if any(n not in existing or existing[n]["state"] != "pending"
               for n in plan["already_replaced"]):
            raise ValueError("existing replacement is missing or claimed")
        sidecar = root / AUDIT_NAME
        if sidecar.exists():
            raise FileExistsError(sidecar)
        now = time.time()
        audit = dict(plan=plan, plan_commit=plan_commit,
                     previous_queue=queue, queued_at=now,
                     appended_names=names)
        queue["jobs"].extend(dict(name=n, state="pending",
                                  transfer_sidecar=str(sidecar)) for n in names)
        queue["updated"] = now
        atomic_json(sidecar, audit)
        atomic_json(path, queue)
    return sidecar


def source(root: Path, plan: dict, plan_commit: str, destination_audit: Path) -> Path:
    if str(root) != plan["source_root"]:
        raise ValueError("wrong source root")
    receipt = json.loads(destination_audit.read_text())
    expected = plan["pending_to_transfer"] + plan["failed_to_restart_fresh"]
    if (receipt.get("plan_commit") != plan_commit or receipt.get("plan") != plan or
            receipt.get("appended_names") != expected):
        raise ValueError("destination receipt does not match plan")
    with (root / "queue.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / "queue.json"
        queue = json.loads(path.read_text())
        check_settings(queue, plan)
        if queue["state"] != "paused":
            raise ValueError("source queue is no longer paused")
        existing = {j["name"]: j for j in queue["jobs"]}
        if any(existing.get(n, {}).get("state") != "pending"
               for n in plan["pending_to_transfer"]):
            raise ValueError("source pending job was claimed or changed")
        if any(existing.get(n, {}).get("state") != "failed"
               for n in plan["failed_to_restart_fresh"]):
            raise ValueError("source failed job was changed")
        if any(existing.get(n, {}).get("state") != "failed"
               for n in plan["already_replaced"]):
            raise ValueError("old Hard SQL failure was changed")
        sidecar = root / AUDIT_NAME
        if sidecar.exists():
            raise FileExistsError(sidecar)
        now = time.time()
        audit = dict(plan=plan, plan_commit=plan_commit,
                     destination_receipt=receipt, previous_queue=queue,
                     transferred_at=now)
        for name in plan["pending_to_transfer"]:
            existing[name].update(state="transferred",
                                  destination=plan["destination_host"],
                                  transferred_at=now)
        for name in plan["failed_to_restart_fresh"]:
            existing[name]["replacement_on"] = plan["destination_host"]
        queue["updated"] = now
        atomic_json(sidecar, audit)
        atomic_json(path, queue)
    return sidecar


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--phase", choices=("destination", "source"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-commit", required=True)
    parser.add_argument("--destination-audit", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if args.phase == "destination":
        if args.destination_audit:
            parser.error("destination-audit belongs only to source phase")
        result = destination(args.root, plan, args.plan_commit)
    else:
        if not args.destination_audit:
            parser.error("source phase requires destination-audit")
        result = source(args.root, plan, args.plan_commit, args.destination_audit)
    print(json.dumps({"phase": args.phase, "audit": str(result)}))


if __name__ == "__main__":
    main()
