"""Resume non-DIPO pending work after separately scheduling DIPO environment repairs."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import time

from .run_papermaze_job import verify_run


def atomic_json(path: Path, payload: dict) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-commit", required=True)
    parser.add_argument("--smoke", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    commit = plan["training_source_commit"]
    proof = verify_run(args.smoke, commit, "simple", "dipo", 256, 4096, True)
    if (proof["steps"], proof["updates"]) != (4352, 16):
        raise ValueError("DIPO smoke update proof mismatch")
    repair = Path(plan["new_root"])
    repair_state = json.loads((repair / "queue.json").read_text())
    if (repair_state["source_commit"] != commit or
            [item["name"] for item in repair_state["jobs"]] != plan["new_queue_order"] or
            any(item["state"] != "pending" for item in repair_state["jobs"])):
        raise ValueError("repair queue is absent or unexpected")
    root = args.root.resolve()
    with (root / "queue.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / "queue.json"
        state = json.loads(path.read_text())
        if state["source_commit"] != commit or state["state"] != "paused":
            raise ValueError("main queue is not the expected paused source")
        entries = {item["name"]: item for item in state["jobs"]}
        if (any(entries[name]["state"] != "failed" for name in plan["previous_failed"]) or
                entries[plan["transfer_pending"]]["state"] != "pending"):
            raise ValueError("failed and pending DIPO states changed")
        for name in plan["previous_failed"]:
            job = json.loads((root / "jobs" / f"{name}.json").read_text())
            log = Path(job["candidates"][0]["log"]).read_text(errors="replace")
            if job["state"] != "failed" or "missing MuJoCo" not in log:
                raise ValueError(f"DIPO failure reason changed: {name}")
        sidecar = root / "envfix-recovery.json"
        if sidecar.exists():
            raise FileExistsError(sidecar)
        now = time.time()
        original = state
        state = json.loads(json.dumps(state))
        entries = {item["name"]: item for item in state["jobs"]}
        entries[plan["transfer_pending"]].update(
            state="transferred", destination=str(repair), transferred_at=now)
        state["state"] = "running"
        state["recovered_error"] = state.pop("error")
        state["recovery"] = dict(at=now, plan_commit=args.plan_commit,
                                 repair_root=str(repair), dipo_smoke=proof)
        state["updated"] = now
        atomic_json(sidecar, dict(before=original, after=state,
                                  plan_commit=args.plan_commit,
                                  dipo_smoke=proof))
        atomic_json(path, state)
    print(json.dumps({"main_queue": state["state"],
                      "repair_jobs": plan["new_queue_order"]}))


if __name__ == "__main__":
    main()
