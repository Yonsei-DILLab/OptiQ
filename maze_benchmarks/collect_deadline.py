"""Read-only collection and SHA verification of the 1M replacement campaign."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np

from .deadline_queue import CAMPAIGN, PLAN, verify
from .run_nway_job import verify_run as verify_historical

TRAINING_SOURCE = "a55aaf13f7803b5cf8da7ba56f318675c7794171"
REMOTE = "/home/heechan/optiq-experiments/" + CAMPAIGN


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def sync(remote, local):
    local.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-az", "--exclude", "learner/", remote, str(local)], check=True)


def snapshot(host):
    script = f'''import json
from pathlib import Path
r=Path({REMOTE!r})
out={{"host":{host!r},"queue":None,"progress":{{}},"jobs":{{}},"transfers":{{}}}}
if (r/"queue.json").exists():
 out["queue"]=json.loads((r/"queue.json").read_text())
 for p in r.glob("runs/*/progress.json"):out["progress"][p.parent.name]=json.loads(p.read_text())
 for p in r.glob("jobs/*.json"):out["jobs"][p.stem]=json.loads(p.read_text())
 for p in r.glob("transfer-*.json"):out["transfers"][p.name]=json.loads(p.read_text())
print(json.dumps(out))'''
    return json.loads(subprocess.check_output(["ssh", "-o", "ConnectTimeout=10", host,
                                               "python3 -c " + shlex.quote(script)], text=True))


def audit_ownership(plan, snapshots):
    expected = {j["name"]: j for j in plan["jobs"]}
    owners = {}
    for snap in snapshots:
        if snap["queue"] is None: continue
        for job in snap["queue"]["jobs"]:
            name = job["name"]
            if name not in expected: raise ValueError(f"unplanned job: {name}")
            if job["state"] == "transferred":
                if job.get("destination") != expected[name]["host"]:
                    raise ValueError("transfer destination disagrees with committed plan")
                continue
            if name in owners: raise ValueError(f"duplicate active ownership: {name}")
            if any(job.get(k) != v for k, v in expected[name].items()):
                raise ValueError(f"queue configuration/placement differs: {name}")
            if job["host"] != snap["host"]: raise ValueError("wrong queue host")
            owners[name] = snap["host"]
    return dict(owners=owners, missing=sorted(set(expected) - set(owners)))


def verify_archive(root, plan, results):
    fresh = {j["name"]: j for j in plan["jobs"]}
    reused = {j["name"]: j for j in plan["reused"]}
    expected = set(fresh) | set(reused)
    if not set(results) <= expected: raise ValueError("unrequested results in archive")
    for name, result in results.items():
        directory = root / "runs" / name
        for relative, wanted in result["sha256"].items():
            if result["reused"] and any(p.startswith(".") for p in Path(relative).parts):
                continue
            if digest(directory / relative) != wanted:
                raise ValueError(f"archived content changed: {name}/{relative}")
        if name in fresh:
            if result["reused"]: raise ValueError("fresh result mislabeled as reuse")
            checked = verify(directory, fresh[name], TRAINING_SOURCE, False)
            if checked["sha256"] != result["sha256"]:
                raise ValueError("fresh archived file inventory differs")
        else:
            if not result["reused"]: raise ValueError("historical result duplicated")
            wanted = reused[name]
            verify_historical(directory, wanted["source_commit"], wanted["task"],
                              "optiq", 1_000_000, 1024)
            config = json.loads((directory / "config.json").read_text())
            if config["temperature"] != 1.: raise ValueError("historical temperature mismatch")
    return dict(verified=sorted(results), missing=sorted(expected - set(results)),
                complete=set(results) == expected, checked_at=time.time())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--archive-completed", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--verify-all", action="store_true",
                        help="Recheck every archived file and raw evaluation before final completion")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "runs").mkdir(exist_ok=True)
    plan = json.loads(PLAN.read_text())
    manifest_path = root / "archive-manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else dict(runs={})
    results = manifest["runs"]
    snapshots = []
    for host in sorted({job["host"] for job in plan["jobs"]}):
        snap = snapshot(host)
        snapshots.append(snap)
        meta = root / "hosts" / host
        meta.mkdir(parents=True, exist_ok=True)
        (meta / "snapshot.json").write_text(json.dumps(snap, indent=2))
        queue = snap["queue"]
        if queue is None: continue
        if queue["source_commit"] != TRAINING_SOURCE: raise ValueError("wrong training source")
        if not args.archive_completed: continue
        for job in queue["jobs"]:
            if job.get("host") != host: raise ValueError("queue contains a job assigned elsewhere")
            if job["state"] != "complete" or job["name"] in results: continue
            name = job["name"]
            destination = root / "runs" / name
            destination.mkdir(exist_ok=True)
            sync(f"{host}:{REMOTE}/runs/{name}/", destination)
            sync(f"{host}:{REMOTE}/proofs/{name}-runs.json", meta / f"{name}-proof.json")
            proof = json.loads((meta / f"{name}-proof.json").read_text())
            for relative, expected in proof["sha256"].items():
                if digest(destination / relative) != expected: raise ValueError(f"SHA mismatch {name}/{relative}")
            local_proof = verify(destination, job, TRAINING_SOURCE, False)
            if local_proof["sha256"] != proof["sha256"]: raise ValueError("local/remote proof disagreement")
            results[name] = dict(host=host, reused=False, training_source=TRAINING_SOURCE,
                                 steps=proof["steps"], updates=proof["updates"],
                                 sha256=proof["sha256"], record=proof["record"])
        for part in ("manifest.json", "queue.json", "registration.json"):
            sync(f"{host}:{REMOTE}/{part}", meta / part)
    if args.archive_completed:
        historical_manifest = json.loads((args.historical_root / "archive-manifest.json").read_text())
        for reuse in plan["reused"]:
            name = reuse["name"]
            if name in results: continue
            source = (args.historical_root / "runs" / f"{reuse['task']}-optiq-s0").resolve()
            proof = verify_historical(source, reuse["source_commit"], reuse["task"], "optiq", 1_000_000, 1024)
            hashes = historical_manifest["runs"][reuse["task"]]["archived_files"]
            for relative, expected in hashes.items():
                # Old rsync temporary files are not scientific artifacts.
                if any(part.startswith(".") for part in Path(relative).parts): continue
                if digest(source / relative) != expected: raise ValueError(f"reused SHA mismatch: {relative}")
            destination = root / "runs" / name
            if not destination.exists(): destination.symlink_to(source, target_is_directory=True)
            config = json.loads((source / "config.json").read_text())
            if config["temperature"] != 1.: raise ValueError("reused T is not1")
            record = json.loads((source / "progress.json").read_text())["latest_evaluation"]
            results[name] = dict(reused=True, training_source=reuse["source_commit"],
                                 steps=1_000_000, updates=998976, num_envs=16, batch_size=256, utd=1.,
                                 path=str(source), sha256=hashes, record=record)
    for result in results.values():
        for mode in ("policy", "mu_only", "obstacle_policy"):
            if mode in result["record"]:
                row = result["record"][mode]
                row.setdefault("reachable_goals", sum(count > 0 for count in row["goals"]))
    ownership = audit_ownership(plan, snapshots)
    expected_names = {j["name"] for j in plan["jobs"] + plan["reused"]}
    if len(expected_names) != 33 or not set(results) <= expected_names:
        raise ValueError("result scope differs from requested 33 policies")
    if args.verify_all:
        audit = verify_archive(root, plan, results)
        audit["ownership"] = ownership
        audit["complete"] = audit["complete"] and not ownership["missing"]
        (root / "completion-audit.json").write_text(json.dumps(audit, indent=2))
    manifest.update(expected=33, complete=len(results), updated=time.time(),
                    ownership=ownership,
                    fresh_training_source=TRAINING_SOURCE,
                    note="8/16-Way T1 historical UTD1; new jobs UTD0.0625. Not a matched temperature ablation.")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    (root / "status.json").write_text(json.dumps(dict(updated=time.time(), hosts=snapshots), indent=2))
    with (root / "results.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["name", "mode", "reused", "steps", "updates", "success", "reachable_goals", "goal_counts", "source"])
        for name, result in sorted(results.items()):
            for mode in ("policy", "mu_only", "obstacle_policy"):
                if mode not in result["record"]: continue
                row = result["record"][mode]
                writer.writerow([name, mode, result["reused"], result["steps"], result["updates"],
                                 row["success"], row["reachable_goals"], row["goals"], result["training_source"]])
    if args.render:
        from .report_deadline import render
        render(root)
    print(json.dumps(dict(complete=len(results), expected=33,
                         states={s["host"]: None if s["queue"] is None else s["queue"]["state"] for s in snapshots})))


if __name__ == "__main__": main()
