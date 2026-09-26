"""Transfer only unclaimed deadline jobs, release first, with durable audit."""
from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
import time

TRAINING = "a55aaf13f7803b5cf8da7ba56f318675c7794171"
SOURCE = "vast-heechan-199"
TRANSFER_ID = "pending-eight-20260926"


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def release(root, plan, commit):
    names = {j["name"]: j for j in plan["jobs"]
             if j["method"] in ("sac", "td3") or j["name"] in
             ("pm_medium-meow-s0", "pm_simple-meow-s0")}
    assert len(names) == 8 and all(j["host"] != SOURCE for j in names.values())
    audit_path = root / "transfer-out-199.json"
    with (root / "queue.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        queue = json.loads((root / "queue.json").read_text())
        if queue["source_commit"] != TRAINING or queue["state"] != "running":
            raise ValueError("source queue not eligible")
        found = {j["name"]: j for j in queue["jobs"] if j["name"] in names}
        if set(found) != set(names): raise ValueError("source inventory mismatch")
        for name, job in found.items():
            if (job["state"] == "transferred" and
                    job.get("transfer_id") == TRANSFER_ID): continue
            if job["state"] != "pending": raise ValueError(f"job already claimed: {name}")
            if any((root / part / name).exists() for part in ("preflights", "runs")):
                raise ValueError(f"preserved attempt exists: {name}")
            for key in ("task", "method", "temperature"):
                if job[key] != names[name][key]: raise ValueError("learning setting changed")
        audit = dict(id=TRANSFER_ID, source=SOURCE, training_source=TRAINING,
                     scheduling_source=commit, jobs=list(names.values()),
                     phase="prepared", released_at=time.time())
        if audit_path.exists():
            previous = json.loads(audit_path.read_text())
            if any(previous[k] != audit[k] for k in
                   ("id", "source", "training_source", "scheduling_source", "jobs")):
                raise ValueError("conflicting preserved transfer")
            audit = previous
        write(audit_path, audit)
        for name, job in found.items():
            job.update(state="transferred", transfer_id=TRANSFER_ID,
                       destination=names[name]["host"], scheduling_source=commit)
        queue["updated"] = time.time()
        write(root / "queue.json", queue)
        audit["phase"] = "released"
        write(audit_path, audit)
    return audit


def accept(root, transfer, host):
    if transfer["id"] != TRANSFER_ID or transfer["phase"] != "released":
        raise ValueError("source has not released jobs")
    incoming = [j for j in transfer["jobs"] if j["host"] == host]
    if not incoming: raise ValueError("no jobs for destination")
    with (root / "queue.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        queue = json.loads((root / "queue.json").read_text())
        if queue["source_commit"] != TRAINING or queue["state"] != "running":
            raise ValueError("destination queue not eligible")
        old = {j["name"]: j for j in queue["jobs"]}
        for job in incoming:
            name = job["name"]
            if name in old:
                if old[name].get("transfer_id") != TRANSFER_ID:
                    raise ValueError(f"duplicate destination job: {name}")
                if any(old[name][k] != v for k, v in job.items()):
                    raise ValueError("existing transferred configuration differs")
                continue
            if any((root / part / name).exists() for part in ("preflights", "runs")):
                raise ValueError(f"destination attempt exists: {name}")
            queue["jobs"].append(dict(job, state="pending", transfer_id=TRANSFER_ID,
                                      transferred_from=SOURCE,
                                      scheduling_source=transfer["scheduling_source"]))
        write(root / "transfer-in-199.json", dict(transfer, destination=host,
                                                  accepted_at=time.time()))
        queue["updated"] = time.time()
        write(root / "queue.json", queue)
    return incoming


def finalize(root):
    with (root / "queue.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        queue = json.loads((root / "queue.json").read_text())
        if queue["source_commit"] != TRAINING: raise ValueError("wrong training source")
        if not all(j["state"] in ("complete", "transferred") for j in queue["jobs"]):
            raise ValueError("queue still has incomplete owned jobs")
        for job in queue["jobs"]:
            if job["state"] == "complete" and not (root / "proofs" / (job["name"] + "-runs.json")).is_file():
                raise ValueError("completed job lacks validation proof")
        queue.update(state="complete", updated=time.time(),
                     completion_scope="owned jobs; transferred jobs tracked at destinations")
        write(root / "queue.json", queue)


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--phase", choices=("release", "accept", "finalize"), required=True)
    p.add_argument("--scheduling-commit")
    p.add_argument("--host")
    p.add_argument("--transfer", type=Path)
    args = p.parse_args()
    if args.phase == "release":
        if not args.scheduling_commit or len(args.scheduling_commit) != 40:
            raise ValueError("full committed scheduling SHA required")
        plan = json.loads(Path(__file__).with_name("DEADLINE_1M_PLAN.json").read_text())
        result = release(args.root, plan, args.scheduling_commit)
    elif args.phase == "accept":
        result = accept(args.root, json.loads(args.transfer.read_text()), args.host)
    else:
        result = finalize(args.root)
    print(json.dumps(result))


if __name__ == "__main__": main()
