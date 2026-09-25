"""Per-host, per-GPU queue with immediate backfill and no implicit retry."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .run_nway_job import METHODS, atomic_json, verify_source
from pointmaze.drac_paper import MAP_NAMES


def locked_status(root: Path, fn):
    with (root / "queue.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / "queue.json"
        status = json.loads(path.read_text())
        result = fn(status)
        atomic_json(path, status)
        fcntl.flock(lock, fcntl.LOCK_UN)
        return result


def claim(root: Path, worker: str):
    def update(status):
        if status["state"] != "running":
            return None
        for item in status["jobs"]:
            if item["state"] == "pending":
                item.update(state="running", worker=worker,
                            pid=os.getpid(), started=time.time())
                status["updated"] = time.time()
                return item["name"]
        return None
    return locked_status(root, update)


def finish(root: Path, name: str, successful: bool, error=None):
    def update(status):
        item = next(entry for entry in status["jobs"] if entry["name"] == name)
        item["state"] = "complete" if successful else "failed"
        item["finished"] = time.time()
        if error:
            item["error"] = error
            status["state"] = "paused"
            status["error"] = f"{name}: {error}"
        status["updated"] = time.time()
        if all(entry["state"] == "complete" for entry in status["jobs"]):
            status["state"] = "complete"
        return None
    locked_status(root, update)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--worker", type=str)
    parser.add_argument("--job", action="append", default=[])
    args = parser.parse_args()
    if args.initialize == bool(args.worker):
        raise ValueError("choose initialize or worker")
    source = Path(__file__).resolve().parents[1]
    verify_source(source, args.source_commit)
    root = args.root.resolve()
    if args.initialize:
        names = []
        for name in args.job:
            maze, sep, method = name.partition(":")
            if not sep or maze not in MAP_NAMES or method not in METHODS:
                raise ValueError(f"invalid queued job: {name}")
            names.append(f"pm_{maze}-{method}-s0")
        if not names or len(set(names)) != len(names):
            raise ValueError("empty or duplicate job list")
        root.mkdir(parents=True, exist_ok=False)
        (root / "logs").mkdir()
        state = dict(source_commit=args.source_commit, state="running",
                     started=time.time(), jobs=[dict(name=name, state="pending")
                                                   for name in names])
        atomic_json(root / "queue.json", state)
        print(json.dumps(state, indent=2))
        return
    if args.job:
        raise ValueError("jobs are accepted only during initialization")
    worker = args.worker
    while True:
        name = claim(root, worker)
        if name is None:
            return
        task, method, _seed = name.split("-")
        maze = task.removeprefix("pm_")
        command = [sys.executable, "-m", "maze_benchmarks.run_papermaze_job",
                   "--root", str(root), "--source-commit", args.source_commit,
                   "--maze", maze, "--method", method]
        result = subprocess.run(command, cwd=source, check=False)
        if result.returncode:
            finish(root, name, False, f"job exit code {result.returncode}")
            raise RuntimeError(f"{name} failed; pending queue paused")
        finish(root, name, True)


if __name__ == "__main__":
    main()
