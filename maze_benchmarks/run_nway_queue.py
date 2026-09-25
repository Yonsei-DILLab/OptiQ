"""Sequential per-GPU N-Way queue with independent shards and no retries."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .run_nway_job import METHODS, atomic_json, verify_source
from .nway import SUPPORTED_GOALS


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--shard", required=True)
    parser.add_argument("--job", action="append", required=True,
                        help="task:method, for example 8way:optiq")
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    verify_source(source, args.source_commit)
    jobs = []
    for specification in args.job:
        task, sep, method = specification.partition(":")
        if not sep or task not in (f"{n}way" for n in SUPPORTED_GOALS) or method not in METHODS:
            raise ValueError(f"invalid queued task: {specification}")
        jobs.append((task, method))
    if len(set(jobs)) != len(jobs):
        raise ValueError("duplicate jobs in shard")
    args.root.mkdir(parents=True, exist_ok=True)
    status_path = args.root / f"queue-{args.shard}.json"
    if status_path.exists():
        raise FileExistsError(f"existing queue state must be inspected: {status_path}")
    status = dict(source_commit=args.source_commit, shard=args.shard,
                  jobs=[f"{task}:{method}" for task, method in jobs],
                  started=time.time(), pid=os.getpid(), completed=[], state="running")
    atomic_json(status_path, status)
    try:
        for task, method in jobs:
            status["active"] = f"{task}:{method}"
            status["updated"] = time.time()
            atomic_json(status_path, status)
            command = [sys.executable, "-m", "maze_benchmarks.run_nway_job",
                       "--task", task, "--method", method, "--root", str(args.root),
                       "--source-commit", args.source_commit]
            subprocess.run(command, cwd=source, check=True)
            status["completed"].append(f"{task}:{method}")
            status["updated"] = time.time()
            atomic_json(status_path, status)
        status["state"] = "complete"
        status.pop("active", None)
        status["finished"] = time.time()
        atomic_json(status_path, status)
    except BaseException as error:
        status["state"] = "failed"
        status["error"] = repr(error)
        status["failed"] = time.time()
        atomic_json(status_path, status)
        raise


if __name__ == "__main__":
    main()
