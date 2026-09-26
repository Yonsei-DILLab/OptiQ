"""Independent per-GPU 4/8/12/16-Way queue; failed jobs stop their shard."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time

from .run_nway_job import METHODS, atomic_json, verify_source
from .run_way_grid_job import TASKS, TEMPERATURES, job_name


def parse_specification(value: str):
    parts = value.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"invalid job: {value}")
    task, method = parts[:2]
    temperature = float(parts[2]) if len(parts) == 3 else 1.
    if task not in TASKS or method not in METHODS:
        raise ValueError(f"invalid job: {value}")
    if method == "optiq" and temperature not in TEMPERATURES:
        raise ValueError(f"OptiQ temperature outside approved grid: {value}")
    if method != "optiq" and temperature != 1.:
        raise ValueError(f"baseline-specific temperature not approved: {value}")
    return task, method, temperature


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--shard", required=True)
    parser.add_argument("--job", action="append", required=True,
                        help="task:method[:temperature]")
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    verify_source(source, args.source_commit)
    jobs = [parse_specification(value) for value in args.job]
    names = [job_name(*job) for job in jobs]
    if len(set(names)) != len(names):
        raise ValueError("duplicate jobs in shard")
    args.root.mkdir(parents=True, exist_ok=True)
    status_path = args.root / f"queue-{args.shard}.json"
    if status_path.exists():
        raise FileExistsError("queue already started; inspect preserved state")
    status = dict(source_commit=args.source_commit, shard=args.shard,
                  jobs=names, started=time.time(), completed=[], state="running")
    atomic_json(status_path, status)
    try:
        for (task, method, temperature), name in zip(jobs, names):
            status.update(active=name, updated=time.time())
            atomic_json(status_path, status)
            command = [sys.executable, "-m", "maze_benchmarks.run_way_grid_job",
                       "--task", task, "--method", method,
                       "--temperature", str(temperature), "--root", str(args.root),
                       "--source-commit", args.source_commit]
            subprocess.run(command, cwd=source, check=True)
            status["completed"].append(name)
            status["updated"] = time.time()
            atomic_json(status_path, status)
        status["state"] = "complete"
        status.pop("active", None)
        status["finished"] = time.time()
        atomic_json(status_path, status)
    except BaseException as error:
        status.update(state="failed", error=repr(error), failed=time.time())
        atomic_json(status_path, status)
        raise


if __name__ == "__main__":
    main()
