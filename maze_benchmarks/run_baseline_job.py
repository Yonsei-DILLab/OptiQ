"""Validate one frozen baseline on GPU, then run its independent 100k job.

The enclosing supervisor service holds the per-GPU flock while this process
performs both phases. An interrupted or failed job is preserved, never resumed
or retried automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


METHODS = ("sql", "meow", "mfpo", "dipo", "td3")


def atomic_json(path: Path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def verify_source(source: Path, commit: str):
    manifest = json.loads((source / "maze_source_manifest.json").read_text())
    if manifest["source_commit"] != commit:
        raise ValueError("training source commit differs from frozen manifest")
    for relative, digest in manifest["sha256"].items():
        path = source / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"frozen upstream source hash mismatch: {relative}")


def verify_run(output: Path, commit: str, steps: int, updates: int, episodes: int):
    progress = json.loads((output / "progress.json").read_text())
    config = json.loads((output / "config.json").read_text())
    latest = progress["latest_evaluation"]
    if progress["status"] != "complete" or progress["steps"] != steps or progress["updates"] != updates:
        raise ValueError(f"incomplete run: {output}: {progress}")
    if latest["source_commit"] != commit or config["source_commit"] != commit:
        raise ValueError("source commit mismatch in training outputs")
    if config["num_envs"] != 16 or config["batch_size"] != 256 or config["utd"] != 1.:
        raise ValueError("unexpected data/update profile")
    summary = latest["policy"]
    if summary["episodes"] != episodes or sum(summary["goals"]) + summary["failure"] != episodes:
        raise ValueError("missing evaluated episodes")
    if not (output / "evaluations" / f"{steps:09d}_policy.npz").is_file():
        raise FileNotFoundError("missing raw trajectories")
    if not (output / "evaluations" / f"{steps:09d}_probe.npz").is_file():
        raise FileNotFoundError("missing learned Q/action probe")
    if not list((output / "checkpoints").glob(f"policy_{steps:09d}.*")):
        raise FileNotFoundError("missing policy/critic checkpoint")
    return summary


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    verify_source(source, args.source_commit)
    args.root.mkdir(parents=True, exist_ok=True)
    job_path = args.root / "jobs" / f"{args.method}-s0.json"
    job_path.parent.mkdir(exist_ok=True)
    if job_path.exists() or (args.root / "preflights" / f"{args.method}-s0").exists() or (args.root / "runs" / f"{args.method}-s0").exists():
        raise FileExistsError("job already started; preserve its state and inspect before any retry")
    job = dict(method=args.method, task="4way", source_commit=args.source_commit,
               pid=os.getpid(), started=time.time(), status="preflight")
    atomic_json(job_path, job)
    try:
        for phase, steps, warmup, eval_every, episodes in (
            ("preflights", 1040, 1024, 1040, 8),
            ("runs", 100000, 1024, 20000, 100),
        ):
            target = args.root / phase / f"{args.method}-s0"
            target.parent.mkdir(exist_ok=True)
            command = [sys.executable, "-m", "maze_benchmarks.run",
                       "--task", "4way", "--method", args.method,
                       "--output", str(target), "--seed", "0",
                       "--steps", str(steps), "--num-envs", "16",
                       "--updates-per-collect", "16", "--batch-size", "256",
                       "--warmup", str(warmup), "--eval-every", str(eval_every),
                       "--eval-episodes", str(episodes), "--temperature", "1",
                       "--source-commit", args.source_commit]
            job.update(status=phase, command=command, updated=time.time())
            atomic_json(job_path, job)
            subprocess.run(command, cwd=source, check=True)
            summary = verify_run(target, args.source_commit, steps, steps - warmup, episodes)
            job.update(status=f"{phase}_complete", summary=summary, updated=time.time())
            atomic_json(job_path, job)
        job.update(status="complete", completed=time.time())
        atomic_json(job_path, job)
    except BaseException as error:
        job.update(status="failed", error=repr(error), failed=time.time())
        atomic_json(job_path, job)
        raise


if __name__ == "__main__":
    main()
