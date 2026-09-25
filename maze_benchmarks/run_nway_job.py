"""Run one frozen N-Way job: real-update preflight, then fresh 1M training.

The enclosing supervisor service owns a GPU flock across both phases. Existing
jobs are never restarted automatically; all failures and partial files remain.
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

import numpy as np

from .nway import SUPPORTED_GOALS


METHODS = ("optiq", "sac", "sql", "meow", "mfpo", "dipo", "td3")
WARMUP = 1024
STEPS = 1_000_000


def atomic_json(path: Path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def verify_source(source: Path, commit: str):
    manifest = json.loads((source / "maze_source_manifest.json").read_text())
    if manifest["source_commit"] != commit:
        raise ValueError("frozen source commit mismatch")
    for relative, digest in manifest["sha256"].items():
        path = source / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"frozen source checksum mismatch: {relative}")


def verify_run(output: Path, commit: str, task: str, method: str,
               steps: int, episodes: int):
    progress = json.loads((output / "progress.json").read_text())
    config = json.loads((output / "config.json").read_text())
    record = progress["latest_evaluation"]
    goal_count = int(task[:-3])
    if progress["status"] != "complete" or progress["steps"] != steps:
        raise ValueError("training did not finish its step budget")
    if progress["updates"] != steps - WARMUP:
        raise ValueError("unexpected actor/critic update count")
    if (config["source_commit"], config["task"], config["method"]) != (commit, task, method):
        raise ValueError("source, task or method mismatch")
    if (config["num_envs"], config["batch_size"], config["utd"]) != (16, 256, 1.):
        raise ValueError("data/update profile mismatch")
    if record["step"] != steps or record["source_commit"] != commit:
        raise ValueError("last evaluation does not match final checkpoint")
    summary = record["policy"]
    if (summary["episodes"] != episodes or len(summary["goals"]) != goal_count or
            sum(summary["goals"]) + summary["failure"] != episodes):
        raise ValueError("malformed goal/episode accounting")
    raw = output / "evaluations" / f"{steps:09d}_policy.npz"
    probe = output / "evaluations" / f"{steps:09d}_probe.npz"
    figure = output / "figures" / f"{steps:09d}_policy_q_trajectories.png"
    checkpoint = list((output / "checkpoints").glob(f"policy_{steps:09d}.*"))
    if not raw.is_file() or not probe.is_file() or not figure.is_file() or not checkpoint:
        raise FileNotFoundError("missing final rollout, probe, figure or checkpoint")
    with np.load(raw) as data:
        goals = data["goal_ids"]
        if (goals.shape != (episodes,) or
                np.bincount(goals[goals >= 0], minlength=goal_count).tolist() != summary["goals"] or
                not np.isfinite(data["returns"]).all()):
            raise ValueError("raw rollout differs from summary")
    with np.load(probe) as data:
        if not np.isfinite(data["q"]).all() or not np.isfinite(data["actions"]).all():
            raise ValueError("nonfinite Q or action probe")
    if steps == STEPS and not (output / "checkpoints" / f"replay_{steps:09d}.npz").is_file():
        raise FileNotFoundError("missing final replay")
    return summary


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--task", choices=[f"{n}way" for n in SUPPORTED_GOALS], required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    verify_source(source, args.source_commit)
    args.root.mkdir(parents=True, exist_ok=True)
    name = f"{args.task}-{args.method}-s0"
    job_path = args.root / "jobs" / f"{name}.json"
    job_path.parent.mkdir(exist_ok=True)
    if job_path.exists() or (args.root / "preflights" / name).exists() or (args.root / "runs" / name).exists():
        raise FileExistsError("job already started; inspect preserved state before any retry")
    job = dict(task=args.task, method=args.method, source_commit=args.source_commit,
               pid=os.getpid(), started=time.time(), status="preflight")
    atomic_json(job_path, job)
    try:
        for phase, steps, episodes, final_episodes in (
            ("preflights", 1040, 8, 8),
            ("runs", STEPS, 256, 1024),
        ):
            target = args.root / phase / name
            target.parent.mkdir(exist_ok=True)
            command = [sys.executable, "-m", "maze_benchmarks.run",
                       "--task", args.task, "--method", args.method,
                       "--output", str(target), "--seed", "0", "--steps", str(steps),
                       "--num-envs", "16", "--updates-per-collect", "16",
                       "--batch-size", "256", "--warmup", str(WARMUP),
                       "--eval-every", str(1040 if phase == "preflights" else 100_000),
                       "--eval-episodes", str(episodes),
                       "--final-eval-episodes", str(final_episodes),
                       "--temperature", "1", "--render-each-eval",
                       "--source-commit", args.source_commit]
            environment = os.environ.copy()
            environment["MPLBACKEND"] = "Agg"
            job.update(status=phase, command=command, updated=time.time())
            atomic_json(job_path, job)
            subprocess.run(command, cwd=source, env=environment, check=True)
            summary = verify_run(target, args.source_commit, args.task, args.method,
                                 steps, final_episodes)
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
