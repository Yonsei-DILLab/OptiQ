"""One MEOW entropy ablation on the original wall-free 4-Way task.

The native alpha=.2 run is retained as the baseline. Every candidate gets an
independent seed-0 preflight and fresh 100k training run; failed jobs are not
resumed or retried automatically.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .run_nway_job import atomic_json, verify_source


WARMUP = 1024
STEPS = 100_000
ALPHAS = (2., 10., 30.)


def audit(output: Path, commit: str, alpha: float, steps: int, episodes: int) -> dict:
    config = json.loads((output / "config.json").read_text())
    progress = json.loads((output / "progress.json").read_text())
    if (config["source_commit"], config["method"], config["task"],
            config["meow_alpha"], config["agent"]["alpha"]) != (
            commit, "meow", "4way", alpha, alpha):
        raise ValueError("MEOW run used a different source, task, or alpha")
    if (config["num_envs"], config["batch_size"], config["utd"],
            config["warmup"]) != (16, 256, 1., WARMUP):
        raise ValueError("unexpected 4-Way data/update profile")
    if (progress["status"], progress["steps"], progress["updates"]) != (
            "complete", steps, steps - WARMUP):
        raise ValueError("incomplete MEOW training")
    record = progress["latest_evaluation"]
    if record["step"] != steps or record["source_commit"] != commit:
        raise ValueError("evaluation source/step mismatch")
    raw = output / "evaluations" / f"{steps:09d}_policy.npz"
    probe = output / "evaluations" / f"{steps:09d}_probe.npz"
    if not raw.is_file() or not probe.is_file():
        raise FileNotFoundError("missing policy trajectories or Q/action probe")
    with np.load(raw) as data:
        xy = data["xy"]
        goals = data["goal_ids"]
        lengths = data["lengths"]
        returns = data["returns"]
    if (xy.shape != (episodes, 21, 2) or goals.shape != (episodes,) or
            not np.isfinite(returns).all()):
        raise ValueError("malformed 4-Way raw rollouts")
    counts = np.bincount(goals[goals >= 0], minlength=4).tolist()
    if counts != record["policy"]["goals"] or sum(counts) + int((goals < 0).sum()) != episodes:
        raise ValueError("goal counts differ from raw trajectories")
    if not list((output / "checkpoints").glob(f"policy_{steps:09d}.*")):
        raise FileNotFoundError("missing policy checkpoint")
    if steps == STEPS and not (output / "checkpoints" / f"replay_{steps:09d}.npz").is_file():
        raise FileNotFoundError("missing full replay")
    first = xy[:, 1] - xy[:, 0]
    return dict(alpha=alpha, source_commit=commit, steps=steps,
                success=float(np.mean(goals >= 0)), goals=counts,
                failures=int((goals < 0).sum()), mean_length=float(lengths.mean()),
                mean_return=float(returns.mean()),
                first_action_mean=first.mean(axis=0).tolist(),
                first_action_saturation=float(np.mean(np.abs(first) >= .999)),
                first_x_saturation=float(np.mean(np.abs(first[:, 0]) >= .999)))


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--alpha", type=float, choices=ALPHAS, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    verify_source(source, args.source_commit)
    name = f"meow-alpha{args.alpha:g}-s0"
    job_path = args.root / "jobs" / f"{name}.json"
    job_path.parent.mkdir(parents=True, exist_ok=True)
    if (job_path.exists() or (args.root / "preflights" / name).exists() or
            (args.root / "runs" / name).exists()):
        raise FileExistsError("MEOW alpha job already started; preserve and inspect")
    job = dict(task="4way", method="meow", alpha=args.alpha,
               source_commit=args.source_commit, pid=os.getpid(),
               started=time.time(), status="preflight")
    atomic_json(job_path, job)
    try:
        for phase, steps, episodes in (("preflights", 1040, 8),
                                       ("runs", STEPS, 100)):
            output = args.root / phase / name
            output.parent.mkdir(exist_ok=True)
            command = [sys.executable, "-m", "maze_benchmarks.run",
                       "--task", "4way", "--method", "meow", "--output", str(output),
                       "--seed", "0", "--steps", str(steps), "--num-envs", "16",
                       "--updates-per-collect", "16", "--batch-size", "256",
                       "--warmup", str(WARMUP), "--eval-every",
                       str(1040 if phase == "preflights" else 20_000),
                       "--eval-episodes", str(episodes), "--temperature", "1",
                       "--meow-alpha", str(args.alpha), "--source-commit", args.source_commit]
            job.update(status=phase, command=command, updated=time.time())
            atomic_json(job_path, job)
            subprocess.run(command, cwd=source, check=True)
            metrics = audit(output, args.source_commit, args.alpha, steps, episodes)
            job.update(status=f"{phase}_complete", metrics=metrics, updated=time.time())
            atomic_json(job_path, job)
        job.update(status="complete", completed=time.time())
        atomic_json(job_path, job)
    except BaseException as error:
        job.update(status="failed", error=repr(error), failed=time.time())
        atomic_json(job_path, job)
        raise


if __name__ == "__main__":
    main()
