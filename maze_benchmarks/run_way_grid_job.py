"""One frozen 4/8/12/16-Way job, with a real-update preflight and no retry."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .run_nway_job import METHODS, STEPS, WARMUP, atomic_json, verify_source


TASKS = ("4way", "8way", "12way", "16way")
TEMPERATURES = (1., 3., 5.)


def job_name(task: str, method: str, temperature: float) -> str:
    return f"{task}-{method}-t{temperature:g}-s0" if method == "optiq" else f"{task}-{method}-s0"


def verify_run(output: Path, commit: str, task: str, method: str,
               temperature: float, steps: int, episodes: int):
    progress = json.loads((output / "progress.json").read_text())
    config = json.loads((output / "config.json").read_text())
    record = progress["latest_evaluation"]
    if progress["status"] != "complete" or progress["steps"] != steps:
        raise ValueError("incomplete transition budget")
    if progress["updates"] != steps - WARMUP:
        raise ValueError("actor/critic update count differs from UTD=1")
    if (config["source_commit"], config["task"], config["method"]) != (commit, task, method):
        raise ValueError("source, task or method mismatch")
    if (config["num_envs"], config["batch_size"], config["updates_per_collect"],
            config["warmup"], config["seed"]) != (16, 256, 16, WARMUP, 0):
        raise ValueError("vector collection/update profile mismatch")
    if method == "optiq" and config["temperature"] != temperature:
        raise ValueError("OptiQ temperature mismatch")
    if record["step"] != steps or record["source_commit"] != commit:
        raise ValueError("final evaluation/provenance mismatch")
    summary = record["policy"]
    goal_count = int(task[:-3])
    if (summary["episodes"] != episodes or len(summary["goals"]) != goal_count or
            sum(summary["goals"]) + summary["failure"] != episodes):
        raise ValueError("invalid goal accounting")
    raw = output / "evaluations" / f"{steps:09d}_policy.npz"
    probe = output / "evaluations" / f"{steps:09d}_probe.npz"
    figure = output / "figures" / f"{steps:09d}_trajectories.png"
    checkpoint = list((output / "checkpoints").glob(f"policy_{steps:09d}.*"))
    if not raw.is_file() or not probe.is_file() or not figure.is_file() or not checkpoint:
        raise FileNotFoundError("final rollout, probe, figure or checkpoint missing")
    with np.load(raw) as data:
        goals = data["goal_ids"]
        if (goals.shape != (episodes,) or
                np.bincount(goals[goals >= 0], minlength=goal_count).tolist() != summary["goals"] or
                not np.isfinite(data["returns"]).all()):
            raise ValueError("raw rollouts and summary differ")
    with np.load(probe) as data:
        if not np.isfinite(data["q"]).all() or not np.isfinite(data["actions"]).all():
            raise ValueError("nonfinite Q/action probe")
    if steps == STEPS and not (output / "checkpoints" / f"replay_{steps:09d}.npz").is_file():
        raise FileNotFoundError("final replay missing")
    return summary


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--temperature", type=float, default=1.)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if args.method == "optiq" and args.temperature not in TEMPERATURES:
        raise ValueError("OptiQ temperature outside approved grid")
    if args.method != "optiq" and args.temperature != 1.:
        raise ValueError("baseline temperatures must retain native setting")
    source = Path(__file__).resolve().parents[1]
    verify_source(source, args.source_commit)
    name = job_name(args.task, args.method, args.temperature)
    job_path = args.root / "jobs" / f"{name}.json"
    job_path.parent.mkdir(parents=True, exist_ok=True)
    if job_path.exists() or any((args.root / phase / name).exists() for phase in ("preflights", "runs")):
        raise FileExistsError("job already started; preserved results require inspection")
    job = dict(task=args.task, method=args.method, temperature=args.temperature,
               source_commit=args.source_commit, pid=os.getpid(), started=time.time(), status="preflight")
    atomic_json(job_path, job)
    try:
        for phase, steps, episodes in (("preflights", 1040, 8), ("runs", STEPS, 256)):
            target = args.root / phase / name
            target.parent.mkdir(parents=True, exist_ok=True)
            final_episodes = episodes if phase == "preflights" else 1024
            command = [sys.executable, "-m", "maze_benchmarks.run",
                       "--task", args.task, "--method", args.method,
                       "--output", str(target), "--seed", "0", "--steps", str(steps),
                       "--num-envs", "16", "--updates-per-collect", "16",
                       "--batch-size", "256", "--warmup", str(WARMUP),
                       "--eval-every", str(1040 if phase == "preflights" else 100_000),
                       "--eval-episodes", str(episodes),
                       "--final-eval-episodes", str(final_episodes),
                       "--temperature", str(args.temperature), "--render-each-eval",
                       "--source-commit", args.source_commit]
            environment = os.environ.copy()
            environment["MPLBACKEND"] = "Agg"
            job.update(status=phase, command=command, updated=time.time())
            atomic_json(job_path, job)
            subprocess.run(command, cwd=source, env=environment, check=True)
            summary = verify_run(target, args.source_commit, args.task, args.method,
                                 args.temperature, steps, final_episodes)
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
