"""Validate a real 256-env update, then train one paper-map policy from scratch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .run_nway_job import METHODS, atomic_json, verify_source
from pointmaze.drac_paper import GOAL_COUNTS, MAP_NAMES


BUDGETS = {"simple": 100_000, "medium": 200_000, "hard": 300_000}
MAIN_WARMUP = 8192
PREFLIGHT_WARMUP = 4096
# Each collection uses 256 replay samples per new transition. The paper
# instead takes one batch-256 optimizer step per transition; this profile
# groups those sample accesses into larger batches to reduce GPU launch cost.
CANDIDATES = ((256, 4096), (128, 4096), (64, 4096),
              (64, 2048), (64, 1024), (64, 256))


def parameters(env_count: int, batch_size: int):
    if (256 * env_count) % batch_size:
        raise ValueError("nonintegral matched sample throughput")
    return (256 * env_count) // batch_size


def run_process(command: list[str], log_path: Path, source: Path):
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    with log_path.open("w") as log:
        result = subprocess.run(command, cwd=source, env=environment,
                                stdout=log, stderr=subprocess.STDOUT, check=False)
    return result.returncode


def command(source: Path, commit: str, maze: str, method: str,
            output: Path, env_count: int, batch_size: int, preflight: bool):
    warmup = PREFLIGHT_WARMUP if preflight else MAIN_WARMUP
    steps = warmup + env_count if preflight else math.ceil(BUDGETS[maze] / env_count) * env_count
    eval_every = steps if preflight else BUDGETS[maze] // 5
    episodes = 5 if preflight else 200
    final_episodes = 5 if preflight else 500
    return [sys.executable, "-m", "maze_benchmarks.run",
            "--task", f"pm_{maze}", "--method", method,
            "--output", str(output), "--seed", "0", "--steps", str(steps),
            "--num-envs", str(env_count),
            "--updates-per-collect", str(parameters(env_count, batch_size)),
            "--batch-size", str(batch_size), "--warmup", str(warmup),
            "--eval-every", str(eval_every), "--eval-episodes", str(episodes),
            "--final-eval-episodes", str(final_episodes),
            "--temperature", "1", "--render-each-eval",
            "--source-commit", commit]


def verify_run(output: Path, commit: str, maze: str, method: str,
               env_count: int, batch_size: int, preflight: bool):
    progress = json.loads((output / "progress.json").read_text())
    config = json.loads((output / "config.json").read_text())
    target = output / "evaluations"
    warmup = PREFLIGHT_WARMUP if preflight else MAIN_WARMUP
    steps = warmup + env_count if preflight else math.ceil(BUDGETS[maze] / env_count) * env_count
    updates = ((steps - warmup) // env_count) * parameters(env_count, batch_size)
    if progress["status"] != "complete" or (progress["steps"], progress["updates"]) != (steps, updates):
        raise ValueError("training step or update audit failed")
    if (config["source_commit"], config["task"], config["method"]) != (
            commit, f"pm_{maze}", method):
        raise ValueError("source/task/method mismatch")
    if (config["num_envs"], config["batch_size"], config["updates_per_collect"]) != (
            env_count, batch_size, parameters(env_count, batch_size)):
        raise ValueError("vector/update profile mismatch")
    record = progress["latest_evaluation"]
    expected_episodes = 5 if preflight else 500
    if record["step"] != steps:
        raise ValueError("final evaluation step mismatch")
    for mode in ("policy", "obstacle_policy"):
        summary = record[mode]
        if (summary["episodes"] != expected_episodes or
                len(summary["goals"]) != GOAL_COUNTS[maze] or
                sum(summary["goals"]) + summary["failure"] != expected_episodes):
            raise ValueError(f"goal accounting failed: {mode}")
        raw = target / f"{steps:09d}_{'obstacle_policy' if mode == 'obstacle_policy' else mode}.npz"
        with np.load(raw) as data:
            if (data["goal_ids"].shape != (expected_episodes,) or
                    not np.isfinite(data["returns"]).all()):
                raise ValueError(f"raw evaluation failed: {raw}")
    if method == "optiq" and not (target / f"{steps:09d}_mu_only.npz").is_file():
        raise FileNotFoundError("missing OptiQ mu-only comparison")
    figure = output / "figures" / f"{steps:09d}_trajectories.png"
    checkpoint = list((output / "checkpoints").glob(f"policy_{steps:09d}.*"))
    replay = output / "checkpoints" / f"replay_{steps:09d}.npz"
    if not figure.is_file() or not checkpoint or not replay.is_file():
        raise FileNotFoundError("missing figure, checkpoint or replay")
    return dict(steps=steps, updates=updates, policy=record["policy"],
                obstacle=record["obstacle_policy"],
                checkpoint_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in checkpoint})


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--maze", choices=MAP_NAMES, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    verify_source(source, args.source_commit)
    root = args.root.resolve()
    name = f"pm_{args.maze}-{args.method}-s0"
    job_path = root / "jobs" / f"{name}.json"
    job_path.parent.mkdir(parents=True, exist_ok=True)
    if job_path.exists() or (root / "runs" / name).exists():
        raise FileExistsError("preserved job already exists; no automatic retry")
    job = dict(name=name, method=args.method, maze=args.maze,
               source_commit=args.source_commit, pid=os.getpid(),
               started=time.time(), state="preflight", candidates=[])
    atomic_json(job_path, job)
    try:
        selected = None
        for env_count, batch_size in CANDIDATES:
            preflight = root / "preflights" / f"{name}-e{env_count}-b{batch_size}"
            preflight.parent.mkdir(exist_ok=True)
            log = root / "logs" / f"{name}-e{env_count}-b{batch_size}-preflight.log"
            code = run_process(command(source, args.source_commit, args.maze,
                                       args.method, preflight, env_count,
                                       batch_size, True), log, source)
            attempt = dict(num_envs=env_count, batch_size=batch_size,
                           updates_per_collect=parameters(env_count, batch_size),
                           exit_code=code, log=str(log))
            job["candidates"].append(attempt)
            atomic_json(job_path, job)
            if code == 0:
                attempt["proof"] = verify_run(preflight, args.source_commit,
                                              args.maze, args.method,
                                              env_count, batch_size, True)
                selected = env_count, batch_size
                atomic_json(job_path, job)
                break
            message = log.read_text(errors="replace").lower()
            if "out of memory" not in message and "cuda error: out of memory" not in message and code != -9:
                raise RuntimeError(f"non-memory preflight failure: {log}")
        if selected is None:
            raise MemoryError("all validated vector/batch settings exhausted")
        env_count, batch_size = selected
        job["state"] = "training"
        job["selected"] = dict(num_envs=env_count, batch_size=batch_size,
                               updates_per_collect=parameters(env_count, batch_size))
        atomic_json(job_path, job)
        run = root / "runs" / name
        code = run_process(command(source, args.source_commit, args.maze,
                                   args.method, run, env_count, batch_size, False),
                           root / "logs" / f"{name}-training.log", source)
        if code:
            raise RuntimeError(f"main training failed with exit code {code}")
        job["result"] = verify_run(run, args.source_commit, args.maze,
                                   args.method, env_count, batch_size, False)
        job["state"] = "complete"
        job["finished"] = time.time()
        atomic_json(job_path, job)
    except BaseException as error:
        job["state"] = "failed"
        job["error"] = repr(error)
        job["finished"] = time.time()
        atomic_json(job_path, job)
        raise


if __name__ == "__main__":
    main()
