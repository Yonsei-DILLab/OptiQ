"""Immutable, independently backfilled PointMaze Medium sensitivity jobs."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .run_nway_job import atomic_json, verify_source


PLAN_PATH = Path(__file__).with_name("POINTMAZE_MEDIUM_BASELINE_GRID.json")
PREFLIGHT_STEPS = 8192 + 256
GOAL_COUNT = 4
METHODS = ("optiq", "sac", "sql", "meow", "mfpo", "dipo", "td3")


def read_plan():
    data = PLAN_PATH.read_bytes()
    plan = json.loads(data)
    jobs = [job for shard in plan["shards"].values() for job in shard]
    if (len(jobs) != 10 or len({job["name"] for job in jobs}) != 10 or
            set(job["method"] for job in jobs) != set(METHODS) or
            any(job["updates_per_collect"] != 32 for job in jobs) or
            plan["num_envs"] != 256 or plan["batch_size"] != 4096 or
            plan["steps"] != math.ceil(2_000_000 / 256) * 256 or
            plan["warmup"] != 8192 or plan["final_eval_episodes"] != 2000):
        raise ValueError("PointMaze sensitivity plan has changed unexpectedly")
    for job in jobs:
        if (("sql_particles" in job and job["method"] != "sql") or
                ("meow_alpha" in job and job["method"] != "meow") or
                ("mfpo_target_entropy_per_dim" in job and job["method"] != "mfpo")):
            raise ValueError(f"method/override mismatch: {job['name']}")
    return plan, hashlib.sha256(data).hexdigest()


def with_lock(root: Path, operation):
    with (root / "queue.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / "queue.json"
        state = json.loads(path.read_text())
        result = operation(state)
        atomic_json(path, state)
        fcntl.flock(lock, fcntl.LOCK_UN)
        return result


def claim(root: Path, worker: str):
    def operation(state):
        if state["state"] != "running":
            return None
        for item in state["jobs"]:
            if item["state"] == "pending":
                item.update(state="running", worker=worker, pid=os.getpid(), started=time.time())
                state["updated"] = time.time()
                return item["name"]
        return None
    return with_lock(root, operation)


def finish(root: Path, name: str, success: bool, error: str | None = None):
    def operation(state):
        item = next(entry for entry in state["jobs"] if entry["name"] == name)
        item.update(state="complete" if success else "failed", finished=time.time())
        if error:
            item["error"] = error
            state.update(state="paused", error=f"{name}: {error}")
        state["updated"] = time.time()
        if all(entry["state"] == "complete" for entry in state["jobs"]):
            state["state"] = "complete"
    with_lock(root, operation)


def command(source: Path, commit: str, plan: dict, job: dict, output: Path,
            preflight: bool):
    steps = PREFLIGHT_STEPS if preflight else plan["steps"]
    cmd = [sys.executable, "-m", "maze_benchmarks.run", "--task", plan["task"],
           "--method", job["method"], "--output", str(output), "--seed", "0",
           "--steps", str(steps), "--num-envs", "256", "--batch-size", "4096",
           "--updates-per-collect", "32", "--warmup", "8192",
           "--eval-every", str(steps if preflight else plan["eval_every"]),
           "--eval-episodes", str(5 if preflight else plan["eval_episodes"]),
           "--final-eval-episodes", str(5 if preflight else plan["final_eval_episodes"]),
           "--temperature", "3" if job["method"] == "optiq" else "1",
           "--source-commit", commit]
    if "sql_particles" in job:
        cmd.extend(("--sql-particles", str(job["sql_particles"])))
    if "meow_alpha" in job:
        cmd.extend(("--meow-alpha", str(job["meow_alpha"])))
    if "mfpo_target_entropy_per_dim" in job:
        cmd.extend(("--mfpo-target-entropy-per-dim",
                    str(job["mfpo_target_entropy_per_dim"])))
    if not preflight:
        cmd.append("--render-each-eval")
    return cmd


def check_run(output: Path, commit: str, plan: dict, job: dict, preflight: bool):
    steps = PREFLIGHT_STEPS if preflight else plan["steps"]
    episodes = 5 if preflight else plan["final_eval_episodes"]
    updates = (steps - plan["warmup"]) // plan["num_envs"] * job["updates_per_collect"]
    progress = json.loads((output / "progress.json").read_text())
    config = json.loads((output / "config.json").read_text())
    if (progress["status"] != "complete" or progress["steps"] != steps or
            progress["updates"] != updates or config["steps"] != steps or
            config["method"] != job["method"] or config["task"] != plan["task"] or
            config["seed"] != 0 or config["source_commit"] != commit or
            config["num_envs"] != 256 or config["batch_size"] != 4096 or
            config["updates_per_collect"] != 32 or config["warmup"] != 8192 or
            config["final_eval_episodes"] != episodes or
            config["sql_particles"] != job.get("sql_particles", 16) or
            config["meow_alpha"] != job.get("meow_alpha", .2) or
            config["mfpo_target_entropy_per_dim"] !=
                job.get("mfpo_target_entropy_per_dim", -.5)):
        raise ValueError(f"training profile mismatch: {output}")
    if job["method"] == "optiq" and config["agent"]["dacer"]["enabled"] is not False:
        raise ValueError("OptiQ DACER must be disabled")
    latest = progress["latest_evaluation"]
    if latest["step"] != steps or latest["updates"] != updates:
        raise ValueError(f"missing final evaluation: {output}")
    for mode in (("policy", "mu_only", "obstacle_policy") if job["method"] == "optiq"
                 else ("policy", "obstacle_policy")):
        with np.load(output / "evaluations" / f"{steps:09d}_{mode}.npz") as raw:
            goals = raw["goal_ids"]
            observed = np.bincount(goals[goals >= 0], minlength=GOAL_COUNT)
            if (len(goals) != episodes or raw["xy"].shape[0] != episodes or
                    not np.isfinite(raw["returns"]).all() or
                    observed.tolist() != latest[mode]["goals"]):
                raise ValueError(f"invalid raw policy rollouts: {output} {mode}")
    if not list((output / "checkpoints").glob(f"policy_{steps:09d}.*")):
        raise FileNotFoundError(f"missing policy checkpoint: {output}")
    if not preflight and not (output / "checkpoints" / f"replay_{steps:09d}.npz").exists():
        raise FileNotFoundError(f"missing final replay: {output}")
    return dict(steps=steps, updates=updates, episodes=episodes,
                policy=latest["policy"])


def run_one(root: Path, source: Path, commit: str, plan: dict, job: dict):
    name = job["name"]
    record = root / "jobs" / f"{name}.json"
    preflight = root / "preflights" / name
    main = root / "runs" / name
    if any(path.exists() for path in (record, preflight, main)):
        raise FileExistsError(f"preserved grid job already exists: {name}")
    record.parent.mkdir(exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    status = dict(**job, source_commit=commit, state="preflight", started=time.time(),
                  pid=os.getpid())
    atomic_json(record, status)
    for target, probe in ((preflight, True), (main, False)):
        phase = "preflight" if probe else "training"
        target.parent.mkdir(parents=True, exist_ok=True)
        status["state"] = phase
        status["command"] = command(source, commit, plan, job, target, probe)
        atomic_json(record, status)
        with (root / "logs" / f"{name}-{phase}.log").open("w") as log:
            result = subprocess.run(status["command"], cwd=source, stdout=log,
                                    stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            raise RuntimeError(f"{phase} exit code {result.returncode}")
        status[f"{phase}_result"] = check_run(target, commit, plan, job, probe)
        atomic_json(record, status)
    status.update(state="complete", finished=time.time())
    atomic_json(record, status)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--host", choices=("vast-heechan-180", "vast-heechan-199"), required=True)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--worker")
    args = parser.parse_args()
    if args.initialize == bool(args.worker):
        raise ValueError("choose initialize or worker")
    source = Path(__file__).resolve().parents[1]
    verify_source(source, args.source_commit)
    plan, plan_sha = read_plan()
    root = args.root.resolve()
    if args.initialize:
        if root.exists():
            raise FileExistsError(f"grid queue already registered: {root}")
        root.mkdir(parents=True)
        (root / "logs").mkdir()
        atomic_json(root / "queue.json", dict(
            campaign=plan["campaign"], host=args.host,
            source_commit=args.source_commit, plan_sha256=plan_sha, state="running",
            started=time.time(), jobs=[dict(name=job["name"], state="pending")
                                       for job in plan["shards"][args.host]]))
        print(json.dumps(dict(host=args.host, jobs=len(plan["shards"][args.host]))))
        return
    registered = json.loads((root / "queue.json").read_text())
    if (registered["source_commit"] != args.source_commit or
            registered["host"] != args.host or registered["plan_sha256"] != plan_sha):
        raise ValueError("registered grid profile differs from frozen source")
    jobs = {job["name"]: job for job in plan["shards"][args.host]}
    while True:
        name = claim(root, args.worker)
        if name is None:
            return
        try:
            run_one(root, source, args.source_commit, plan, jobs[name])
        except BaseException as error:
            record = root / "jobs" / f"{name}.json"
            if record.exists():
                state = json.loads(record.read_text())
                state.update(state="failed", error=repr(error), finished=time.time())
                atomic_json(record, state)
            finish(root, name, False, repr(error))
            raise
        finish(root, name, True)


if __name__ == "__main__":
    main()
