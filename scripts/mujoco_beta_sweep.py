"""Deterministic task table for Ant/Humanoid sampling x fixed-beta sweeps."""

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BETAS = (0.1, 0.25, 0.5, 0.75, 1.0)
# Supplied run uses seed 1; requested four-seed sweep follows existing 1-based suites.
DEFAULT_SEEDS = (1, 2, 3, 4)


@dataclass(frozen=True)
class Task:
    environment: str
    sampling: str
    beta: float
    seed: int


def tasks(seeds=DEFAULT_SEEDS):
    return [Task(env, mode, beta, seed)
            for env in ("ant", "humanoid")
            for mode in ("stratified", "exact")
            for beta in BETAS for seed in seeds]


def command(task, overrides):
    return [sys.executable, str(ROOT / "run_optiq_dime.py"),
            "--config-name=optiq_dime_mujoco",
            f"mujoco_env={task.environment}", f"seed={task.seed}",
            f"alg.actor.proposal_sampling_mode={task.sampling}",
            f"alg.actor.density_beta={task.beta}",
            "alg.actor.adaptive_density_beta=false", *overrides]


@contextmanager
def worker_lock():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not gpu or "," in gpu:
        raise ValueError("Set CUDA_VISIBLE_DEVICES to one allocated GPU (e.g. 0)")
    lock_dir = ROOT / "outputs" / ".gpu-locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    # UUIDs and numeric indices are valid CUDA identifiers.
    with (lock_dir / f"{gpu.replace('/', '_')}.lock").open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"A sweep worker already holds GPU {gpu}") from exc
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--list", action="store_true", help="Print commands; no training")
    selection.add_argument("--task", type=int, help="Run one zero-based task ID")
    selection.add_argument("--worker", type=int, nargs=2, metavar=("INDEX", "COUNT"),
                           help="Run task IDs congruent to INDEX modulo COUNT")
    selection.add_argument("--all", action="store_true", help="Run all tasks sequentially")
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    args, overrides = parser.parse_known_args()
    try:
        seeds = [int(s) for s in args.seeds.split(",")]
        if not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds):
            raise ValueError
    except ValueError:
        parser.error("--seeds must contain distinct nonnegative integers")
    # Keep labels honest: table axes cannot be overridden by free-form Hydra args.
    protected = {"mujoco_env", "env_name", "seed", "alg.actor.proposal_sampling_mode",
                 "alg.actor.density_beta", "alg.actor.adaptive_density_beta"}
    for override in overrides:
        if "=" not in override or override.split("=", 1)[0].lstrip("+~") in protected:
            parser.error(f"Invalid or protected override: {override}")
    table = tasks(seeds)
    indices = list(range(len(table)))
    if args.task is not None:
        if not 0 <= args.task < len(table):
            parser.error(f"Task ID must be in 0-{len(table)-1}")
        indices = [args.task]
    if args.worker:
        index, count = args.worker
        if count < 1 or not 0 <= index < count:
            parser.error("Require 0 <= worker INDEX < COUNT")
        indices = indices[index::count]
    print(f"tasks={len(table)} selected={len(indices)} seeds={seeds}", flush=True)
    if args.list:
        for i in indices:
            print(f"{i}: {shlex.join(command(table[i], overrides))}")
        return
    with worker_lock():
        for i in indices:
            cmd = command(table[i], overrides)
            print(f"task={i} {shlex.join(cmd)}", flush=True)
            # Fail fast. Never mark an interrupted run complete or silently skip it.
            subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        # Allow inspecting --list output through tools such as head.
        os._exit(0)
