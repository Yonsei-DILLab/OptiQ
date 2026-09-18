"""Queue the shared no-anchor baseline; default benchmark MyoHand, seeds 0,1,2."""

import argparse
from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
import sys

from scripts.mujoco_beta_sweep import worker_lock

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (0, 1, 2)
BENCHMARKS = ("pen_twirl_hard", "ant", "humanoid", "reach_hard", "obj_hold_hard")


@dataclass(frozen=True)
class Task:
    benchmark: str
    seed: int


def tasks(benchmarks=("pen_twirl_hard",), seeds=DEFAULT_SEEDS):
    return [Task(benchmark, seed) for benchmark in benchmarks for seed in seeds]


def command(task, overrides=()):
    return [sys.executable, str(ROOT / "run_optiq_dime.py"),
            "--config-name=optiq_dime_no_anchor", f"benchmark={task.benchmark}",
            f"seed={task.seed}", *overrides]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--list", action="store_true", help="Print the queue without running")
    selection.add_argument("--task", type=int, help="Run one zero-based task ID")
    selection.add_argument("--worker", type=int, nargs=2, metavar=("INDEX", "COUNT"))
    parser.add_argument("--benchmarks", default="pen_twirl_hard", help="Comma-separated benchmark names")
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    args, overrides = parser.parse_known_args()
    benchmarks = args.benchmarks.split(",")
    if len(set(benchmarks)) != len(benchmarks) or any(b not in BENCHMARKS for b in benchmarks):
        parser.error(f"Choose distinct benchmarks from {BENCHMARKS}")
    try:
        seeds = tuple(int(s) for s in args.seeds.split(","))
        if min(seeds) < 0 or len(set(seeds)) != len(seeds):
            raise ValueError
    except ValueError:
        parser.error("--seeds must be distinct nonnegative integers")
    for override in overrides:
        if "=" not in override or override.split("=", 1)[0].lstrip("+~") in {"seed", "benchmark", "env_name", "task"}:
            parser.error(f"Use --seeds / --benchmarks for queue axes; invalid override: {override}")
    table = tasks(benchmarks, seeds)
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
    print(f"tasks={len(table)} selected={len(indices)} seeds={list(seeds)}", flush=True)
    if args.list:
        for i in indices:
            print(f"{i}: {shlex.join(command(table[i], overrides))}")
        return
    with worker_lock():
        for i in indices:
            cmd = command(table[i], overrides)
            print(f"task={i} {shlex.join(cmd)}", flush=True)
            subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
