"""Archive and verify the longer fresh PointMaze campaign without touching learners."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import zipfile

import numpy as np

from .report_pointmaze import (MAZES, METHODS, render_curves,
                               render_medium_hard, render_optiq_modes,
                               render_trajectories)


CAMPAIGN = "paper-pointmaze-seven-long-1m2m3m-s0-20260926"
BUDGETS = {"simple": 1_000_000, "medium": 2_000_000, "hard": 3_000_000}
HOSTS = ("vast-heechan-180", "vast-heechan-199")
REMOTE_ROOT = f"/home/heechan/optiq-experiments/{CAMPAIGN}"


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def sync(source: str, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-a", "--checksum", source, str(destination)], check=True)


def verify_run(run: Path, job: dict, commit: str) -> dict:
    config = json.loads((run / "config.json").read_text())
    progress = json.loads((run / "progress.json").read_text())
    maze, method = job["maze"], job["method"]
    steps = math.ceil(BUDGETS[maze] / 256) * 256
    updates = ((steps - 8192) // 256) * 16
    if (job["state"] != "complete" or progress["status"] != "complete" or
            progress["steps"] != steps or progress["updates"] != updates or
            job["result"]["steps"] != steps or job["result"]["updates"] != updates):
        raise ValueError(f"step/update mismatch: {run.name}")
    if (config["source_commit"] != commit or config["task"] != f"pm_{maze}" or
            config["method"] != method or config["seed"] != 0 or
            config["steps"] != steps or config["num_envs"] != 256 or
            config["batch_size"] != 4096 or config["updates_per_collect"] != 16 or
            config["warmup"] != 8192 or config["final_eval_episodes"] != 2000 or
            config["temperature"] != (3.0 if method == "optiq" else 1.0)):
        raise ValueError(f"config mismatch: {run.name}")
    if method == "optiq" and config["agent"]["dacer"]["enabled"] is not False:
        raise ValueError(f"DACER enabled unexpectedly: {run.name}")
    last = progress["latest_evaluation"]
    if last["step"] != steps:
        raise ValueError(f"final evaluation mismatch: {run.name}")
    for mode in (("policy", "obstacle_policy", "mu_only") if method == "optiq"
                 else ("policy", "obstacle_policy")):
        path = run / "evaluations" / f"{steps:09d}_{mode}.npz"
        with np.load(path) as data:
            goals = data["goal_ids"]
            if (len(goals) != 2000 or data["xy"].shape[0] != 2000 or
                    not np.isfinite(data["returns"]).all()):
                raise ValueError(f"invalid raw evaluation: {path}")
            actual = np.bincount(goals[goals >= 0], minlength=len(last[mode]["goals"]))
            if actual.tolist() != last[mode]["goals"]:
                raise ValueError(f"goal ID mismatch: {path}")
    for filename, expected in job["result"]["checkpoint_sha256"].items():
        path = run / "checkpoints" / filename
        if digest(path) != expected:
            raise ValueError(f"checkpoint hash mismatch: {path}")
    replay = run / "checkpoints" / f"replay_{steps:09d}.npz"
    with zipfile.ZipFile(replay) as archive:
        if archive.testzip() is not None:
            raise ValueError(f"corrupt replay: {replay}")
    return {str(path.relative_to(run)): digest(path) for path in sorted(run.rglob("*"))
            if path.is_file()}


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    results = {}
    for host in HOSTS:
        meta = root / "hosts" / host
        sync(f"{host}:{REMOTE_ROOT}/queue.json", meta / "queue.json")
        sync(f"{host}:{REMOTE_ROOT}/scheduler-manifest.json", meta / "scheduler-manifest.json")
        status = json.loads((meta / "queue.json").read_text())
        scheduler = json.loads((meta / "scheduler-manifest.json").read_text())
        if (status["source_commit"] != args.source_commit or
                status["budget_multiplier"] != 10 or
                status["final_eval_episodes"] != 2000 or
                scheduler["source_commit"] != args.source_commit):
            raise ValueError(f"unexpected source/profile on {host}")
        for entry in status["jobs"]:
            if entry["state"] != "complete":
                continue
            name = entry["name"]
            if name in results:
                raise ValueError(f"duplicate completed job: {name}")
            sync(f"{host}:{REMOTE_ROOT}/jobs/{name}.json", meta / f"{name}.json")
            destination = root / "runs" / name
            destination.mkdir(parents=True, exist_ok=True)
            sync(f"{host}:{REMOTE_ROOT}/runs/{name}/", destination)
            job = json.loads((meta / f"{name}.json").read_text())
            results[name] = dict(host=host, sha256=verify_run(destination, job,
                                                              args.source_commit),
                                 selected=job["selected"], result=job["result"])
    reporting_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True,
        cwd=Path(__file__).resolve().parents[1]).strip()
    manifest = dict(training_source_commit=args.source_commit,
                    reporting_source_commit=reporting_commit,
                    completed=len(results), expected=len(MAZES) * len(METHODS),
                    runs=results)
    (root / "archive-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if len(results) == len(MAZES) * len(METHODS):
        figures = root / "figures"
        render_curves(root, figures / "learning_curves.png")
        for maze in MAZES:
            render_trajectories(root, maze, figures / f"trajectories_{maze}.png")
            render_optiq_modes(root, maze,
                               figures / f"optiq_{maze}_policy_vs_mu_only.png")
        render_medium_hard(root, figures / "trajectories_medium_hard.png")
    print(json.dumps(dict(completed=len(results), expected=21, output=str(root))))


if __name__ == "__main__":
    main()
