"""Collect and verify finished PointMaze policies from both GPU hosts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

import numpy as np

from .report_pointmaze import (MAZES, METHODS, render_curves, render_medium_hard,
                               render_optiq_modes, render_trajectories)


HOSTS = ("vast-heechan-180", "vast-heechan-199")
REMOTE_ROOT = "/home/heechan/optiq-experiments/paper-pointmaze-seven-t3-20260926"


def sync(source: str, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-a", "--checksum", source, str(destination)], check=True)


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def verify_run(run: Path, job: dict, training_commit: str):
    config = json.loads((run / "config.json").read_text())
    progress = json.loads((run / "progress.json").read_text())
    if progress["status"] != "complete" or job["state"] != "complete":
        raise ValueError(f"incomplete job: {run.name}")
    if config["source_commit"] != training_commit:
        raise ValueError(f"source mismatch: {run.name}")
    if config["temperature"] != (3.0 if config["method"] == "optiq" else 1.0):
        raise ValueError(f"temperature profile mismatch: {run.name}")
    selected = job["selected"]
    if ((config["num_envs"], config["batch_size"], config["updates_per_collect"]) !=
            (selected["num_envs"], selected["batch_size"], selected["updates_per_collect"])):
        raise ValueError(f"selected collection profile mismatch: {run.name}")
    if ((progress["steps"], progress["updates"]) !=
            (job["result"]["steps"], job["result"]["updates"])):
        raise ValueError(f"transition or optimizer count mismatch: {run.name}")
    last = progress["latest_evaluation"]
    if last["step"] != progress["steps"]:
        raise ValueError(f"evaluation step mismatch: {run.name}")
    for mode in ("policy", "obstacle_policy"):
        path = run / "evaluations" / f"{last['step']:09d}_{mode}.npz"
        with np.load(path) as data:
            actual = data["goal_ids"]
            tracks = data["xy"]
            if (len(actual) != 500 or tracks.shape[0] != 500 or
                    tracks.shape[-1] != 2 or not np.isfinite(data["returns"]).all()):
                raise ValueError(f"invalid rollout: {path}")
            counts = np.bincount(actual[actual >= 0], minlength=len(last[mode]["goals"]))
            if counts.tolist() != last[mode]["goals"]:
                raise ValueError(f"goal ID mismatch: {path}")
    for filename, expected in job["result"]["checkpoint_sha256"].items():
        path = run / "checkpoints" / filename
        if digest(path) != expected:
            raise ValueError(f"checkpoint checksum mismatch: {path}")
    replay = run / "checkpoints" / f"replay_{last['step']:09d}.npz"
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
        meta.mkdir(parents=True, exist_ok=True)
        sync(f"{host}:{REMOTE_ROOT}/queue.json", meta / "queue.json")
        sync(f"{host}:{REMOTE_ROOT}/scheduler-manifest.json", meta / "scheduler-manifest.json")
        state = json.loads((meta / "queue.json").read_text())
        if state["source_commit"] != args.source_commit:
            raise ValueError(f"unexpected frozen source on {host}")
        for entry in state["jobs"]:
            if entry["state"] != "complete":
                continue
            name = entry["name"]
            sync(f"{host}:{REMOTE_ROOT}/jobs/{name}.json", meta / f"{name}.json")
            destination = root / "runs" / name
            if not destination.exists():
                destination.mkdir(parents=True)
            sync(f"{host}:{REMOTE_ROOT}/runs/{name}/", destination)
            job = json.loads((meta / f"{name}.json").read_text())
            results[name] = dict(host=host, sha256=verify_run(destination, job,
                                                              args.source_commit),
                                 selected=job["selected"], result=job["result"])
    reporting_source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True,
        cwd=Path(__file__).resolve().parents[1]).strip()
    manifest = dict(training_source_commit=args.source_commit,
                    reporting_source_commit=reporting_source,
                    completed=len(results), expected=len(MAZES) * len(METHODS),
                    runs=results)
    (root / "archive-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if results:
        figures = root / "figures"
        render_curves(root, figures / "learning_curves.png")
        for maze in MAZES:
            render_trajectories(root, maze, figures / f"trajectories_{maze}.png")
            render_optiq_modes(root, maze,
                               figures / f"optiq_{maze}_policy_vs_mu_only.png")
        render_medium_hard(root, figures / "trajectories_medium_hard.png")
    print(json.dumps(dict(completed=len(results), expected=21,
                          output=str(root)), indent=2))


if __name__ == "__main__":
    main()
