"""Archive verified Medium grid runs and render honest goal/trajectory summaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps, ImageDraw

from .run_pointmaze_baseline_grid import read_plan, check_run, METHODS


def digest(path: Path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def sync(remote: str, local: Path):
    local.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-a", "--checksum", remote, str(local)], check=True)


def render(root: Path, rows: list[dict]):
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    order = ([f"medium-{method}-u16" for method in METHODS] +
             [job["name"] for shard in read_plan()[0]["shards"].values() for job in shard])
    rows.sort(key=lambda x: order.index(x["name"]))
    labels = [x["name"].removeprefix("medium-") for x in rows]
    fig, axes = plt.subplots(2, 1, figsize=(max(13, 1.4 * len(rows)), 8),
                             constrained_layout=True)
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    for goal in range(4):
        values = np.array([row["goals"][goal] / 2000 for row in rows])
        axes[0].bar(x, values, bottom=bottom, label=f"Goal {goal + 1}")
        bottom += values
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Success fraction by goal")
    axes[0].legend(ncol=4, loc="upper right")
    axes[1].bar(x, [sum(count > 0 for count in row["goals"]) for row in rows],
                color="#437fa1")
    axes[1].set_ylim(0, 4.2)
    axes[1].set_yticks(range(5))
    axes[1].set_ylabel("Reached goals / 4")
    axes[1].set_xticks(x, labels, rotation=35, ha="right")
    fig.suptitle("PointMaze Medium · direct sampled policy · seed 0 · 2M transitions")
    fig.savefig(figures / "goal_counts.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for row in rows:
        evaluations = Path(row["run_path"]) / "evaluations"
        history = [json.loads(path.read_text()) for path in sorted(evaluations.glob("*_summary.json"))]
        ax.plot([item["step"] / 1e6 for item in history],
                [item["policy"]["success"] for item in history],
                marker="o", label=row["name"].removeprefix("medium-"))
    ax.set(xlabel="Environment transitions (millions)", ylabel="Direct-policy success",
           ylim=(0, 1.03), title="PointMaze Medium · learning curves · seed 0")
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(figures / "success_curves.png", dpi=180)
    plt.close(fig)

    thumbs = []
    for row in rows:
        file = Path(row["run_path"]) / "figures" / "002000128_trajectories.png"
        if not file.is_file():
            continue
        with Image.open(file) as image:
            thumb = ImageOps.contain(image.convert("RGB"), (450, 380))
        canvas = Image.new("RGB", (480, 430), "white")
        canvas.paste(thumb, ((480 - thumb.width) // 2, 35))
        ImageDraw.Draw(canvas).text((12, 10), row["name"], fill="black")
        thumbs.append(canvas)
    if thumbs:
        columns = 2
        plate = Image.new("RGB", (480 * columns, 430 * ((len(thumbs) + 1) // 2)),
                          "white")
        for index, thumb in enumerate(thumbs):
            plate.paste(thumb, (480 * (index % columns), 430 * (index // columns)))
        plate.save(figures / "final_trajectories.png")


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--control-root", type=Path,
                        help="verified completed 21-policy long campaign archive")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan, plan_sha = read_plan()
    expected = {job["name"]: job for shard in plan["shards"].values() for job in shard}
    rows = []
    manifest = dict(source_commit=args.source_commit, plan_sha256=plan_sha,
                    completed={}, expected=len(expected))
    for host in plan["shards"]:
        remote_root = f"/home/heechan/optiq-experiments/{plan['campaign']}"
        host_root = root / "hosts" / host
        sync(f"{host}:{remote_root}/queue.json", host_root / "queue.json")
        sync(f"{host}:{remote_root}/scheduler-manifest.json",
             host_root / "scheduler-manifest.json")
        queue = json.loads((host_root / "queue.json").read_text())
        scheduler = json.loads((host_root / "scheduler-manifest.json").read_text())
        if (queue["source_commit"] != args.source_commit or
                queue["plan_sha256"] != plan_sha or
                scheduler["source_commit"] != args.source_commit or
                scheduler["plan_sha256"] != plan_sha):
            raise ValueError(f"grid source/profile mismatch on {host}")
        for entry in queue["jobs"]:
            if entry["state"] != "complete":
                continue
            name = entry["name"]
            if name not in expected or name in manifest["completed"]:
                raise ValueError(f"unregistered or duplicate completed grid job: {name}")
            job_path = host_root / "jobs" / f"{name}.json"
            sync(f"{host}:{remote_root}/jobs/{name}.json", job_path)
            job = json.loads(job_path.read_text())
            if job["state"] != "complete" or job["source_commit"] != args.source_commit:
                raise ValueError(f"incomplete grid job record: {name}")
            run = root / "runs" / name
            run.mkdir(parents=True, exist_ok=True)
            sync(f"{host}:{remote_root}/runs/{name}/", run)
            verified = check_run(run, args.source_commit, plan, expected[name], False)
            with zipfile.ZipFile(run / "checkpoints" / f"replay_{plan['steps']:09d}.npz") as archive:
                if archive.testzip() is not None:
                    raise ValueError(f"corrupt replay archive: {name}")
            checksums = {str(path.relative_to(run)): digest(path)
                         for path in sorted(run.rglob("*")) if path.is_file()}
            manifest["completed"][name] = dict(host=host, sha256=checksums,
                                               verified=verified)
            rows.append(dict(name=name, method=job["method"],
                             updates_per_collect=job["updates_per_collect"],
                             sql_particles=job.get("sql_particles", 16),
                             meow_alpha=job.get("meow_alpha", .2),
                             mfpo_target_entropy_per_dim=job.get("mfpo_target_entropy_per_dim", -.5),
                             steps=verified["steps"], updates=verified["updates"],
                             success=verified["policy"]["success"],
                             goals=verified["policy"]["goals"],
                             run_path=str(run)))
    (root / "archive-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (root / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (root / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("name", "method", "updates_per_collect",
                                                  "sql_particles", "meow_alpha",
                                                  "mfpo_target_entropy_per_dim", "steps",
                                                  "updates", "success", "goals", "run_path"))
        writer.writeheader()
        writer.writerows(rows)
    if len(rows) == len(expected):
        comparison = list(rows)
        if args.control_root is not None:
            controls = args.control_root.resolve()
            for method in METHODS:
                run = controls / "runs" / f"pm_medium-{method}-s0"
                config = json.loads((run / "config.json").read_text())
                progress = json.loads((run / "progress.json").read_text())
                if (config["source_commit"] != "d710cc668a61e69b52f0b71bbe724d6a85c499fb" or
                        config["steps"] != plan["steps"] or config["batch_size"] != 4096 or
                        config["updates_per_collect"] != 16 or progress["status"] != "complete" or
                        progress["latest_evaluation"]["policy"]["episodes"] != 2000):
                    raise ValueError(f"incompatible Medium UTD16 control: {method}")
                comparison.append(dict(name=f"medium-{method}-u16", method=method,
                                       updates_per_collect=16, steps=progress["steps"],
                                       updates=progress["updates"],
                                       success=progress["latest_evaluation"]["policy"]["success"],
                                       goals=progress["latest_evaluation"]["policy"]["goals"],
                                       run_path=str(run)))
        (root / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
        render(root, comparison)
    print(json.dumps(dict(completed=len(rows), expected=len(expected), output=str(root))))


if __name__ == "__main__":
    main()
