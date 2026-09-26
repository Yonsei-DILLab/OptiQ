"""Archive six temperature runs and compare preserved T3 with mu-only T5/T10."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .collect_deadline import digest, sync, snapshot
from .deadline_queue import verify
from .run_nway_job import atomic_json
from .visualize_pointmaze import plot_map, plot_rollouts
from .plot_style import LEARNING_LINEWIDTH

SOURCE = "bfb06944705685a7dea84558aba40500c22349b9"
PLAN = Path(__file__).with_name("POINTMAZE_T5_T10_PLAN.json")
REMOTE = "/home/heechan/optiq-experiments/pointmaze-optiq-t5-t10-1m-20260926"


def render(root, controls, results):
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(13, 13), constrained_layout=True)
    curves, curve_axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    rows = []
    for row, maze in enumerate(("simple", "medium", "hard")):
        for col, temperature in enumerate((3, 5, 10)):
            name = f"pm_{maze}-optiq-s0" if temperature == 3 else f"pm_{maze}-optiq-t{temperature}-s0"
            folder = (controls if temperature == 3 else root) / "runs" / name
            paths = sorted((folder / "evaluations").glob("*_summary.json"))
            ax = axes[row, col]
            plot_map(ax, maze)
            if temperature != 3 and name not in results:
                paths = []
            if not paths:
                ax.set_title(f"{maze.title()} T={temperature} · pending")
                continue
            config = json.loads((folder / "config.json").read_text())
            if config["temperature"] != temperature:
                raise ValueError("temperature/config mismatch")
            records = [json.loads(p.read_text()) for p in paths]
            final = records[-1]
            data = final["mu_only"]
            raw = folder / "evaluations" / f"{final['step']:09d}_mu_only.npz"
            actual = plot_rollouts(ax, raw)
            counts = actual["goals"] + [0] * (len(data["goals"]) - len(actual["goals"]))
            if counts != data["goals"] or actual["episodes"] != data["episodes"]:
                raise ValueError("raw/summary mismatch")
            ax.set_title(f"{maze.title()} T={temperature} · {final['step']:,}\n"
                         f"success {data['success']:.1%} · {sum(c > 0 for c in counts)}/{len(counts)} goals")
            ax.set_xlabel(f"Counts {counts}\nFailures {data['failure']}", fontsize=9)
            for metric, key in enumerate(("success", "reachable_goals")):
                curve_axes[metric, row].plot([r["step"] / 1000 for r in records],
                    [r["mu_only"][key] for r in records], label=f"T={temperature}",
                    marker="o", linewidth=LEARNING_LINEWIDTH)
            rows.append(dict(run=name, temperature=temperature, maze=maze,
                             source=config["source_commit"], mode="mu_only", **data))
        for metric in range(2):
            curve_axes[metric, row].set(title=maze.title(), xlabel="Environment transitions (k)",
                ylabel="Success rate" if metric == 0 else "Goals reached", xlim=(0, 1001))
            curve_axes[metric, row].grid(alpha=.2)
            if curve_axes[metric, row].lines: curve_axes[metric, row].legend()
    fig.suptitle("OptiQ PointMaze · fresh random z at every action · mu-only · seed0\n"
                 "All500 final evaluation trajectories shown; counts include failures")
    fig.savefig(figures / "trajectories_mu_only.png", dpi=170, bbox_inches="tight")
    curves.suptitle("OptiQ random-z mu-only · T3 reused; T5/T10 fresh · seed0")
    curves.savefig(figures / "learning_curves_mu_only.png", dpi=160, bbox_inches="tight")
    plt.close(fig); plt.close(curves)
    atomic_json(root / "results.json", dict(rows=rows,
        reporting_source=subprocess.check_output(["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1], text=True).strip(),
        missing_t3_obstacle_mu=not all((controls / "posthoc_mu" / f"pm_{maze}-optiq-s0" / "proof.json").exists() for maze in ("simple", "medium", "hard")), single_seed=True))


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--archive-completed", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=True)
    plan = json.loads(PLAN.read_text())
    expected = {j["name"]: j for j in plan["jobs"]}
    manifest_path = root / "archive-manifest.json"
    results = json.loads(manifest_path.read_text())["runs"] if manifest_path.exists() else {}
    owners = set()
    for host in sorted({j["host"] for j in plan["jobs"]}):
        snap = snapshot(host, remote=REMOTE)
        meta = root / "hosts" / host; meta.mkdir(parents=True, exist_ok=True)
        atomic_json(meta / "snapshot.json", snap)
        queue = snap["queue"]
        if queue is None: continue
        if queue["source_commit"] != SOURCE: raise ValueError("wrong source")
        for job in queue["jobs"]:
            name = job["name"]
            if name not in expected or name in owners or any(job.get(k) != v for k, v in expected[name].items()):
                raise ValueError("unexpected/duplicate/modified job")
            owners.add(name)
            if not args.archive_completed or job["state"] != "complete" or name in results:
                continue
            destination = root / "runs" / name; destination.mkdir(parents=True, exist_ok=True)
            sync(f"{host}:{REMOTE}/runs/{name}/", destination)
            sync(f"{host}:{REMOTE}/proofs/{name}-runs.json", meta / f"{name}-proof.json")
            proof = json.loads((meta / f"{name}-proof.json").read_text())
            for relative, sha in proof["sha256"].items():
                if digest(destination / relative) != sha: raise ValueError(f"SHA mismatch {name}/{relative}")
            checked = verify(destination, job, SOURCE, False)
            if checked != proof: raise ValueError("local/remote proof differs")
            results[name] = dict(host=host, training_source=SOURCE, **proof)
            atomic_json(manifest_path, dict(runs=results, complete=len(results), expected=6))
        for filename in ("manifest.json", "queue.json", "registration.json"):
            sync(f"{host}:{REMOTE}/{filename}", meta / filename)
    if args.archive_completed:
        for name, result in results.items():
            for relative, sha in result["sha256"].items():
                if digest(root / "runs" / name / relative) != sha:
                    raise ValueError("preserved local archive changed")
    atomic_json(manifest_path, dict(runs=results, complete=len(results), expected=6,
                                    checked_at=time.time(), missing_owners=sorted(set(expected)-owners)))
    if args.render: render(root, args.controls, results)
    print(json.dumps(dict(complete=len(results), expected=6, missing=sorted(set(expected)-set(results)))))


if __name__ == "__main__": main()
