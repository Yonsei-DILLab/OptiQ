"""Register the PointMaze Medium baseline grid behind existing GPU locks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from .run_nway_job import atomic_json, verify_source
from .run_pointmaze_baseline_grid import read_plan


SUPERVISOR_ROOT = Path("/home/heechan/OptiQ-ops/supervisor")
TEMPLATE = Path(__file__).with_name("supervisor_pointmaze_baseline_grid.conf.template")
GPU_SLOTS = {"vast-heechan-180": (0, 3),
             "vast-heechan-199": (0, 1, 2, 3)}


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--host", choices=tuple(GPU_SLOTS), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--phase", choices=("prepare", "start"), required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    verify_source(source, args.source_commit)
    plan, plan_sha = read_plan()
    root = args.root.resolve()
    services = [f"{plan['campaign']}-{args.host}-gpu{gpu}"
                for gpu in GPU_SLOTS[args.host]]
    socket = f"unix://{SUPERVISOR_ROOT / 'supervisor.sock'}"
    manifest_path = root / "scheduler-manifest.json"
    if args.phase == "prepare":
        if root.exists():
            raise FileExistsError(f"grid queue already exists: {root}")
        subprocess.run([sys.executable, "-m", "maze_benchmarks.run_pointmaze_baseline_grid",
                        "--root", str(root), "--source-commit", args.source_commit,
                        "--host", args.host, "--initialize"], cwd=source, check=True)
        template = TEMPLATE.read_text()
        for gpu, service in zip(GPU_SLOTS[args.host], services):
            path = SUPERVISOR_ROOT / "jobs" / f"{service}.conf"
            if path.exists():
                raise FileExistsError(f"supervisor service already exists: {path}")
            config = (template.replace("__SERVICE__", service)
                      .replace("__SOURCE__", str(source))
                      .replace("__PYTHON__", sys.executable)
                      .replace("__ROOT__", str(root))
                      .replace("__SHA__", args.source_commit)
                      .replace("__HOST__", args.host)
                      .replace("__GPU__", str(gpu)))
            path.write_text(config)
        atomic_json(manifest_path, dict(campaign=plan["campaign"],
                    host=args.host, source_commit=args.source_commit,
                    plan_sha256=plan_sha, root=str(root), source=str(source),
                    jobs=[job["name"] for job in plan["shards"][args.host]],
                    services=services, allowed_gpu_slots=GPU_SLOTS[args.host],
                    pointmaze_long_source="d710cc668a61e69b52f0b71bbe724d6a85c499fb",
                    reserved_gpu2_on_180=True,
                    unavailable_gpu1_on_180=True, prepared=time.time(), started=False))
        print(json.dumps(dict(prepared=len(plan["shards"][args.host]),
                              services=services, root=str(root))))
        return
    manifest = json.loads(manifest_path.read_text())
    if (manifest["host"] != args.host or manifest["source_commit"] != args.source_commit or
            manifest["plan_sha256"] != plan_sha or manifest["services"] != services or
            manifest["started"]):
        raise ValueError("registered grid manifest mismatched or already started")
    for operation in ("reread", "update"):
        subprocess.run(["sudo", "supervisorctl", "-s", socket, operation], check=True)
    for service in services:
        subprocess.run(["sudo", "supervisorctl", "-s", socket, "start", service], check=True)
    manifest["started"] = time.time()
    atomic_json(manifest_path, manifest)
    print(json.dumps(dict(started=len(services), root=str(root))))


if __name__ == "__main__":
    main()
