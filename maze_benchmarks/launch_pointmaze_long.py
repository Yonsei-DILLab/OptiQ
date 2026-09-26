"""Register and start the fresh 1M/2M/3M PointMaze queue on one 5090 host."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from .run_nway_job import atomic_json, verify_source


PLAN = "POINTMAZE_LONG_1_2_3M_PLAN.json"
TEMPLATE = "supervisor_pointmaze_long.conf.template"
SUPERVISOR_ROOT = Path("/home/heechan/OptiQ-ops/supervisor")
CAMPAIGN = "paper-pointmaze-seven-long-1m2m3m-s0-20260926"


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--host", choices=("vast-heechan-180", "vast-heechan-199"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--phase", choices=("prepare", "start"), required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    verify_source(source, args.source_commit)
    plan = json.loads((Path(__file__).parent / PLAN).read_text())
    if plan["campaign"] != CAMPAIGN or plan["optiq_dacer_enabled"] is not False:
        raise ValueError("unexpected PointMaze campaign plan")
    root = args.root.resolve()
    names = plan["shards"][args.host]
    services = [f"{CAMPAIGN}-{args.host}-gpu{index}" for index in range(4)]
    socket = f"unix://{SUPERVISOR_ROOT / 'supervisor.sock'}"
    if args.phase == "prepare":
        if root.exists():
            raise FileExistsError(f"fresh queue root already exists: {root}")
        command = [sys.executable, "-m", "maze_benchmarks.run_papermaze_queue",
                   "--root", str(root), "--source-commit", args.source_commit,
                   "--budget-multiplier", "10", "--final-eval-episodes", "2000",
                   "--initialize"]
        for name in names:
            command.extend(("--job", name))
        subprocess.run(command, cwd=source, check=True)
        template = (Path(__file__).parent / TEMPLATE).read_text()
        for index, service in enumerate(services):
            path = SUPERVISOR_ROOT / "jobs" / f"{service}.conf"
            if path.exists():
                raise FileExistsError(path)
            config = (template.replace("__SERVICE__", service)
                      .replace("__SOURCE__", str(source))
                      .replace("__PYTHON__", sys.executable)
                      .replace("__ROOT__", str(root))
                      .replace("__SHA__", args.source_commit)
                      .replace("__GPU__", str(index)))
            path.write_text(config)
        atomic_json(root / "scheduler-manifest.json", dict(
            campaign=CAMPAIGN, host=args.host, source_commit=args.source_commit,
            source=str(source), root=str(root), jobs=names, services=services,
            budgets=plan["budgets"], budget_multiplier=10,
            final_eval_episodes=2000, optiq_dacer_enabled=False,
            prepared=time.time(), started=False))
        print(json.dumps(dict(prepared=len(names), services=services, root=str(root))))
        return
    manifest_path = root / "scheduler-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest["source_commit"] != args.source_commit or
            manifest["host"] != args.host or manifest["services"] != services or
            manifest["started"]):
        raise ValueError("queue manifest mismatch or already started")
    for argument in ("reread", "update"):
        subprocess.run(["sudo", "supervisorctl", "-s", socket, argument], check=True)
    for service in services:
        subprocess.run(["sudo", "supervisorctl", "-s", socket, "start", service], check=True)
    manifest["started"] = time.time()
    atomic_json(manifest_path, manifest)
    print(json.dumps(dict(started=len(services), root=str(root))))


if __name__ == "__main__":
    main()
