"""Resume renamed W&B runs and relay evaluation metrics from SB3 CSV logs."""

from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as mp
from pathlib import Path
import subprocess
import time

import wandb


RUNS = (
    (0, "nlppqf7t"),
    (1, "7mzxka76"),
    (2, "nhv7xzo7"),
    (3, "76ktfhwh"),
    (4, "b5hm8fvh"),
    (5, "0jh8fn5d"),
)
ENTITY = "OptiQ"
PROJECT = "optiq-direct-gmm-trg-vs-smem-tr"


def remote_last_eval_step(run_id: str) -> int:
    public_run = wandb.Api(timeout=30).run(f"{ENTITY}/{PROJECT}/{run_id}")
    steps = [
        int(row["env_steps"])
        for row in public_run.scan_history(
            keys=["env_steps", "eval/mean_reward"], page_size=1000
        )
        if row.get("env_steps") is not None
        and row.get("eval/mean_reward") is not None
    ]
    return max(steps, default=-1)


def locate_progress_csv(campaign: Path, run_id: str) -> Path:
    matches = list(campaign.glob(f"outputs/*/wandb/run-*-{run_id}/run-{run_id}.wandb"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one local W&B file for {run_id}, got {matches}")
    progress = matches[0].parents[2] / "logs" / "progress.csv"
    if not progress.is_file():
        raise FileNotFoundError(progress)
    return progress


def new_eval_rows(progress: Path, after_step: int) -> list[tuple[int, dict[str, float]]]:
    result: list[tuple[int, dict[str, float]]] = []
    with progress.open(newline="") as handle:
        for row in csv.DictReader(handle):
            reward = row.get("eval/mean_reward", "")
            raw_step = row.get("time/total_timesteps", "")
            if not reward or not raw_step:
                continue
            step = int(float(raw_step))
            if step <= after_step:
                continue
            metrics: dict[str, float] = {"env_steps": step}
            for key, value in row.items():
                if not key.startswith("eval/") or not value:
                    continue
                number = float(value)
                if math.isfinite(number):
                    metrics[key] = number
            result.append((step, metrics))
    return sorted(result)


def task_is_active(array_job_id: str, task_index: int) -> bool:
    result = subprocess.run(
        ["squeue", "-h", "-j", f"{array_job_id}_{task_index}", "-o", "%T"],
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def watch_one(campaign: str, array_job_id: str, task_index: int, run_id: str) -> None:
    campaign_path = Path(campaign)
    progress = locate_progress_csv(campaign_path, run_id)
    last_step = remote_last_eval_step(run_id)
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        id=run_id,
        resume="must",
        settings=wandb.Settings(_service_wait=300),
        dir=str(progress.parents[1]),
    )
    run.define_metric("env_steps")
    run.define_metric("*", step_metric="env_steps")
    print(f"[{run_id}] resumed after eval env_steps={last_step}", flush=True)
    missing_polls = 0
    try:
        while True:
            for step, metrics in new_eval_rows(progress, last_step):
                run.log(metrics)
                last_step = step
                print(f"[{run_id}] relayed eval env_steps={step}", flush=True)
            if task_is_active(array_job_id, task_index):
                missing_polls = 0
            else:
                missing_polls += 1
                if missing_polls >= 2:
                    break
            time.sleep(20)
        # Catch the final CSV flush after the training process leaves squeue.
        time.sleep(10)
        for step, metrics in new_eval_rows(progress, last_step):
            run.log(metrics)
            last_step = step
            print(f"[{run_id}] relayed final eval env_steps={step}", flush=True)
    finally:
        run.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--array-job-id", required=True)
    args = parser.parse_args()
    context = mp.get_context("spawn")
    processes = [
        context.Process(
            target=watch_one,
            args=(str(args.campaign), args.array_job_id, task_index, run_id),
            name=f"wandb-{run_id}",
        )
        for task_index, run_id in RUNS
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    failures = [process.name for process in processes if process.exitcode != 0]
    if failures:
        raise SystemExit(f"W&B sidecars failed: {failures}")


if __name__ == "__main__":
    main()
