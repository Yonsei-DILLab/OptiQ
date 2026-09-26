"""Committed 1M-only replacement campaign, independent GPU workers, no retries."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .run_nway_job import atomic_json, verify_source

CAMPAIGN = "maze-deadline-1m-20260926"
PLAN = Path(__file__).with_name("DEADLINE_1M_PLAN.json")
OPS = Path("/home/heechan/OptiQ-ops")


def transaction(root, action):
    with (root / "queue.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        status = json.loads((root / "queue.json").read_text())
        result = action(status)
        status["updated"] = time.time()
        atomic_json(root / "queue.json", status)
        return result


def claim(root, worker):
    def change(status):
        if status["state"] != "running":
            return None
        for job in status["jobs"]:
            if job["state"] == "pending":
                job.update(state="running", worker=worker, pid=os.getpid(), started=time.time())
                return dict(job)
        return None
    return transaction(root, change)


def command(job, output, commit, preflight):
    is_maze = job["task"].startswith("pm_")
    num_envs = job.get("num_envs", 256)
    warmup = job.get("warmup", 8192)
    steps = warmup + num_envs if preflight else job.get("steps", 1_000_192)
    updates_per_collect = job.get("updates_per_collect", 16)
    final = 5 if preflight else 500 if is_maze else 1024
    cmd = [sys.executable, "-m", "maze_benchmarks.run", "--task", job["task"],
            "--method", job["method"], "--temperature", str(job["temperature"]),
            "--seed", str(job.get("seed", 0)), "--output", str(output), "--source-commit", commit,
            "--steps", str(steps), "--num-envs", str(num_envs),
            "--batch-size", str(job.get("batch_size", 4096)),
            "--updates-per-collect", str(updates_per_collect), "--warmup", str(warmup),
            "--eval-every", str(steps if preflight else job.get("eval_every", 200_000)),
            "--eval-episodes", str(5 if preflight else 200),
            "--final-eval-episodes", str(final), "--render-each-eval"]
    for key in ("meow_alpha", "mfpo_target_entropy_per_dim"):
        if key in job:
            cmd += ["--" + key.replace("_", "-"), str(job[key])]
    if job.get("entropy_diagnostics"):
        cmd.append("--entropy-diagnostics")
    return cmd


def verify(output, job, commit, preflight):
    config = json.loads((output / "config.json").read_text())
    progress = json.loads((output / "progress.json").read_text())
    num_envs = job.get("num_envs", 256)
    warmup = job.get("warmup", 8192)
    updates_per_collect = job.get("updates_per_collect", 16)
    steps = warmup + num_envs if preflight else job.get("steps", 1_000_192)
    updates = (steps - warmup) // num_envs * updates_per_collect
    if (progress["status"], progress["steps"], progress["updates"]) != ("complete", steps, updates):
        raise ValueError("incomplete step/update audit")
    wanted = dict(task=job["task"], method=job["method"], temperature=job["temperature"],
                  source_commit=commit, seed=job.get("seed", 0), num_envs=num_envs,
                  batch_size=job.get("batch_size", 4096),
                  updates_per_collect=updates_per_collect, warmup=warmup, steps=steps)
    for key in ("meow_alpha", "mfpo_target_entropy_per_dim", "entropy_diagnostics"):
        if key in job:
            wanted[key] = job[key]
    if any(config.get(k) != v for k, v in wanted.items()):
        raise ValueError("configuration differs from approved plan")
    if job["method"] == "dipo" and (config["agent"].get("upstream") != "BellmanTimeHut/DIPO" or
                                   config["agent"].get("actual_actor_class") != "Diffusion" or
                                   config["agent"].get("n_timesteps") != 100):
        raise ValueError("official MuJoCo DIPO implementation was not constructed")
    if job.get("dipo_upstream_memory") and (
            config["agent"].get("diffusion_memory_replace_is_upstream") is not True or
            config["agent"].get("diffusion_memory_replace") !=
            "upstream np.copyto (no persistent writeback)"):
        raise ValueError("DIPO replay replace must inherit the pinned upstream behavior")
    for key, agent_key in (("meow_alpha", "alpha"),
                           ("mfpo_target_entropy_per_dim", "target_entropy_coeff")):
        if key in job and config["agent"].get(agent_key) != job[key]:
            raise ValueError("agent sensitivity parameter differs from plan")
    record = progress["latest_evaluation"]
    if record["step"] != steps:
        raise ValueError("final evaluation step mismatch")
    if job.get("entropy_diagnostics"):
        diagnostic = json.loads((output / "evaluations" / f"{steps:09d}_entropy.json").read_text())
        if (diagnostic["step"] != steps or diagnostic["sample_count"] != 512 or
                not np.isfinite(diagnostic["entropy_estimate"]) or diagnostic["alpha"] <= 0):
            raise ValueError("invalid entropy diagnostic")
        if not (output / "training-metrics.jsonl").is_file():
            raise ValueError("missing entropy optimization history")
    mu_primary = job["method"] == "optiq" and config.get("optiq_eval_mode") == "mu_only"
    modes = ["mu_only"] if mu_primary else ["policy"]
    if job["method"] == "optiq" and not mu_primary: modes.append("mu_only")
    if job["task"].startswith("pm_"):
        modes.append("obstacle_mu_only" if mu_primary else "obstacle_policy")
    if mu_primary and ("policy" in record or record.get("primary_evaluation_mode") != "mu_only"):
        raise ValueError("OptiQ evaluation contains unexpected conditional sigma mode")
    episodes = 5 if preflight else 500 if job["task"].startswith("pm_") else 1024
    for mode in modes:
        summary = record[mode]
        if summary["episodes"] != episodes or sum(summary["goals"]) + summary["failure"] != episodes:
            raise ValueError("invalid goal/episode accounting")
        with np.load(output / "evaluations" / f"{steps:09d}_{mode}.npz") as raw:
            ids = raw["goal_ids"]
            if ids.shape != (episodes,) or not np.isfinite(raw["returns"]).all():
                raise ValueError("invalid raw evaluation")
            if np.bincount(ids[ids >= 0], minlength=len(summary["goals"])).tolist() != summary["goals"]:
                raise ValueError("raw goals disagree with summary")
    checkpoint = list((output / "checkpoints").glob(f"policy_{steps:09d}.*"))
    if not checkpoint or not (output / "checkpoints" / f"replay_{steps:09d}.npz").is_file():
        raise ValueError("missing checkpoint/replay")
    if not (output / "figures" / f"{steps:09d}_trajectories.png").is_file():
        raise ValueError("missing automatic figure")
    if not job["task"].startswith("pm_"):
        with np.load(output / "evaluations" / f"{steps:09d}_probe.npz") as raw:
            if not np.isfinite(raw["q"]).all() or not np.isfinite(raw["actions"]).all():
                raise ValueError("nonfinite Q/action probe")
    hashes = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in output.rglob("*") if p.is_file() and "learner" not in p.parts}
    return dict(steps=steps, updates=updates, record=record, sha256=hashes)


def work(root, commit, worker, source):
    while (job := claim(root, worker)) is not None:
        path = root / "jobs" / (job["name"] + ".json")
        atomic_json(path, job)
        try:
            for stage, preflight in (("preflights", True), ("runs", False)):
                output = root / stage / job["name"]
                if output.exists():
                    raise FileExistsError("preserved attempt exists; automatic restart forbidden")
                job.update(stage=stage, updated=time.time())
                atomic_json(path, job)
                cmd = command(job, output, commit, preflight)
                atomic_json(root / "logs" / f"{job['name']}-{stage}-command.json", cmd)
                with (root / "logs" / f"{job['name']}-{stage}.log").open("w") as log:
                    result = subprocess.run(cmd, cwd=source, env=dict(os.environ, PYTHONPATH=str(source)),
                                            stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f"{stage} exit {result.returncode}; inspect preserved log")
                proof = verify(output, job, commit, preflight)
                atomic_json(root / "proofs" / f"{job['name']}-{stage}.json", proof)
            job.update(state="complete", finished=time.time())
            atomic_json(path, job)
            def complete(status):
                next(j for j in status["jobs"] if j["name"] == job["name"]).update(job)
                if all(j["state"] == "complete" for j in status["jobs"]): status["state"] = "complete"
            transaction(root, complete)
        except BaseException as error:
            job.update(state="failed", error=repr(error), failed=time.time())
            atomic_json(path, job)
            def fail(status):
                next(j for j in status["jobs"] if j["name"] == job["name"]).update(job)
                status.update(state="paused", error=f"{job['name']}: {error}")
            transaction(root, fail)
            raise


def prepare(root, commit, host, source, *, plan_path=PLAN, campaign=CAMPAIGN, gpus=None):
    plan = json.loads(plan_path.read_text())
    jobs = [dict(j, state="pending") for j in plan["jobs"] if j["host"] == host]
    gpu_count = {"vast-heechan-180": 4, "vast-heechan-199": 4, "vast-heechan-46": 8, "vast-heechan-6": 4}[host]
    gpu_indices = list(range(gpu_count)) if gpus is None else list(gpus)
    if not jobs or not gpu_indices or len(set(gpu_indices)) != len(gpu_indices) or any(i < 0 or i >= gpu_count for i in gpu_indices):
        raise ValueError("invalid host jobs or GPU selection")
    controller = Path(__file__).resolve().parents[1]
    root.mkdir(parents=True, exist_ok=False)
    for part in ("logs", "jobs", "proofs", "preflights", "runs"): (root / part).mkdir()
    atomic_json(root / "manifest.json", dict(plan=plan, source_commit=commit, source=str(source),
                                            controller_source=str(controller), host=host))
    atomic_json(root / "queue.json", dict(state="running", source_commit=commit, jobs=jobs, created=time.time()))
    sup = OPS / "supervisor"
    for part in (sup / "jobs", sup / "logs", OPS / "locks"): part.mkdir(parents=True, exist_ok=True)
    config = sup / "supervisord.conf"
    if not config.exists():
        config.write_text(f"[unix_http_server]\nfile={sup}/supervisor.sock\nchmod=0700\n[supervisord]\nlogfile={sup}/supervisord.log\npidfile={sup}/supervisord.pid\nchildlogdir={sup}/logs\n[rpcinterface:supervisor]\nsupervisor.rpcinterface_factory=supervisor.rpcinterface:make_main_rpcinterface\n[supervisorctl]\nserverurl=unix://{sup}/supervisor.sock\n[include]\nfiles={sup}/jobs/*.conf\n")
    services = []
    for gpu in gpu_indices:
        service = f"{campaign}-{host}-gpu{gpu}"
        services.append(service)
        filename = sup / "jobs" / f"{service}.conf"
        if filename.exists(): raise FileExistsError(filename)
        filename.write_text(f'''[program:{service}]
directory={controller}
command={sys.executable} -m maze_benchmarks.gpu_guard --gpu {gpu} --lock {OPS}/locks/gpu-{gpu}.lock -- {sys.executable} -m maze_benchmarks.deadline_queue --root {root} --source-commit {commit} --training-source {source} --host {host} --worker gpu{gpu} --phase work
environment=PYTHONPATH="{controller}",CUDA_VISIBLE_DEVICES="{gpu}",XLA_PYTHON_CLIENT_PREALLOCATE="false",MPLBACKEND="Agg",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",LD_LIBRARY_PATH="/home/heechan/.mujoco/mujoco210/bin",MUJOCO_PY_MUJOCO_PATH="/home/heechan/.mujoco/mujoco210"
autostart=false
autorestart=false
stopasgroup=true
killasgroup=true
stopwaitsecs=20
startsecs=0
stdout_logfile={root}/logs/{service}.log
redirect_stderr=true
''')
    atomic_json(root / "registration.json", dict(services=services, source_commit=commit, started=False))


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--host", choices=("vast-heechan-180", "vast-heechan-199", "vast-heechan-46", "vast-heechan-6"), required=True)
    parser.add_argument("--training-source", type=Path)
    parser.add_argument("--phase", choices=("prepare", "start", "work"), required=True)
    parser.add_argument("--worker")
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--campaign", default=CAMPAIGN)
    parser.add_argument("--gpus", type=int, nargs="+")
    args = parser.parse_args()
    source = args.training_source.resolve() if args.training_source else Path(__file__).resolve().parents[1]
    verify_source(source, args.source_commit)
    if args.phase == "prepare":
        return prepare(args.root, args.source_commit, args.host, source,
                       plan_path=args.plan, campaign=args.campaign, gpus=args.gpus)
    if args.phase == "work":
        if not args.worker: raise ValueError("worker name required")
        return work(args.root, args.source_commit, args.worker, source)
    sup = OPS / "supervisor"
    registration = json.loads((args.root / "registration.json").read_text())
    if registration["started"] or registration["source_commit"] != args.source_commit:
        raise ValueError("already started or source mismatch")
    if not (sup / "supervisor.sock").exists():
        subprocess.run(["/usr/local/bin/supervisord", "-c", str(sup / "supervisord.conf")], check=True)
    base = ["sudo", "supervisorctl", "-s", f"unix://{sup}/supervisor.sock"]
    for op in ("reread", "update"): subprocess.run(base + [op], check=True)
    for name in registration["services"]: subprocess.run(base + ["start", name], check=True)
    registration["started"] = time.time()
    atomic_json(args.root / "registration.json", registration)


if __name__ == "__main__": main()
