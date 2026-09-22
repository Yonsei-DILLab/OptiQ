"""Each slot validates and trains its own job; no maze-wide completion gate."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from antmaze.evaluation import atomic_json
from antmaze.v1_dipo_noveld001.campaign import PYTHON, WRAPPER, UPSTREAM, source_sha

NAME = "antmaze-v234-noveld01-1m-s0-20260922"
PROFILE = "v234-noveld01-1m"
METHODS = ("optiq", "sac", "dipo", "mfpo")
TASKS = ("v2", "v3", "v4")
# Start slow DiPo jobs early. v1 DiPo already owns one GPU on host180.
SHARDS = {
    0: (("v3", "dipo"), ("v3", "optiq"), ("v3", "sac"),
        ("v3", "mfpo"), ("v4", "optiq"), ("v4", "sac")),
    1: (("v2", "dipo"), ("v4", "dipo"), ("v2", "optiq"),
        ("v2", "sac"), ("v2", "mfpo"), ("v4", "mfpo")),
}


def validate(folder, smoke):
    from antmaze.multimodal.dense_noveld_report import verify_run
    return verify_run(folder, allow_smoke=smoke, expected_source=source_sha()[1],
        expected_steps=1000000, expected_profile=PROFILE, expected_coefficient=.1)


def compare_config(folder, task, method):
    import numpy as np
    import torch
    c = json.loads((folder / "config.json").read_text())
    source, sha = source_sha()
    assert c["task"] == task and c["method"] == method and c["seed"] == 0
    assert c["eval_interval"] == 250000 and c["checkpoint_interval"] == 0
    references = json.loads(Path(__file__).with_name("references.json").read_text())
    baseline = references[f"{task}-{'optiq' if method == 'dipo' else method}-s0"]
    reference = baseline["reference"]
    old = baseline["config"]
    for field in ("environment", "xml_sha256", "reward", "batch_size", "num_envs", "utd"):
        assert c[field] == old[field], field
    intrinsic = dict(c["intrinsic"])
    assert intrinsic.pop("coefficient") == .1
    previous_intrinsic = dict(old["intrinsic"])
    assert previous_intrinsic.pop("coefficient") == .01
    assert intrinsic == previous_intrinsic
    rnd = json.loads((folder / "intrinsic-audit.json").read_text())["initial"]
    assert rnd == baseline["initial_rnd"]
    if method != "dipo":
        native, previous = dict(c["native"]), dict(old["native"])
        native.pop("output_root", None)
        previous.pop("output_root", None)
        assert native == previous, "Native training hyperparameters changed"
        assert json.loads((folder / "parameter-audit.json").read_text())["initial"] == baseline["initial_model"]
    else:
        assert c["native"] == dict(policy_type="Diffusion", noise_ratio=1., beta_schedule="cosine",
            n_timesteps=100, diffusion_lr=3e-4, critic_lr=3e-4, action_gradient_steps=20,
            ratio=.1, ac_grad_norm=2., tau=.005, update_actor_target_every=1, action_lr=.03)
        dependency = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source / "gmm40-baseline/DIPO", text=True).strip()
        assert dependency == UPSTREAM
        cp = folder / "resume" / f"step_{c['steps']:010d}"
        state = torch.load(cp / "state.pt", map_location="cpu", weights_only=False)
        for name in ("actor_optimizer", "critic_optimizer"):
            group = state["model"]["parameters"][name]["param_groups"][0]
            assert group["lr"] == 3e-4 and group["eps"] == 1e-5
        with np.load(cp / "replay.npz") as data:
            assert not np.array_equal(state["model"]["diffusion_memory"]["best_actions"], data["actions"])
    return dict(passed=True, source_commit=sha, reference=str(reference),
        coefficient=.1, previous_coefficient=.01, common_config_and_rnd_match=True,
        native_hyperparameters_preserved=True)


def worker(root, task, method):
    source, sha = source_sha()
    job = f"{task}-{method}-s0"
    def call(folder, steps, *extra):
        subprocess.run([PYTHON, "-u", "-m", "antmaze.multimodal.dense_noveld_run",
            "--task", task, "--method", method, "--output", str(folder), "--steps", str(steps),
            "--noveld-coefficient", ".1", "--campaign-name", NAME, "--run-name", job + "-c0.1-1m",
            "--profile", PROFILE, "--eval-interval", "250000", "--checkpoint-interval", "0", *extra],
            cwd=source, check=True)
    stage_file = root / "stages" / (job + ".json")
    atomic_json(stage_file, dict(stage="preflight", source_commit=sha))
    first = root / "preflight" / job / "initial"
    second = root / "preflight" / job / "continued"
    call(first, 272, "--smoke")
    initial_proof = validate(first, True)
    config_proof = compare_config(first, task, method)
    call(second, 280, "--smoke", "--resume", str(first / "resume/step_0000000272"))
    continued_proof = validate(second, True)
    restored = json.loads((second / "resume-verification.json").read_text())
    assert restored["passed"] and restored["loaded_step"] == 272 and restored["loaded_updates"] == 16
    atomic_json(root / "proofs" / (job + ".json"), dict(passed=True, source_commit=sha,
        initial=initial_proof, continued=continued_proof, baseline_match=config_proof, restored=restored))
    atomic_json(stage_file, dict(stage="training", source_commit=sha, started=time.time()))
    folder = root / "runs" / job
    call(folder, 1000000)
    proof = validate(folder, False)
    atomic_json(folder / "verification.json", proof)
    atomic_json(stage_file, dict(stage="completed", source_commit=sha, finished=time.time()))


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--shard", type=int, choices=(0, 1))
    p.add_argument("--worker", choices=METHODS)
    p.add_argument("--task", choices=TASKS)
    a = p.parse_args()
    root = a.root
    if a.worker:
        assert a.task
        return worker(root, a.task, a.worker)
    assert a.shard is not None
    root.mkdir(exist_ok=True)
    lock = (root / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (root / "manifest.json").exists(), "Refuse duplicate launch"
    source, sha = source_sha()
    assert len(set(SHARDS[0] + SHARDS[1])) == 12
    assert set(SHARDS[0] + SHARDS[1]) == {(t, m) for t in TASKS for m in METHODS}
    for name in ("jobs", "stages", "proofs", "logs", "runs", "preflight"):
        (root / name).mkdir(exist_ok=True)
    jobs = [dict(id=f"{t}-{m}-s0", task=t, method=m, seed=0, steps=1000000,
        status="pending", stage="queued") for t, m in SHARDS[a.shard]]
    atomic_json(root / "manifest.json", dict(source_commit=sha, source=str(source), shard=a.shard,
        methods=METHODS, tasks=TASKS, coefficient=.1, total_across_hosts=12, jobs=jobs,
        project="OptiQ/gmm-trg", group=NAME, eval_interval=250000, checkpoint_interval=0,
        scheduling="independent per-run preflight+training; 2s backfill; no maze barrier; slow DIPO first"))
    live = {}
    failed = False
    while True:
        for gpu, (proc, job, log) in list(live.items()):
            stage = root / "stages" / (job["id"] + ".json")
            if stage.exists():
                job["stage"] = json.loads(stage.read_text())["stage"]
            code = proc.poll()
            if code is None:
                continue
            log.close()
            del live[gpu]
            proof = root / "runs" / job["id"] / "verification.json"
            good = code == 0 and proof.exists() and json.loads(proof.read_text())["passed"]
            job.update(status="completed" if good else "failed", exit_code=code, finished=time.time())
            if not good:
                failed = True
                atomic_json(root / "failure.json", dict(job=job, pending_held=True,
                    live_preserved=True, automatic_restart=False))
            atomic_json(root / "jobs" / (job["id"] + ".json"), job)
        if not failed:
            for gpu in range(4):
                if gpu in live:
                    continue
                job = next((j for j in jobs if j["status"] == "pending"), None)
                if job is None:
                    break
                probe = open(f"/home/heechan/OptiQ-ops/locks/gpu-{gpu}.lock", "a")
                try:
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    probe.close()
                    continue
                probe.close()
                command = [WRAPPER, str(gpu), "--branch", "v5-direct-gmm", PYTHON, "-u", "-m",
                    "antmaze.noveld01_12_1m.campaign", "--root", str(root), "--worker", job["method"], "--task", job["task"]]
                env = os.environ.copy()
                env.update(OPTIQ_SOURCE_DIR=str(source), PYTHONPATH=str(source), CAMPAIGN_GPU=str(gpu))
                log = (root / "logs" / (job["id"] + ".log")).open("x")
                proc = subprocess.Popen(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
                job.update(status="running", stage="preflight", gpu=gpu, pid=proc.pid, started=time.time(), command=command)
                atomic_json(root / "jobs" / (job["id"] + ".json"), job)
                live[gpu] = proc, job, log
        atomic_json(root / "status.json", dict(source_commit=sha, shard=a.shard, updated=time.time(),
            completed=sum(j["status"] == "completed" for j in jobs), running=len(live),
            pending=sum(j["status"] == "pending" for j in jobs), failed=failed, jobs=jobs))
        if not live and (failed or all(j["status"] == "completed" for j in jobs)):
            break
        time.sleep(2)
    if failed:
        raise SystemExit(1)
    atomic_json(root / "result.json", dict(completed=True, source_commit=sha, shard=a.shard, jobs=jobs))


if __name__ == "__main__":
    main()
