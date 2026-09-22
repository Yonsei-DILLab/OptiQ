"""Validate and run one v1 DiPo control without resuming cancelled campaigns."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from antmaze.evaluation import atomic_json

NAME = "antmaze-v1-dipo-noveld001-1m-s0-20260922"
PROFILE = "v1-dipo-noveld001-1m"
PYTHON = "/home/heechan/.venv-optiq-antmaze/bin/python"
WRAPPER = "/home/heechan/OptiQ-ops/run-gpu.sh"
UPSTREAM = "c6d8d1b39d6cea22e7d779e08111dbf974dbb4fc"


def source_sha():
    source = Path(__file__).resolve().parents[2]
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=source, text=True).strip()
    return source, sha


def verify(folder, smoke=False):
    from antmaze.multimodal.dense_noveld_report import verify_run
    _, sha = source_sha()
    return verify_run(folder, allow_smoke=smoke, expected_source=sha,
        expected_steps=1000000, expected_profile=PROFILE, expected_coefficient=.01)


def check_config(folder):
    from antmaze.multimodal.resume import digest
    import torch
    source, sha = source_sha()
    c = json.loads((folder / "config.json").read_text())
    assert c["task"] == "v1" and c["method"] == "dipo" and c["seed"] == 0
    assert c["eval_interval"] == 250000 and c["checkpoint_interval"] == 0
    expected = dict(policy_type="Diffusion", noise_ratio=1., beta_schedule="cosine",
        n_timesteps=100, diffusion_lr=3e-4, critic_lr=3e-4, action_gradient_steps=20,
        ratio=.1, ac_grad_norm=2., tau=.005, update_actor_target_every=1, action_lr=.03)
    assert c["native"] == expected
    dependency = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source / "gmm40-baseline/DIPO", text=True).strip()
    assert dependency == UPSTREAM
    reference = Path("/home/heechan/optiq-experiments/antmaze-dense-noveld-1m-s0-20260922/runs/v1-optiq-s0")
    old = json.loads((reference / "config.json").read_text())
    for field in ("environment", "xml_sha256", "intrinsic", "reward", "batch_size", "num_envs", "utd"):
        assert c[field] == old[field], field
    old_rnd = json.loads((reference / "intrinsic-audit.json").read_text())["initial"]
    assert json.loads((folder / "intrinsic-audit.json").read_text())["initial"] == old_rnd
    step = c["steps"]
    snapshot = folder / "resume" / f"step_{step:010d}"
    state = torch.load(snapshot / "state.pt", map_location="cpu", weights_only=False)
    parameters = state["model"]["parameters"]
    for name in ("actor_optimizer", "critic_optimizer"):
        group = parameters[name]["param_groups"][0]
        assert group["lr"] == 3e-4 and group["eps"] == 1e-5
    import numpy as np
    with np.load(snapshot / "replay.npz") as data:
        improved = state["model"]["diffusion_memory"]["best_actions"]
        assert not np.array_equal(improved, data["actions"]), "DiPo best-action updates were not persisted"
    return dict(passed=True, source_commit=sha, upstream=dependency,
        common_settings_and_rnd_initialization_match=str(reference),
        persistent_action_improvement=True, native_optimizer=True)


def worker(root, stage):
    source, sha = source_sha()
    def run(folder, steps, *extra):
        subprocess.run([PYTHON, "-u", "-m", "antmaze.multimodal.dense_noveld_run",
            "--task", "v1", "--method", "dipo", "--output", str(folder), "--steps", str(steps),
            "--noveld-coefficient", ".01", "--campaign-name", NAME,
            "--run-name", "v1-dipo-s0-c0.01-1m", "--profile", PROFILE,
            "--eval-interval", "250000", "--checkpoint-interval", "0", *extra],
            cwd=source, check=True)
    if stage == "preflight":
        first, second = root / "preflight/initial", root / "preflight/continued"
        run(first, 272, "--smoke")
        first_proof = verify(first, True)
        config_proof = check_config(first)
        run(second, 280, "--smoke", "--resume", str(first / "resume/step_0000000272"))
        proof = verify(second, True)
        restored = json.loads((second / "resume-verification.json").read_text())
        assert restored["passed"] and restored["loaded_step"] == 272 and restored["loaded_updates"] == 16
        atomic_json(root / "preflight-proof.json", dict(passed=True, source_commit=sha,
            initial=first_proof, continued=proof, restored=restored, config=config_proof))
    else:
        proof = json.loads((root / "preflight-proof.json").read_text())
        assert proof["passed"] and proof["source_commit"] == sha
        folder = root / "runs/v1-dipo-s0"
        run(folder, 1000000)
        validation = verify(folder)
        config_proof = check_config(folder)
        atomic_json(folder / "verification.json", dict(**validation, config=config_proof))
        atomic_json(root / "result.json", dict(completed=True, source_commit=sha,
            task="v1", method="dipo", seed=0, steps=1000000,
            run=str(folder), verification=validation))


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--worker", choices=("preflight", "training"))
    args = parser.parse_args()
    root = args.root
    if args.worker:
        return worker(root, args.worker)
    root.mkdir(exist_ok=True)
    lock = (root / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (root / "manifest.json").exists(), "Refuse duplicate launch"
    source, sha = source_sha()
    atomic_json(root / "manifest.json", dict(source_commit=sha, source=str(source),
        task="v1", method="dipo", seed=0, steps=1000000, coefficient=.01,
        eval_interval=250000, checkpoint_interval=0, project="OptiQ/gmm-trg", group=NAME,
        total_runs=1, no_other_jobs=True))
    for stage in ("preflight", "training"):
        chosen = None
        while chosen is None:
            for gpu in range(4):
                probe = open(f"/home/heechan/OptiQ-ops/locks/gpu-{gpu}.lock", "a")
                try:
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    probe.close()
                    continue
                probe.close()
                chosen = gpu
                break
            if chosen is None:
                atomic_json(root / "status.json", dict(stage=stage, status="waiting_for_gpu", source_commit=sha))
                time.sleep(2)
        env = os.environ.copy()
        env.update(OPTIQ_SOURCE_DIR=str(source), PYTHONPATH=str(source), CAMPAIGN_GPU=str(chosen))
        command = [WRAPPER, str(chosen), "--branch", "v5-direct-gmm", PYTHON, "-u", "-m",
            "antmaze.v1_dipo_noveld001.campaign", "--root", str(root), "--worker", stage]
        with (root / f"{stage}.log").open("x") as log:
            proc = subprocess.Popen(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
            status = dict(stage=stage, status="running", pid=proc.pid, gpu=chosen,
                source_commit=sha, started=time.time(), command=command)
            atomic_json(root / "status.json", status)
            code = proc.wait()
        proof = root / ("preflight-proof.json" if stage == "preflight" else "result.json")
        if code != 0 or not proof.exists():
            atomic_json(root / "failure.json", dict(**status, exit_code=code, automatic_restart=False))
            atomic_json(root / "status.json", dict(status, status="failed", exit_code=code))
            raise SystemExit(code or 1)
    atomic_json(root / "status.json", dict(status="completed", stage="training", source_commit=sha, finished=time.time()))


if __name__ == "__main__":
    main()
