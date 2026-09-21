"""Six approved runs, four GPU slots, immediate backfill, no automatic retries."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CAMPAIGN = "trg-dacer-mean1-20260921"
ROOT = Path("/home/heechan/optiq-experiments") / CAMPAIGN
PYTHON = "/home/heechan/.venv-optiq-mujoco/bin/python"
DEPS = ROOT.parent / "trg-dacer-priority-20260921/deps"
CTL = ["supervisorctl", "-c", "/home/heechan/OptiQ-ops/supervisor/supervisord.conf"]
spec = importlib.util.spec_from_file_location("mean1_previous_controller", HERE.parent / "20260921_dacer_priority/controller.py")
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
save = previous.save


def plan():
    value = json.loads((HERE / "plan.json").read_text())
    assert value["campaign"] == CAMPAIGN and value["gpus"] == list(range(4))
    assert [(j["task"], j["seed"]) for j in value["jobs"]] == [
        ("halfcheetah", 0), ("halfcheetah", 1), ("ant", 0), ("ant", 1), ("ant", 2), ("ant", 3)]
    return value


def available(gpu):
    busy = subprocess.check_output(["nvidia-smi", f"--id={gpu}",
        "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    if busy:
        return False
    with open(f"/home/heechan/OptiQ-ops/locks/gpu-{gpu}.lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
    return True


def setup():
    from train_mean1 import compose_config, name
    from omegaconf import OmegaConf
    sha = previous.old.source()
    requested = plan()
    assert DEPS.is_dir()
    assert not ROOT.exists(), "Campaign already exists; do not relaunch"
    # Resolve every configuration before registering or launching anything.
    jobs = []
    for row in requested["jobs"]:
        cfg = compose_config(row["task"], row["seed"], str(ROOT / "outputs"))
        jobs.append(dict(row, name=name(**row), stage="dacer", temperature=.25, beta=1.,
                         mean_output_init_scale=1., commit=sha, status="queued",
                         resolved_config=OmegaConf.to_container(cfg, resolve=True)))
    ROOT.mkdir()
    for folder in ("jobs", "logs", "outputs"):
        (ROOT / folder).mkdir()
    for job in jobs:
        save(ROOT / "jobs" / (job["name"] + ".json"), job)
    manifest = dict(requested, commit=sha, source=str(REPO), branch="direct-gmm-trg",
                    project="OptiQ/gmm-trg", jobs=[j["name"] for j in jobs], deps=str(DEPS),
                    created=time.time(), total_steps=1_000_000,
                    scheduling="First four immediately; backfill any free GPU every 2 seconds",
                    failure_policy="Preserve active runs; block pending jobs; no automatic retries")
    save(ROOT / "manifest.json", manifest)
    conf = Path("/home/heechan/OptiQ-ops/supervisor/jobs") / (CAMPAIGN + ".conf")
    assert not conf.exists()
    conf.write_text(f"""[program:{CAMPAIGN}]
command={PYTHON} -u {HERE}/controller.py run
directory={REPO}
autostart=true
autorestart=false
startsecs=3
startretries=0
stopasgroup=true
killasgroup=true
stopwaitsecs=60
redirect_stderr=true
stdout_logfile={ROOT}/controller.log
stdout_logfile_maxbytes=10MB
stdout_logfile_backups=2
environment=PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2"
""")
    subprocess.run(CTL + ["reread"], check=True)
    subprocess.run(CTL + ["update", CAMPAIGN], check=True)
    print(json.dumps(manifest), flush=True)


def verify(job):
    result = previous.verified(job, ROOT)
    cfg = json.loads((Path(result["run_dir"]) / "config.json").read_text())
    expected = job["resolved_config"]
    assert cfg["alg"] == expected["alg"] and cfg["dacer"] == expected["dacer"]
    for key in ("seed", "task", "total_steps", "eval_interval", "num_eval_episodes", "dual_mu_eval", "mu_only_eval"):
        assert cfg[key] == expected[key]
    return result


def run():
    lock = (ROOT / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((ROOT / "manifest.json").read_text())
    assert previous.old.source() == manifest["commit"]
    jobs = [json.loads((ROOT / "jobs" / (n + ".json")).read_text()) for n in manifest["jobs"]]
    assert not any(j["status"] in ("running", "failed") for j in jobs), "Manual review required; no automatic restart"
    pending = [j for j in jobs if j["status"] == "queued"]
    active = {}
    while pending or active:
        for gpu, (process, job, log) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            del active[gpu]
            job.update(status="failed", exit_code=code, finished=time.time())
            try:
                job.update(verify(job))
                assert code == 0 or job.get("logging_warning"), f"Unexpected exit code {code}"
            except Exception as error:
                job.update(status="failed", error=repr(error))
            save(ROOT / "jobs" / (job["name"] + ".json"), job)
            print(json.dumps(dict(event="finished", name=job["name"], gpu=gpu, status=job["status"])), flush=True)
        failed = [j for j in jobs if j["status"] == "failed"]
        for gpu in manifest["gpus"]:
            if failed or gpu in active or not pending or not available(gpu):
                continue
            job = pending.pop(0)
            args = ["/home/heechan/OptiQ-ops/run-gpu.sh", str(gpu), "--branch", "v5-direct-gmm",
                    PYTHON, "-u", str(HERE / "train_mean1.py"), job["task"], str(job["seed"]), str(ROOT / "outputs")]
            env = os.environ.copy()
            env.pop("JAX_PLATFORMS", None)
            env.update(OPTIQ_SOURCE_DIR=str(REPO), CAMPAIGN_GPU=str(gpu),
                       PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(DEPS), WANDB_MODE="online")
            log = (ROOT / "logs" / (job["name"] + ".log")).open("x")
            process = subprocess.Popen(args, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            job.update(status="running", pid=process.pid, gpu=gpu, started=time.time(), command=args)
            save(ROOT / "jobs" / (job["name"] + ".json"), job)
            active[gpu] = (process, job, log)
            print(json.dumps(dict(event="started", name=job["name"], gpu=gpu)), flush=True)
        phase = "failed" if failed else "running" if active or pending else "completed"
        save(ROOT / "status.json", dict(phase=phase, updated=time.time(), commit=manifest["commit"],
             **{state: [j["name"] for j in jobs if j["status"] == state] for state in ("running", "queued", "completed", "failed")}))
        if failed:
            save(ROOT / "failure.json", dict(failed=failed, updated=time.time()))
            if not active:
                raise RuntimeError("Failed runs need review; pending jobs preserved")
        if pending or active:
            time.sleep(2)
    aggregate = {}
    for task in ("halfcheetah", "ant"):
        cohort = [j for j in jobs if j["task"] == task]
        aggregate[task] = {mode: dict(mean=float(np.mean(values := [j["metrics"][mode + "_last_100k_mean"] for j in cohort])),
                                     sample_sd=float(np.std(values, ddof=1)), seeds=[j["seed"] for j in cohort])
                           for mode in ("stochastic_z", "zero_z")}
    save(ROOT / "result.json", dict(status="completed", commit=manifest["commit"], jobs=jobs,
         aggregate=aggregate, window="900000 < env_steps <= 1000000; 20 evaluations x 10 episodes", completed=time.time()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("setup", "run"))
    args = parser.parse_args()
    (setup if args.command == "setup" else run)()
