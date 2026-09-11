"""Wait for the entire v3 sweep, freeze one best temperature, run 4x4 jobs."""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

SOURCE = Path(__file__).resolve().parents[1]
PYTHON = "/root/.venv-optiq-mujoco/bin/python"
TEMPERATURES = [.05, .1, .25, .5, 1.]
SEEDS = [0, 1, 2, 3]
VARIANTS = [("fixed_std015", "fixed", 0.), ("adaptive_hm09", "adaptive", -.9),
            ("adaptive_hm05", "adaptive", -.5), ("adaptive_h0", "adaptive", 0.)]


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity(source=SOURCE):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=source, text=True).strip()
    if git("status", "--porcelain"):
        raise ValueError("Experiment source must be a clean, committed snapshot")
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=source).decode().split("\0")
    return {"commit": git("rev-parse", "HEAD"),
            "hashes": {name: sha(source / name) for name in names if name}}


def dependency_hashes(path):
    return {str(p.relative_to(path)): sha(p) for p in sorted(path.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}


def completed_result(item, commit, project="v3_test", exploration=False):
    """Require complete per-run evidence, not just a final scalar in a state file."""
    import numpy as np
    configs = list(Path(item["output_root"]).glob("*/config.json"))
    if len(configs) != 1:
        raise ValueError(f"Expected exactly one fresh run for {item['id']}")
    directory = configs[0].parent
    cfg, done = read(configs[0]), read(directory / "completed.json")
    expected = item["resolved_config"]
    if (cfg["alg"] != expected["alg"] or cfg["seed"] != item["seed"]
            or cfg["env_name"] != "Ant-v4" or cfg["total_steps"] != 1_000_000
            or cfg["wandb"] != expected["wandb"]
            or cfg["runtime"]["git_commit"] != commit or cfg["runtime"]["git_dirty"]):
        raise ValueError(f"Runtime configuration/source mismatch: {item['id']}")
    if done["timesteps"] != 1_000_000 or done["updates"] != 995_000:
        raise ValueError(f"Incomplete training budget: {item['id']}")
    if not done["wandb_url"].startswith(f"https://wandb.ai/OptiQ/{project}/runs/"):
        raise ValueError("Unexpected W&B destination")
    for kind in ("actor", "critic"):
        checkpoints = list(directory.rglob(f"{kind}_state_1000000.msgpack"))
        if len(checkpoints) != 1 or checkpoints[0].stat().st_size == 0:
            raise ValueError(f"Missing final {kind} checkpoint")
    files = list(directory.rglob("evaluations.npz"))
    if len(files) != 1:
        raise ValueError("Expected one evaluations.npz")
    with np.load(files[0]) as archive:
        steps, returns = archive["timesteps"], archive["results"]
        tail = (steps >= 900_000) & (steps <= 1_000_000)
        if (not np.array_equal(steps[tail], np.arange(900_000, 1_000_001, 5000))
                or returns.shape != (len(steps), 10) or not np.isfinite(returns).all()
                or steps[-1] != 1_000_000):
            raise ValueError(f"Incomplete or invalid evaluations: {item['id']}")
        score = float(returns[tail].mean())
    result = {**done, "mean_return_900k_1m": score, "output": str(directory),
              "evaluation_file": str(files[0]), "evaluation_sha256": sha(files[0]),
              "config_sha256": sha(configs[0]), "completion_sha256": sha(directory / "completed.json")}
    if exploration:
        files = list(directory.rglob("exploration_state_1000000.json"))
        if len(files) != 1:
            raise ValueError("Missing final exploration checkpoint")
        extra = read(files[0])
        expected_updates = 99 if expected["alg"]["collection_exploration"]["mode"] == "adaptive" else 0
        if extra["alpha_controller"]["updates"] != expected_updates or extra["action_count"] != 995_000:
            raise ValueError("Unexpected exploration update/collection count")
        result["final_exploration_noise_std"] = extra["noise_std"]
    return result


def select_temperature(manifest, state):
    """Return None until ALL 20 runs complete; never choose per seed."""
    import numpy as np
    expected = {(t, s) for t in TEMPERATURES for s in SEEDS}
    jobs = manifest["jobs"]
    pairs = [(j["temperature"], j["seed"]) for j in jobs]
    if len(jobs) != 20 or len(set(pairs)) != 20 or set(pairs) != expected:
        raise ValueError("Baseline manifest must contain exactly the expected five temperatures and four seeds")
    if state.get("phase") != "completed" or any(
            state["jobs"].get(j["id"], {}).get("state") != "completed" for j in jobs):
        return None
    records = []
    for item in jobs:
        if state["jobs"][item["id"]].get("returncode") != 0:
            raise ValueError("Baseline run must exit successfully")
        records.append({"id": item["id"], "temperature": item["temperature"], "seed": item["seed"],
                        **completed_result(item, manifest["source_commit"])})
    scores = [{"temperature": t, "seed_scores": [next(r["mean_return_900k_1m"] for r in records
                if r["temperature"] == t and r["seed"] == s) for s in SEEDS]} for t in TEMPERATURES]
    for score in scores:
        score["mean"] = float(np.mean(score["seed_scores"]))
        score["sample_std"] = float(np.std(score["seed_scores"], ddof=1))
    winner = min(scores, key=lambda x: (-x["mean"], x["temperature"]))
    return {"selected_utc": now(), "temperature": winner["temperature"],
            "selection_metric": "mean over 4 seeds of mean return over 21 evaluations (900K:5K:1M), 10 episodes each",
            "tie_break": "lower temperature on exact equal means", "scope": "one common value for all 16 runs",
            "temperature_scores": scores, "baseline_records": records}


class Campaign:
    def __init__(self, root, baseline):
        self.root, self.baseline = Path(root).resolve(), Path(baseline).resolve()
        self.manifest_path = self.root / "manifest.json"
        self.state_path = self.root / "state.json"
        self.group = "Ant-v4_v3_collection_exploration_" + self.root.name.rsplit("_", 1)[-1]

    def environment(self, gpu=None):
        env = os.environ.copy()
        for name in ("WANDB_RUN_ID", "WANDB_RESUME", "WANDB_RUN_GROUP", "WANDB_NAME", "WANDB_PROJECT",
                     "JAX_PLATFORMS", "JAX_PLATFORM_NAME"):
            env.pop(name, None)
        env.update(OPTIQ_PYTHON=PYTHON, OPTIQ_ENV_FILE="/root/OptiQ/.env",
            OPTIQ_CONFIG="mujoco_v3_exploration", WANDB_ENTITY="OptiQ", WANDB_MODE="online",
            PYTHONPATH=str(self.root / "deps"), PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
            XLA_PYTHON_CLIENT_PREALLOCATE="false", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        return env

    def make_jobs(self, temperature):
        from scripts.verify_v3 import verify
        from omegaconf import OmegaConf
        base = OmegaConf.to_container(verify(["benchmark=ant"]), resolve=True)
        jobs = []
        for name, mode, target in VARIANTS:
            for seed in SEEDS:
                job_id = f"{name}_seed{seed}"
                output = self.root / "outputs" / job_id
                overrides = ["benchmark=ant", f"seed={seed}", f"alg.actor.temperature={temperature}",
                    f"alg.collection_exploration.mode={mode}",
                    f"alg.collection_exploration.target_entropy_per_dim={target}",
                    "wandb.project=v3_test", f"wandb.group={self.group}",
                    "wandb.job_type=v3-collection-exploration", f"run_name=ant-v3-explore-{name}-T{temperature}-s{seed}",
                    f"output_root={output}", "progress_bar=false"]
                cfg = OmegaConf.to_container(verify(overrides, "mujoco_v3_exploration"), resolve=True)
                expected = copy.deepcopy(base)
                expected["seed"] = seed
                expected["alg"]["actor"]["temperature"] = temperature
                expected["alg"]["collection_exploration"] = cfg["alg"]["collection_exploration"]
                for field in ("wandb", "run_name", "output_root", "progress_bar"):
                    expected[field] = cfg[field]
                if cfg != expected or cfg["total_steps"] != 1_000_000:
                    raise ValueError("Unexpected baseline algorithm or protocol change")
                jobs.append({"id": job_id, "variant": name, "mode": mode, "target_entropy_per_dim": target,
                    "seed": seed, "gpu": seed, "temperature": temperature, "output_root": str(output),
                    "console": str(self.root / "logs" / f"{job_id}.log"), "resolved_config": cfg,
                    "command": ["bash", str(SOURCE / "scripts/run_v3.sh"), str(seed), *overrides]})
        return jobs

    def prepare(self):
        if self.manifest_path.exists() or self.state_path.exists():
            raise ValueError("Refusing to overwrite an existing campaign")
        baseline = read(self.baseline / "manifest.json")
        if baseline["source_commit"] != "b41b024ae1747abc827c5b58a54b6cc62d4b2e44":
            raise ValueError("Unexpected baseline version")
        # Validate every complete config now. 0.1 is a validation placeholder;
        # no command is eligible to run until selected_jobs.json is generated.
        templates = self.make_jobs(.1)
        for directory in ("logs", "outputs"):
            (self.root / directory).mkdir(parents=True, exist_ok=True)
        manifest = {"created_utc": now(), "source": str(SOURCE), "source_identity": source_identity(),
            "dependency_hashes": dependency_hashes(self.root / "deps"), "baseline": str(self.baseline),
            "baseline_manifest_sha256": sha(self.baseline / "manifest.json"),
            "baseline_runner_sha256": sha(self.baseline / "run.py"),
            "entity": "OptiQ", "project": "v3_test", "group": self.group,
            "temperature": None, "selection": "all 20 completed first; equal 4-seed mean of 900K–1M evaluations",
            "same_temperature_all_seeds_and_variants": True,
            "jobs": [{k: j[k] for k in ("id", "variant", "mode", "target_entropy_per_dim", "seed", "gpu")}
                     for j in templates], "total_steps_per_job": 1_000_000, "total_requested_steps": 16_000_000,
            "performance_early_stop": False, "automatic_retry": False, "checkpoint_resume": False,
            "config_validation": "all 16 configurations passed; temperature remains unresolved"}
        save(self.manifest_path, manifest)
        save(self.state_path, {"phase": "waiting_for_all_baseline_runs", "created_utc": now(),
            "selected_temperature": None, "jobs": {j["id"]: {**j, "state": "waiting_for_baseline", "temperature": None}
            for j in manifest["jobs"]}})
        print(json.dumps({"registered_jobs": 16, "temperature": None, "waiting_for": "all 20 complete baseline runs"}), flush=True)

    def verify_frozen_inputs(self, manifest):
        if source_identity() != manifest["source_identity"]:
            raise ValueError("Frozen follow-up source changed")
        if dependency_hashes(self.root / "deps") != manifest["dependency_hashes"]:
            raise ValueError("Isolated exploration dependencies changed")
        if (sha(self.baseline / "manifest.json") != manifest["baseline_manifest_sha256"]
                or sha(self.baseline / "run.py") != manifest["baseline_runner_sha256"]):
            raise ValueError("Baseline campaign definition changed")

    def heartbeat(self, state, **values):
        state.update(updated_utc=now(), **values)
        save(self.state_path, state)

    def await_selection(self, manifest, state):
        selection_path = self.root / "selection.json"
        if selection_path.exists():
            selection = read(selection_path)
            if state.get("selection_sha256") != sha(selection_path):
                raise ValueError("Selection artifact changed or selection was interrupted")
            return selection
        last_reason = None
        while True:
            baseline_state = read(self.baseline / "state.json")
            counts = {status: sum(j["state"] == status for j in baseline_state["jobs"].values())
                      for status in ("running", "queued", "completed", "failed", "interrupted")}
            reason = f"Baseline complete {counts['completed']}/20, phase={baseline_state['phase']}"
            try:
                selection = select_temperature(read(self.baseline / "manifest.json"), baseline_state)
                if selection is not None:
                    self.verify_frozen_inputs(manifest)
                    save(selection_path, selection)
                    self.heartbeat(state, phase="temperature_selected", selected_temperature=selection["temperature"],
                                   selection_sha256=sha(selection_path), baseline_counts=counts)
                    print(f"SELECTED one common temperature: {selection['temperature']}", flush=True)
                    return selection
            except (KeyError, ValueError, OSError) as error:
                reason = f"Waiting for valid complete baseline evidence: {error}"
            self.heartbeat(state, phase="waiting_for_all_baseline_runs", waiting_reason=reason, baseline_counts=counts)
            if reason != last_reason:
                print(reason, flush=True)
                last_reason = reason
            time.sleep(30)

    def verify_launch(self, item, state):
        """Read W&B after launch; transient API errors never interrupt training."""
        entry = state["jobs"][item["id"]]
        if entry.get("wandb_verified") or time.time() - entry.get("last_verify_time", 0) < 120:
            return
        entry["last_verify_time"] = time.time()
        try:
            configs = list(Path(item["output_root"]).glob("*/config.json"))
            if len(configs) != 1:
                return
            cfg = read(configs[0])
            if cfg["alg"] != item["resolved_config"]["alg"] or cfg["seed"] != item["seed"]:
                raise ValueError("Launched configuration mismatch")
            with open(item["console"]) as stream:
                startup = stream.read(100_000)
            match = re.search(r"https://wandb\.ai/OptiQ/v3_test/runs/([A-Za-z0-9]+)", startup)
            if match is None:
                return
            from optiq_dime.runtime import load_environment
            import wandb
            load_environment()
            run = wandb.Api(timeout=10).run(f"OptiQ/v3_test/{match.group(1)}")
            history = run.history(samples=2, pandas=False)
            if (run.config.get("seed") != item["seed"] or run.group != self.group
                    or run.config.get("alg", {}).get("actor", {}).get("temperature") != item["temperature"]):
                raise ValueError("W&B run identity or selected temperature mismatch")
            if not history:
                return
            entry.update(wandb_verified=True, wandb_url=run.url, wandb_verified_utc=now())
            save(self.root / "launch_verification.json", {name: {key: value for key, value in j.items()
                if key.startswith("wandb_")} for name, j in state["jobs"].items()})
        except Exception as error:
            entry["wandb_verification_pending"] = str(error)

    def summarize(self, state):
        import numpy as np
        results = []
        for name, _, _ in VARIANTS:
            jobs = [j for j in state["jobs"].values() if j["variant"] == name]
            values = [j["mean_return_900k_1m"] for j in jobs if j["state"] == "completed"]
            results.append({"variant": name, "completed_seeds": len(values),
                "mean": float(np.mean(values)) if values else None,
                "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None, "jobs": jobs})
        save(self.root / "summary.json", {"phase": state["phase"], "updated_utc": now(),
             "temperature": state["selected_temperature"], "variants": results})

    def run(self):
        with (self.root / "controller.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            manifest, state = read(self.manifest_path), read(self.state_path)
            self.verify_frozen_inputs(manifest)
            if any(j["state"] in {"running", "interrupted"} for j in state["jobs"].values()):
                raise ValueError("Inspect interrupted jobs; never silently resume or duplicate training")
            self.heartbeat(state, controller_pid=os.getpid())
            selection = self.await_selection(manifest, state)
            selected_path = self.root / "selected_jobs.json"
            if selected_path.exists():
                if state.get("selected_jobs_sha256") != sha(selected_path):
                    raise ValueError("Selected jobs changed")
                jobs = read(selected_path)["jobs"]
            else:
                jobs = self.make_jobs(selection["temperature"])
                save(selected_path, {"temperature": selection["temperature"], "jobs": jobs})
                for job in jobs:
                    state["jobs"][job["id"]].update(state="queued", temperature=selection["temperature"])
                self.heartbeat(state, selected_jobs_sha256=sha(selected_path))
            if {j["temperature"] for j in jobs} != {selection["temperature"]} or len(jobs) != 16:
                raise ValueError("All 16 jobs must use the same selected temperature")
            processes = {}
            self.heartbeat(state, phase="running", started_utc=now())
            def interrupted(signum, frame):
                raise KeyboardInterrupt(f"Controller interrupted: {signum}")
            signal.signal(signal.SIGTERM, interrupted)
            try:
                while True:
                    for name, (process, item) in list(processes.items()):
                        self.verify_launch(item, state)
                        code = process.poll()
                        if code is None:
                            continue
                        entry = state["jobs"][name]
                        entry.update(returncode=code, finished_utc=now())
                        try:
                            if code != 0:
                                raise ValueError(f"Training exited with {code}")
                            entry.update(state="completed", **completed_result(
                                item, manifest["source_identity"]["commit"], exploration=True))
                        except (ValueError, KeyError, OSError) as error:
                            entry.update(state="failed", error=str(error))
                        del processes[name]
                        print(f"END {name}: {entry['state']}", flush=True)
                        self.summarize(state)
                    busy = {item["gpu"] for _, item in processes.values()}
                    for gpu in SEEDS:
                        if gpu in busy:
                            continue
                        pending = [j for j in jobs if j["gpu"] == gpu and state["jobs"][j["id"]]["state"] == "queued"]
                        if not pending:
                            continue
                        occupied = subprocess.check_output(["nvidia-smi", "-i", str(gpu),
                            "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
                        if occupied:
                            continue
                        self.verify_frozen_inputs(manifest)
                        item = pending[0]
                        with open(item["console"], "x", buffering=1) as stream:
                            process = subprocess.Popen(item["command"], cwd=SOURCE,
                                env=self.environment(gpu), stdout=stream, stderr=subprocess.STDOUT)
                        processes[item["id"]] = (process, item)
                        state["jobs"][item["id"]].update(state="running", pid=process.pid, started_utc=now())
                        print(f"START {item['id']}: GPU {gpu}, T={item['temperature']}, pid={process.pid}", flush=True)
                    self.heartbeat(state)
                    if not processes and not any(j["state"] == "queued" for j in state["jobs"].values()):
                        break
                    time.sleep(10)
            except BaseException:
                for process, _ in processes.values():
                    if process.poll() is None:
                        process.terminate()
                for name, (process, _) in processes.items():
                    try:
                        process.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    state["jobs"][name].update(state="interrupted", finished_utc=now())
                self.heartbeat(state, phase="interrupted")
                raise
            failures = [n for n, j in state["jobs"].items() if j["state"] != "completed"]
            self.heartbeat(state, phase="finished_with_failures" if failures else "completed", finished_utc=now())
            self.summarize(state)
            print(f"FINISHED: {16-len(failures)}/16 completed; failures={failures}", flush=True)
            if failures:
                raise SystemExit(1)


if __name__ == "__main__":
    sys.path.insert(0, str(SOURCE))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--baseline", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    campaign = Campaign(args.root, args.baseline)
    campaign.prepare() if args.prepare else campaign.run()
