"""Register only the user-requested inference audit on server 180's free GPUs."""
import fcntl
import json
from pathlib import Path
import subprocess

source = Path(__file__).resolve().parents[2]
sha = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
assert not subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
root = Path("/home/heechan/optiq-experiments/antmaze-route-audit-20260922")
assert not root.exists(), "Refuse duplicate diagnostic registration"
ops = Path("/home/heechan/OptiQ-ops")
training = ops / "sources" / "19fc37a7e51be2bb54d41ab9a980225ecd9f5ba5"
methods = ("optiq", "sac", "mfpo", "meow")
for gpu in range(4):
    with (ops / "locks" / f"gpu-{gpu}.lock").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(f, fcntl.LOCK_UN)
root.mkdir()
(root / "logs").mkdir()
jobs = []
for gpu, method in enumerate(methods):
    name = f"antmaze-route-audit-20260922-{method}"
    config = ops / "supervisor" / "jobs" / f"{name}.conf"
    assert not config.exists()
    command = [str(ops / "run-gpu.sh"), str(gpu), "--branch", "v5-direct-gmm",
        "/home/heechan/.venv-optiq-antmaze/bin/python", "-u", str(source / "antmaze/route_diagnostics/rollout_audit.py"),
        "--method", method, "--output", str(root / method), "--diagnostic-source-sha", sha]
    config.write_text(f"""[program:{name}]
command={' '.join(command)}
directory={training}
environment=OPTIQ_SOURCE_DIR="{training}",PYTHONPATH="{training}",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",XLA_PYTHON_CLIENT_PREALLOCATE="false"
autostart=false
autorestart=false
startsecs=5
startretries=0
stopasgroup=true
killasgroup=true
redirect_stderr=true
stdout_logfile={root}/logs/{method}.log
stdout_logfile_maxbytes=0
""")
    jobs.append(dict(method=method, gpu=gpu, service=name, command=command, config=str(config)))
(root / "manifest.json").write_text(json.dumps(dict(diagnostic_source=sha,
    training_source=training.name, source_root=str(source), inference_only=True,
    production_jobs_unchanged=True, jobs=jobs), indent=2)+"\n")
ctl = ["/usr/local/bin/supervisorctl", "-c", str(ops / "supervisor/supervisord.conf")]
subprocess.run(ctl+["reread"], check=True)
subprocess.run(ctl+["update"]+[j["service"] for j in jobs], check=True)
subprocess.run(ctl+["start"]+[j["service"] for j in jobs], check=True)
print(json.dumps(dict(registered=True, diagnostic_source=sha, jobs=jobs), indent=2))
