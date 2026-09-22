"""Register the single authorized control; never register the held 16-run queue."""
import json
from pathlib import Path
import subprocess
from .campaign import NAME, PYTHON, UPSTREAM, source_sha


def main():
    source, sha = source_sha()
    ops = Path("/home/heechan/OptiQ-ops")
    assert source == ops / "sources" / sha
    dependency = source / "gmm40-baseline/DIPO"
    assert (dependency / "agent/DiPo.py").is_file()
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=dependency, text=True).strip()
    assert actual == UPSTREAM
    root = Path("/home/heechan/optiq-experiments") / NAME
    conf = ops / "supervisor/jobs" / (NAME + ".conf")
    assert not root.exists() and not conf.exists(), "Refuse duplicate registration"
    root.mkdir()
    command = [PYTHON, "-u", "-m", "antmaze.v1_dipo_noveld001.campaign", "--root", str(root)]
    conf.write_text(f'''[program:{NAME}]
command={' '.join(command)}
directory={source}
environment=OPTIQ_SOURCE_DIR="{source}",PYTHONPATH="{source}",PYTHONUNBUFFERED="1",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",XLA_PYTHON_CLIENT_PREALLOCATE="false"
autostart=false
autorestart=false
startsecs=5
startretries=0
stopasgroup=true
killasgroup=true
redirect_stderr=true
stdout_logfile={root}/controller.log
stdout_logfile_maxbytes=0
''')
    registration = dict(source_commit=sha, service=NAME, root=str(root), command=command,
        total_runs=1, task="v1", method="dipo", coefficient=.01, seed=0, steps=1000000,
        dependencies={"DIPO": actual}, autostart=False, autorestart=False)
    (root / "registration.json").write_text(json.dumps(registration, indent=2) + "\n")
    ctl = ["/usr/local/bin/supervisorctl", "-c", str(ops / "supervisor/supervisord.conf")]
    for tail in (["reread"], ["update", NAME], ["start", NAME]):
        subprocess.run(ctl + tail, check=True)
    print(json.dumps(registration))


if __name__ == "__main__":
    main()
