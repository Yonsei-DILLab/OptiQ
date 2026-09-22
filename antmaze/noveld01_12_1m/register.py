"""Register a committed six-job shard with independent GPU backfill."""
import argparse
import json
from pathlib import Path
import subprocess
from .campaign import NAME, PYTHON, SHARDS, source_sha


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--shard", type=int, choices=(0, 1), required=True)
    a = p.parse_args()
    source, sha = source_sha()
    ops = Path("/home/heechan/OptiQ-ops")
    assert source == ops / "sources" / sha
    dependencies = {}
    for relative, entry in (("gmm40-baseline/DIPO", "agent/DiPo.py"), ("gmm40-baseline/MFPO", "jaxrl5/agents/mean_flow_learner.py")):
        dependency = source / relative
        assert (dependency / entry).is_file(), relative
        expected = subprocess.check_output(["git", "ls-tree", "HEAD", relative], cwd=source, text=True).split()[2]
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=dependency, text=True).strip()
        assert actual == expected
        dependencies[relative] = actual
    root = Path("/home/heechan/optiq-experiments") / NAME
    conf = ops / "supervisor/jobs" / (NAME + ".conf")
    assert not root.exists() and not conf.exists(), "Refuse duplicate registration"
    root.mkdir()
    command = [PYTHON, "-u", "-m", "antmaze.noveld01_12_1m.campaign", "--root", str(root), "--shard", str(a.shard)]
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
    record = dict(source_commit=sha, root=str(root), service=NAME, shard=a.shard,
        command=command, jobs=SHARDS[a.shard], dependencies=dependencies,
        autostart=False, autorestart=False, backfill_seconds=2, maze_barrier=False)
    (root / "registration.json").write_text(json.dumps(record, indent=2) + "\n")
    ctl = ["/usr/local/bin/supervisorctl", "-c", str(ops / "supervisor/supervisord.conf")]
    for tail in (["reread"], ["update", NAME], ["start", NAME]):
        subprocess.run(ctl + tail, check=True)
    print(json.dumps(record))


if __name__ == "__main__":
    main()
