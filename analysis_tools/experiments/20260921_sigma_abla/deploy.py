"""Install this clean frozen source in the existing user Supervisor."""
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
host=sys.argv[1]
assert host in ('180','199')
assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
root=Path('/home/heechan/optiq-experiments/trg-sigma-abla-20260921')
root.mkdir(exist_ok=False)
(root/'manifest.json').write_text(json.dumps(dict(commit=sha,source=str(REPO),host=host,
    sigmas=[.1,.2,.01,.05],seed=0,total_steps=10000),indent=2))
names=[]
for gpu in range(4):
    name=f'trg-sigma-abla-20260921-gpu{gpu}'
    names.append(name)
    path=Path('/home/heechan/OptiQ-ops/supervisor/jobs')/(name+'.conf')
    assert not path.exists()
    path.write_text(f'''[program:{name}]
command=/home/heechan/OptiQ-ops/run-gpu.sh {gpu} --branch v5-direct-gmm /home/heechan/.venv-optiq-mujoco/bin/python -u {HERE}/queue.py {host} {gpu} {root}
directory={REPO}
autostart=true
autorestart=false
startsecs=3
startretries=0
stopasgroup=true
killasgroup=true
stopwaitsecs=45
redirect_stderr=true
stdout_logfile={root}/queue-gpu{gpu}.log
stdout_logfile_maxbytes=10MB
stdout_logfile_backups=2
environment=OPTIQ_SOURCE_DIR="{REPO}",PYTHONDONTWRITEBYTECODE="1",WANDB_MODE="online"
''')
ctl=['supervisorctl','-c','/home/heechan/OptiQ-ops/supervisor/supervisord.conf']
subprocess.run(ctl+['reread'],check=True)
for name in names:subprocess.run(ctl+['update',name],check=True)
print(json.dumps(dict(root=str(root),commit=sha,services=names)))
