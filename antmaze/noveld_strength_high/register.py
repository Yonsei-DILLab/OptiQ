"""Register exactly one new managed campaign from committed frozen source."""
import fcntl
import json
from pathlib import Path
import subprocess
from .campaign import NAME, OPS, PYTHON, source_sha

source,sha=source_sha()
assert source==OPS/'sources'/sha,'Registration must use the frozen commit worktree'
root=Path('/home/heechan/optiq-experiments')/NAME
conf=OPS/'supervisor'/'jobs'/(NAME+'.conf')
assert not root.exists() and not conf.exists(),'Refuse duplicate registration'
for gpu in range(2):
    with (OPS/'locks'/f'gpu-{gpu}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(lock,fcntl.LOCK_UN)
assert not subprocess.check_output(['nvidia-smi','--id=0,1','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
root.mkdir();(root/'logs').mkdir()
command=[PYTHON,'-u','-m','antmaze.noveld_strength_high.campaign','--root',str(root)]
conf.write_text(f'''[program:{NAME}]
command={' '.join(command)}
directory={source}
environment=OPTIQ_SOURCE_DIR="{source}",PYTHONPATH="{source}",OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",XLA_PYTHON_CLIENT_PREALLOCATE="false"
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
(root/'registration.json').write_text(json.dumps(dict(source_commit=sha,source=str(source),
    service=NAME,command=command,autostart=False,autorestart=False),indent=2)+'\n')
ctl=['/usr/local/bin/supervisorctl','-c',str(OPS/'supervisor/supervisord.conf')]
subprocess.run(ctl+['reread'],check=True)
subprocess.run(ctl+['update',NAME],check=True)
subprocess.run(ctl+['start',NAME],check=True)
print(json.dumps(dict(registered=True,source_commit=sha,root=str(root),service=NAME)))
