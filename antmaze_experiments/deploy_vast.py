"""Register only this committed 27-run queue, preserving other services."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess

SOURCE=Path(__file__).resolve().parents[1]
ROOT='/workspace/antmaze-temperature-20260924'
HOSTS=('vast2','vast3','vast4','vast5')
sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()
assert not subprocess.check_output(['git','status','--porcelain'],cwd=SOURCE).strip()
bundle=Path('/tmp')/f'antmaze-{sha}.bundle'
subprocess.run(['git','bundle','create',str(bundle),'HEAD'],cwd=SOURCE,check=True)

REMOTE=r"""
import os,json,subprocess,sys
from pathlib import Path
host,sha=sys.argv[1:]
root=Path('/workspace/antmaze-temperature-20260924')
root.mkdir(exist_ok=True)
source=root/'code'
if not source.exists():
    subprocess.run(['git','clone','/tmp/antmaze-'+sha+'.bundle',str(source)],check=True)
    subprocess.run(['git','checkout','--detach',sha],cwd=source,check=True)
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()==sha
base=('/workspace/optiq-ant-highbeta-20260916/venv/bin/python' if host=='vast5'
      else '/workspace/optiq-v5-comparison-20260913/venv/bin/python')
if not (root/'manifest.json').exists():
    subprocess.run([base,'-m','antmaze_experiments.vast_queue','register','--host',host],cwd=source,check=True)
name='antmaze-temperature-20260924'
conf=Path('/etc/supervisor/conf.d')/(name+'.conf')
with conf.open('x') as f:
    f.write(f'''[program:{name}]
command=/bin/bash {source}/antmaze_experiments/vast_bootstrap.sh
directory={source}
environment=ANTMAZE_BASE_PYTHON="{base}",PYTHONDONTWRITEBYTECODE="1"
autostart=true
autorestart=false
startsecs=2
startretries=0
stopasgroup=true
killasgroup=true
stopwaitsecs=60
redirect_stderr=true
stdout_logfile={root}/controller.log
stdout_logfile_maxbytes=20MB
stdout_logfile_backups=2
''')
subprocess.run(['supervisorctl','reread'],check=True)
subprocess.run(['supervisorctl','update',name],check=True)
print(json.dumps(dict(host=host,source_commit=sha,jobs=len(json.loads((root/'manifest.json').read_text())['jobs']))))
"""

def deploy(host):
    subprocess.run(['scp',str(bundle),host+':/tmp/'+bundle.name],check=True)
    r=subprocess.run(['ssh',host,'sudo','-n','python3','-',host,sha],input=REMOTE,text=True,capture_output=True)
    print(host,r.stdout,r.stderr,flush=True)
    if r.returncode:raise RuntimeError(host)

with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(deploy,HOSTS))
