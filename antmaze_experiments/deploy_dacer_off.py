"""Deploy a distinct immutable snapshot and pending-only controllers."""
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor
SOURCE=Path(__file__).resolve().parents[1]
sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()
assert not subprocess.check_output(['git','status','--porcelain'],cwd=SOURCE).strip()
bundle=Path('/tmp')/f'antmaze-off-{sha}.bundle'
subprocess.run(['git','bundle','create',str(bundle),'HEAD'],cwd=SOURCE,check=True)
REMOTE=r'''
import json,subprocess,sys
from pathlib import Path
host,sha=sys.argv[1:]
root=Path('/workspace/antmaze-dacer-off-20260924');root.mkdir(exist_ok=True)
old=Path('/workspace/antmaze-temperature-20260924');source=root/'code'
subprocess.run(['git','clone','/tmp/antmaze-off-'+sha+'.bundle',str(source)],check=True)
subprocess.run(['git','checkout','--detach',sha],cwd=source,check=True)
for name in ('venv','mujoco210'):(root/name).symlink_to(old/name,target_is_directory=True)
python=str(root/'venv/bin/python')
subprocess.run([python,'-m','antmaze_experiments.dacer_off_queue','register','--host',host],cwd=source,check=True)
name='antmaze-dacer-off-20260924'
with (Path('/etc/supervisor/conf.d')/(name+'.conf')).open('x') as f:
 f.write(f"""[program:{name}]
command={python} -m antmaze_experiments.dacer_off_queue run
directory={source}
environment=PYTHONDONTWRITEBYTECODE="1",PYTHONUNBUFFERED="1"
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
""")
subprocess.run(['supervisorctl','reread'],check=True)
subprocess.run(['supervisorctl','update',name],check=True)
print('REGISTERED',host,len(json.loads((root/'manifest.json').read_text())['jobs']),sha)
'''
def deploy(host):
 subprocess.run(['scp',str(bundle),host+':/tmp/'+bundle.name],check=True)
 r=subprocess.run(['ssh',host,'sudo','-n','python3','-',host,sha],input=REMOTE,text=True,capture_output=True)
 print(host,r.stdout,r.stderr,flush=True)
 if r.returncode:raise RuntimeError(host)
with ThreadPoolExecutor(4) as pool:list(pool.map(deploy,['vast2','vast3','vast4','vast5']))
