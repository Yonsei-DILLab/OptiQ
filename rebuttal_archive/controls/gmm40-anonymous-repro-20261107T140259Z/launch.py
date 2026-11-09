import json,sys,subprocess,hashlib,importlib.metadata
from pathlib import Path
c=json.loads(Path(sys.argv[1]).read_text());host=sys.argv[2]
root=Path(c['root']);root.mkdir(parents=True,exist_ok=True)
code=Path(c['code']);code.mkdir(parents=True,exist_ok=True)
subprocess.run(['tar','-xzf','/tmp/ibolt-anonymous-published-source.tgz','-C',str(code)],check=True)
packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version','--format=csv,noheader'],text=True)
(root/'environment.json').write_text(json.dumps(dict(packages=packages,gpus=gpus,source_revision=c['revision'],source_archive_sha256=hashlib.sha256(Path('/tmp/ibolt-anonymous-published-source.tgz').read_bytes()).hexdigest()),indent=2)+'\n')
(root/'campaign.json').write_text(json.dumps(c,indent=2)+'\n')
wrapper=root/'run.sh'
wrapper.write_text('''#!/usr/bin/env bash
set -euo pipefail
gpu="$1"
seed="$2"
output="$3"
cd '''+c['code']+'''
source /workspace/iBOLT/scripts/activate-experiments.sh
export CUDA_VISIBLE_DEVICES="$gpu" JAX_PLATFORMS=cuda
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
[[ ! -e "$output" ]] || exit 2
python -u -m gmm40.run --seed "$seed" --nm 256 --steps 100000 --batch 256 --eval-samples 10000 --output "$output" 2>&1 | tee -a "${output}.log"
''');wrapper.chmod(0o755)
for j in c['jobs']:
 if j['host']!=host: continue
 Path(j['output']).parent.mkdir(parents=True,exist_ok=True)
 name=j['program']
 cfg=f'''[program:{name}]
command={wrapper} {j['gpu']} {j['seed']} {j['output']}
directory={code}
user=heechan
environment=HOME="/home/heechan",USER="heechan",PROC_NAME="%(program_name)s"
autostart=false
autorestart=false
exitcodes=0
startsecs=5
startretries=0
stopasgroup=true
killasgroup=true
stopsignal=TERM
stopwaitsecs=30
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stdout_logfile_backups=0
redirect_stderr=true
'''
 subprocess.run(['sudo','tee',f'/etc/supervisor/conf.d/{name}.conf'],input=cfg,text=True,stdout=subprocess.DEVNULL,check=True)
subprocess.run(['sudo','supervisorctl','reread'],check=True)
for j in c['jobs']:
 if j['host']!=host: continue
 subprocess.run(['sudo','supervisorctl','update',j['program']],check=True)
 subprocess.run(['sudo','supervisorctl','start',j['program']],check=True)
