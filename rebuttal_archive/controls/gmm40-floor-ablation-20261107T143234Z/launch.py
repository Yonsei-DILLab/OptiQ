import sys,json,subprocess,importlib.metadata
from pathlib import Path
c=json.loads(Path(sys.argv[1]).read_text());host=sys.argv[2]
root=Path(c['root']);root.mkdir(parents=True,exist_ok=True);code=Path(c['code'])
subprocess.run(['git','-C','/workspace/iBOLT','worktree','add','--detach','--no-checkout',str(code),c['revision']],check=True)
subprocess.run(['git','-C',str(code),'sparse-checkout','set','--no-cone','/gmm40/','/ibolt/','/models/'],check=True)
subprocess.run(['git','-C',str(code),'checkout','--detach',c['revision']],check=True)
assert subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()==c['revision']
subprocess.run(['git','-C',str(code),'diff','--exit-code'],check=True)
(root/'environment.json').write_text(json.dumps(dict(revision=c['revision'],packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version','--format=csv,noheader'],text=True)),indent=2)+'\n')
(root/'campaign.json').write_text(json.dumps(c,indent=2)+'\n')
wrapper=root/'run.sh'
wrapper.write_text('''#!/usr/bin/env bash
set -euo pipefail
gpu="$1"
seed="$2"
output="$3"
mode="$4"
cd '''+str(code)+'''
source /workspace/iBOLT/scripts/activate-experiments.sh
export CUDA_VISIBLE_DEVICES="$gpu" JAX_PLATFORMS=cuda
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
extra=()
if [[ "$mode" == "floor_off_check" ]]; then extra+=(--no-proposal-floor); fi
[[ ! -e "$output" ]] || exit 2
python -u -m gmm40.run --seed "$seed" --nm 256 --steps 100000 --batch 256 --eval-samples 10000 --log-std-max -3.5 --hidden-init orthogonal_sqrt2 "${extra[@]}" --output "$output" 2>&1 | tee -a "${output}.log"
''');wrapper.chmod(0o755)
for j in c['jobs']:
 if j['host']!=host:continue
 Path(j['output']).parent.mkdir(parents=True,exist_ok=True)
 name=j['program']
 cfg=f'''[program:{name}]
command={wrapper} {j['gpu']} {j['seed']} {j['output']} {j['role']}
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
 if j['host']!=host:continue
 subprocess.run(['sudo','supervisorctl','update',j['program']],check=True)
 subprocess.run(['sudo','supervisorctl','start',j['program']],check=True)
