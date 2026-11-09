import json,sys,subprocess
from pathlib import Path
c=json.loads(Path(sys.argv[1]).read_text());host=sys.argv[2]
Path(c['root']).mkdir(parents=True,exist_ok=True)
for j in c['jobs']:
 if j['host']!=host:continue
 name=j['program']
 cfg=f'''[program:{name}]
command=/workspace/iBOLT/scripts/run-gmm40-baseline.sh {j['gpu']} {j['baseline']} {j['seed']} {j['output']} {c['actor_updates']}
directory=/workspace/iBOLT
user=heechan
environment=HOME="/home/heechan",USER="heechan",PROC_NAME="%(program_name)s"
autostart=false
autorestart=false
exitcodes=0
startsecs=5
startretries=0
stopasgroup=true
killasgroup=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
redirect_stderr=true
'''
 subprocess.run(['sudo','tee',f'/etc/supervisor/conf.d/{name}.conf'],input=cfg,text=True,stdout=subprocess.DEVNULL,check=True)
 subprocess.run(['sudo','supervisorctl','reread'],check=True)
 subprocess.run(['sudo','supervisorctl','update',name],check=True)
 subprocess.run(['sudo','supervisorctl','start',name],check=True)
