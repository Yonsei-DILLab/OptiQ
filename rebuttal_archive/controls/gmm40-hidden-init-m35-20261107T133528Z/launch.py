import json,subprocess,sys
from pathlib import Path
c=json.loads(Path(sys.argv[1]).read_text())
host=sys.argv[2]
root=Path(c['root']); root.mkdir(parents=True,exist_ok=True)
(root/'campaign.json').write_text(json.dumps(c,indent=2)+'\n')
for j in c['jobs']:
 if j['host']!=host: continue
 Path(j['output']).parent.mkdir(parents=True,exist_ok=True)
 name=j['program']
 cfg=f'''[program:{name}]
command=/workspace/iBOLT/scripts/run-gmm40-no-floor.sh {j['gpu']} {j['seed']} {j['output']} -3.5 {j['hidden_init']}
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
 if Path(j['output']).exists(): continue
 subprocess.run(['sudo','supervisorctl','update',j['program']],check=True)
 subprocess.run(['sudo','supervisorctl','start',j['program']],check=True)
