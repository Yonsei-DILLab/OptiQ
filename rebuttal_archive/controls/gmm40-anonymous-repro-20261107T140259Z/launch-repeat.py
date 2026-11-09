import json,sys,subprocess,hashlib,importlib.metadata
from pathlib import Path
c=json.loads(Path(sys.argv[1]).read_text());host=sys.argv[2];root=Path(c['root'])
old=json.loads((root/'environment.json').read_text())
packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version','--format=csv,noheader'],text=True)
assert packages==old['packages'],'Package versions changed between runs'
assert gpus==old['gpus'],'Physical GPUs or drivers changed between runs'
assert hashlib.sha256(Path('/tmp/ibolt-anonymous-published-source.tgz').read_bytes()).hexdigest()==old['source_archive_sha256']
(root/'environment-repeat.json').write_text(json.dumps(dict(packages=packages,gpus=gpus,source_revision=c['revision'],source_archive_sha256=old['source_archive_sha256']),indent=2)+'\n')
(root/'campaign.json').write_text(json.dumps(c,indent=2)+'\n')
for j in c['jobs']:
 if j['host']!=host or j['round']!=2:continue
 assert (Path(j['baseline_output'])/'completed.json').exists(),'First run is not complete'
 Path(j['output']).parent.mkdir(parents=True,exist_ok=True)
 name=j['program']
 cfg=f'''[program:{name}]
command={root}/run.sh {j['gpu']} {j['seed']} {j['output']}
directory={c['code']}
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
 if j['host']!=host or j['round']!=2:continue
 subprocess.run(['sudo','supervisorctl','update',j['program']],check=True)
 subprocess.run(['sudo','supervisorctl','start',j['program']],check=True)
