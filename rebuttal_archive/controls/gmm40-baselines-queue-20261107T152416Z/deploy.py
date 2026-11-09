"""Install only this campaign's supervisor entries; never restart prerequisite work."""
import argparse,json,subprocess,hashlib
from pathlib import Path

parser=argparse.ArgumentParser();parser.add_argument('manifest',type=Path);parser.add_argument('host');args=parser.parse_args()
c=json.loads(args.manifest.read_text());node=next(n for n in c['nodes'] if n['host']==args.host)
root=Path(c['root']);root.mkdir(parents=True,exist_ok=True);code=Path(c['code_root'])
if not code.exists():
 subprocess.run(['git','-C','/workspace/iBOLT','worktree','add','--detach','--no-checkout',str(code),c['revision']],check=True)
 subprocess.run(['git','-C',str(code),'sparse-checkout','set','--no-cone','/gmm40/','/ibolt/','/models/','/scripts/activate-experiments.sh','/scripts/run-gmm40-baseline.sh','/.gitignore'],check=True)
 subprocess.run(['git','-C',str(code),'checkout','--detach',c['revision']],check=True)
 if not (code/'.venv').exists():(code/'.venv').symlink_to('/workspace/iBOLT/.venv',target_is_directory=True)
assert subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()==c['revision']
subprocess.run(['git','-C',str(code),'diff','--exit-code'],check=True)
helper=root/'queue_service.py'
source=Path(__file__).with_name('queue_service.py').read_bytes()
if helper.exists():assert helper.read_bytes()==source,'Existing scheduler source differs'
else:helper.write_bytes(source)
config=dict(campaign=c['campaign'],host=args.host,repository_root='/workspace/iBOLT',allowed_branch='heechan-rebuttal',code_root=c['code_root'],revision=c['revision'],state_file=node['state'],jobs=[j for j in c['jobs'] if j['host']==args.host],prerequisites=node['prerequisites'],blocking_program_prefixes=['pointmaze-'],poll_seconds=30,required_idle_polls=2,max_idle_memory_mib=256,max_idle_utilization=5,gpu_lock_directory='/workspace/.gmm40-gpu-queue-locks',scheduler_sha256=hashlib.sha256(source).hexdigest())
Path(node['config']).write_text(json.dumps(config,indent=2)+'\n')
(root/'campaign.json').write_text(json.dumps(c,indent=2)+'\n')

def install(name,command,directory,autostart,autorestart,startsecs):
 text=f'''[program:{name}]
command={command}
directory={directory}
user=heechan
environment=HOME="/home/heechan",USER="heechan",PYTHONUNBUFFERED="1",PROC_NAME="%(program_name)s"
autostart={autostart}
autorestart={autorestart}
exitcodes=0
startsecs={startsecs}
startretries=0
stopasgroup=true
killasgroup=true
stopwaitsecs=30
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stdout_logfile_backups=0
redirect_stderr=true
'''
 dest=f'/etc/supervisor/conf.d/{name}.conf'
 subprocess.run(['sudo','-n','tee',dest],input=text,text=True,stdout=subprocess.DEVNULL,check=True)

for j in config['jobs']:
 Path(j['output']).parent.mkdir(parents=True,exist_ok=True)
 command=f"{code}/scripts/run-gmm40-baseline.sh {j['gpu']} {j['baseline']} {j['seed']} {j['output']} {j['actor_updates']}"
 install(j['program'],command,code,'false','false',5)
subprocess.run(['sudo','-n','supervisorctl','reread'],check=True)
for j in config['jobs']:
 subprocess.run(['sudo','-n','supervisorctl','update',j['program']],check=True)
# Confirm the gates before installing the automatically starting controller.
subprocess.run(['/workspace/iBOLT/.venv/bin/python',str(helper),'--config',node['config'],'--once','--dry-run'],check=True)
dry=Path(node['state']).with_name(Path(node['state']).stem+'.dry-run.json')
r=json.loads(dry.read_text())
assert all(j['status']=='queued' for j in r['jobs']),'Dry-run found an unexpected running job'
install(node['controller'],f'/workspace/iBOLT/.venv/bin/python -u {helper} --config {node["config"]}',code,'true','unexpected',2)
subprocess.run(['sudo','-n','supervisorctl','reread'],check=True)
subprocess.run(['sudo','-n','supervisorctl','update',node['controller']],check=True)
print(json.dumps(dict(host=args.host,controller=node['controller'],jobs_registered=len(config['jobs']),initial_state='waiting',state_file=node['state'])),flush=True)
