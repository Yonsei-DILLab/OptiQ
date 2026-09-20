"""Launch exactly four independent Humanoid fixed64 seeds with frozen source."""
from datetime import datetime,timezone
import json,hashlib,shlex,socket,subprocess,sys
from pathlib import Path
repo=Path(__file__).resolve().parents[2];ops=Path('/home/heechan/OptiQ-ops');supervisor=ops/'supervisor';conf=supervisor/'supervisord.conf'
def git(*args):return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()
assert git('branch','--show-current')=='v5-direct-gmm'
assert not git('status','--porcelain'), 'Commit exact experiment before launch'
assert [int(x) for x in (ops/'gpus.txt').read_text().split()]==[0,1,2,3]
sha=git('rev-parse','HEAD');stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');source=ops/'sources'/sha
if not source.exists():
 subprocess.run(['git','clone','--local','--no-hardlinks','--no-checkout',str(repo),str(source)],check=True)
 subprocess.run(['git','-C',str(source),'checkout','--detach',sha],check=True)
assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==sha
assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip()
campaign=Path('/home/heechan/optiq-experiments')/f'humanoid-direct-gmm-fixed64-{stamp}';campaign.mkdir(exist_ok=False)
jobs=[]
for seed in range(4):
 name=f'hum-dgmm-f64-{stamp}-s{seed}'
 cmd=[str(ops/'run-gpu.sh'),str(seed),'--branch','v5-direct-gmm','python','run_optiq_dime.py','--config-name=mujoco_v5_direct_gmm_humanoid_fixed64',f'seed={seed}','wandb.project=v5-heechan-gmm',f'run_name={name}',f'output_root={campaign/"runs"}',f'+campaign_source_commit={sha}']
 jobs.append(dict(name=name,seed=seed,gpu=seed,command=cmd,log=str(supervisor/'logs'/f'{name}.log')))
freeze=ops/'requirements-v5.freeze.txt'
manifest=dict(source_commit=sha,source_branch='v5-direct-gmm',source=str(source),host=socket.gethostname(),created_utc=stamp,campaign=str(campaign),seeds=[0,1,2,3],config_name='mujoco_v5_direct_gmm_humanoid_fixed64',jobs=jobs,dependency_freeze=freeze.read_text(),dependency_sha256=hashlib.sha256(freeze.read_bytes()).hexdigest())
(campaign/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
assert conf.exists()
subprocess.run(['supervisorctl','-c',str(conf),'pid'],check=True)
for job in jobs:
 (supervisor/'jobs'/f"{job['name']}.conf").write_text(f'''[program:{job['name']}]
command={shlex.join(job['command']).replace('%','%%')}
directory={source}
environment=OPTIQ_SOURCE_DIR="{source}"
autostart=false
autorestart=false
startsecs=5
startretries=0
stopasgroup=true
killasgroup=true
redirect_stderr=true
stdout_logfile={job['log']}
stdout_logfile_maxbytes=100MB
stdout_logfile_backups=3
''')
subprocess.run(['supervisorctl','-c',str(conf),'reread'],check=True)
subprocess.run(['supervisorctl','-c',str(conf),'update',*[j['name'] for j in jobs]],check=True)
for job in jobs:subprocess.run(['supervisorctl','-c',str(conf),'start',job['name']],check=True)
print(json.dumps(dict(manifest=str(campaign/'manifest.json'),source_commit=sha,jobs=[j['name'] for j in jobs]),indent=2),flush=True)
