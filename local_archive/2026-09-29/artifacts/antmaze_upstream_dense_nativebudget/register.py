"""Register exactly the approved dense campaign; fail rather than overwrite."""
import json, subprocess, sys, time
from pathlib import Path
shard=int(sys.argv[1])
sha='a946a77226629688346d6f4c69952131fb528ed0'
name='antmaze-upstream-dense-nativebudget-64env-s0-20260923'
source=Path('/home/heechan/OptiQ-ops/sources')/sha
root=Path('/home/heechan/optiq-experiments')/name
ctl=['/usr/local/bin/supervisorctl','-c','/home/heechan/OptiQ-ops/supervisor/supervisord.conf']
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()==sha
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=source,text=True).strip()
audit=json.loads(Path('/home/heechan/optiq-experiments/antmaze-dense-nativebudget-env-audit-20260923.json').read_text())
assert audit['passed'] and audit['replay_wrap_verified']
old='antmaze-upstream-64env-1m-s0-20260923'
status=subprocess.run(ctl+['status',old],text=True,capture_output=True)
assert 'STOPPED' in status.stdout,status.stdout
oldroot=root.parent/old
oldstatus=json.loads((oldroot/'status.json').read_text())
for run in oldstatus.get('running',[]):
 assert not Path('/proc') .joinpath(str(run['pid'])).exists(),run
if not oldstatus.get('cancelled'):
 oldstatus.update(cancelled=True,cancelled_at=time.time(),reason='User requested dense reward and original maze budgets',
                  cancelled_running=oldstatus['running'],cancelled_pending=oldstatus['pending'],running=[],pending=[])
 (oldroot/'status.json').write_text(json.dumps(oldstatus,indent=2)+'\n')
first,second=('v1','v3') if shard==0 else ('v2','v4')
order=[(first,'dipo'),(second,'dipo'),(first,'optiq'),(first,'mfpo'),
       (first,'sac'),(second,'optiq'),(second,'mfpo'),(second,'sac')]
budgets=dict(v1=3000000,v2=3000000,v3=4000000,v4=5000000)
manifest=dict(campaign=name,source=str(source),source_commit=sha,shard=shard,seed=0,
 upstream_commit='7edd06c4799abbab0f8fa534c21deb56253b018e',
 reward='negative Euclidean distance from next xy to nearest goal; no sparse bonus',
 noveld_coefficient=.01,num_envs=64,batch_size=4096,updates_per_vector_step=2,
 warmup_transitions=8192,replay_capacity=1000000,eval_interval=250000,checkpoint='final only',
 dipo_support=[-6000,5],dipo_num_atoms=51,budgets=budgets,
 jobs=[dict(id=f'{task}-{method}-s0',task=task,method=method,steps=budgets[task]) for task,method in order])
root.mkdir(exist_ok=False)
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(root/'environment-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
conf=Path('/home/heechan/OptiQ-ops/supervisor/jobs')/(name+'.conf')
assert not conf.exists()
conf.write_text(f'''[program:{name}]
command=/home/heechan/.venv-ddiffpg-native/bin/python -m antmaze_experiments.controller --root {root}
directory={source}
environment=PYTHONDONTWRITEBYTECODE="1"
autostart=false
autorestart=false
startsecs=2
stopasgroup=true
killasgroup=true
stopwaitsecs=30
stdout_logfile={root}/controller.log
stderr_logfile={root}/controller.err
stdout_logfile_maxbytes=0
stderr_logfile_maxbytes=0
''')
for cmd in (['reread'],['update',name],['start',name]):
 subprocess.run(ctl+cmd,check=True)
(root/'registration.json').write_text(json.dumps(dict(time=time.time(),source_commit=sha,supervisor=name,config=str(conf),manifest=manifest),indent=2)+'\n')
print(json.dumps(dict(root=str(root),source=sha,jobs=len(order))))
