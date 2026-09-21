"""Cancel only pending DACER seed 4; preserve live training during queue handoff."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import controller as c

task=sys.argv[1]
assert task in ('ant','humanoid')
sha=c.old.source()
manifest_path=c.ROOT/f'manifest-{task}.json'
manifest=json.loads(manifest_path.read_text())
assert not manifest.get('allow_live_adoption'), 'Already migrated'
name=f'{c.CAMPAIGN}-{task}'
controller_pid=int(subprocess.check_output(c.CTL+['pid',name],text=True).strip())
assert controller_pid>0
controller_cmd=Path(f'/proc/{controller_pid}/cmdline').read_bytes()
assert b'controller.py\0run\0'+task.encode() in controller_cmd
seed4=c.ROOT/'jobs'/f'{task}-trg-dacer-T0.25-b1-s4.json'
job=json.loads(seed4.read_text()); assert job['status']=='queued'
running=[]
for n in manifest['jobs']:
    j=json.loads((c.ROOT/'jobs'/(n+'.json')).read_text())
    if j['status']=='running':
        c.AdoptedProcess(j); running.append(j)
assert len(running)==4 and {j['seed'] for j in running}==set(range(4))
assert all(j['stage']=='dacer' and j['temperature']==.25 for j in running)
# SIGSTOP/SIGKILL target only the controller PID, never its process group.
# Training children retain their original PIDs, source and GPU locks.
os.kill(controller_pid,signal.SIGSTOP)
backup=c.ROOT/f'manifest-{task}.before-four-seeds.json'
backup.write_text(manifest_path.read_text())
job.update(status='canceled',reason='User explicitly limits DACER to seeds 0..3',canceled=time.time())
c.save(seed4,job)
manifest['jobs'].remove(job['name'])
manifest.update(controller_commit=sha,controller_source=str(c.REPO),allow_live_adoption=True,
    canceled_jobs=[job['name']],dacer_seeds=list(range(4)),beta_seeds=list(range(5)),
    order='DACER seeds 0..3 then beta .5/.9 seeds 0..4; immediate per-GPU backfill',
    scheduler_handoff=dict(previous_pid=controller_pid,time=time.time(),preserved_training_pids=[j['pid'] for j in running]))
c.save(manifest_path,manifest)
os.kill(controller_pid,signal.SIGKILL)
confdir=Path('/home/heechan/OptiQ-ops/supervisor/jobs')
oldconf=confdir/(name+'.conf')
oldtext=oldconf.read_text()
oldconf.write_text(oldtext.replace('autostart=true','autostart=false'))
newname=name+'-scheduler2'
newconf=confdir/(newname+'.conf'); assert not newconf.exists()
text=oldtext.replace(f'[program:{name}]',f'[program:{newname}]')
text=text.replace(manifest['source'],str(c.REPO))
newconf.write_text(text)
for _ in range(20):
    if not Path(f'/proc/{controller_pid}').exists(): break
    time.sleep(.2)
assert not Path(f'/proc/{controller_pid}').exists()
subprocess.run(c.CTL+['reread'],check=True)
subprocess.run(c.CTL+['update',name],check=True)
subprocess.run(c.CTL+['update',newname],check=True)
for j in running: c.AdoptedProcess(j)
print(json.dumps(dict(service=newname,controller_commit=sha,canceled=job['name'],preserved_pids=[j['pid'] for j in running])),flush=True)
