import json,subprocess,time
from pathlib import Path
root=Path('/home/heechan/optiq-experiments/antmaze-optiq-progress-2x2-T1-s0-20260924')
side=root/'cancelled-for-progress100.json'
assert not side.exists(), side
before={n:json.loads((root/(n+'.json')).read_text()) for n in ['manifest','status']}
ctl=['/usr/local/bin/supervisorctl','-c','/home/heechan/OptiQ-ops/supervisor/supervisord.conf']
results=[]
for suffix in ['','-wandb-sync']:
 p=subprocess.run(ctl+['stop',root.name+suffix],text=True,capture_output=True)
 results.append({'service':root.name+suffix,'code':p.returncode,'out':p.stdout,'err':p.stderr})
assert results[0]['code']==0,results
p=subprocess.run(['ps','-eo','pid,ppid,args'],text=True,capture_output=True,check=True)
live=[x for x in p.stdout.splitlines() if str(root) in x]
record=dict(time=time.time(),reason='User replaced reward with 100*(d_current-d_next)-1+B; preserve all artifacts; never resume this campaign',snapshot=before,stops=results,remaining_processes=live)
side.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(stops=results,remaining=live,pending_cancelled=len(before['status'].get('pending',[])),running_stopped=len(before['status'].get('running',[])))))
assert not live,live
