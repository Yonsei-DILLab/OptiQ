"""Read-only status/provenance collector for the positive DACER target sweep."""
import concurrent.futures
import json
from pathlib import Path
import subprocess
import time

CAMPAIGN='antmaze-optiq-v3-teacherfloor-250k-s0-20260925'
SOURCE='4607dd5c14e12fb87b395eb2d7192c742f77be22'
OUTPUT=Path(__file__).resolve().parent
HOSTS=('vast-heechan-180',)
REMOTE=r'''
import json, pathlib, subprocess, time
root=pathlib.Path('/home/heechan/optiq-experiments/antmaze-optiq-v3-teacherfloor-250k-s0-20260925')
def read(path):
    if not path.exists():return None
    try:return json.loads(path.read_text())
    except json.JSONDecodeError:return {'read_in_progress':True}
data={'time':time.time(),'files':{},'jobs':{}}
for name in ('manifest','status','failure','result','registration','wandb-sync-status','controller-provenance','priority-dispatch-audit','source-sharing'):
    data['files'][name]=read(root/(name+'.json'))
for entry in data['files']['manifest']['jobs']:
    identifier=entry['id'];job={'entry':entry,'status':read(root/'jobs'/(identifier+'.json'))}
    for phase in ('preflight','runs'):
        folder=root/phase/identifier
        job[phase]={name:read(folder/(name+'.json')) for name in ('progress','result','failure','config','optiq-profile-verification','dacer-target-verification','checkpoint-verification','wandb','dacer_regulator')}
        history=folder/'dacer_regulator_history.jsonl'
        if history.exists():job[phase]['regulator_history']=[json.loads(line) for line in history.read_text().splitlines() if line.strip()]
    data['jobs'][identifier]=job
ps=subprocess.check_output(['ps','-eo','pid,ppid,etimes,pcpu,args'],text=True)
data['processes']=[line for line in ps.splitlines() if root.name in line]
data['gpus']=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader'],text=True)
data['gpu_processes']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
print(json.dumps(data))
'''

def collect(host):
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',host,
                      '/home/heechan/.venv-ddiffpg-native/bin/python','-'],
                     input=REMOTE,text=True,capture_output=True,timeout=60,check=True)
    data=json.loads(r.stdout)
    assert data['files']['manifest']['source_commit']==SOURCE
    (OUTPUT/(host+'.json')).write_text(json.dumps(data,indent=2)+'\n')
    s=data['files']['status'] or {}
    compact=[]
    for key,value in data['jobs'].items():
        status=value['status'] or {};phase=status.get('phase')
        info=value.get(phase,{}) if phase else {}
        p=info.get('progress') or {}
        compact.append(dict(id=key,status=status.get('status','pending'),phase=phase,
                            step=p.get('step'),updates=p.get('updates'),
                            dacer_updates=p.get('dacer_updates'),failure=info.get('failure'),
                            wandb=(value['runs'].get('wandb') or {}).get('url')))
    return dict(host=host,time=data['time'],running=len(s.get('running',[])),
                pending=len(s.get('pending',[])),completed=len(s.get('completed',[])),
                failed=s.get('failed',[]),jobs=compact)

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(2) as pool: summaries=list(pool.map(collect,HOSTS))
    (OUTPUT/'latest-status.json').write_text(json.dumps(summaries,indent=2)+'\n')
    print(json.dumps(summaries,indent=2))
