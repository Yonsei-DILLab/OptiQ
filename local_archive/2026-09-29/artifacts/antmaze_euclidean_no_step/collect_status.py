"""Read-only snapshots of the two Euclidean progress-only campaign shards."""
import concurrent.futures,json,subprocess,time
from pathlib import Path
ROOT='/home/heechan/optiq-experiments/antmaze-optiq-euclidean-no-step-B0-T1-s0-20260925-r2'
OUT=Path(__file__).resolve().parent
CODE='''import json,os,time
from pathlib import Path
root=Path(%r)
def read(p):
 if not p.exists():return None
 try:return json.loads(p.read_text())
 except json.JSONDecodeError:return {'read_in_progress':True}
result={'time':time.time(),'root':str(root)}
for name in ['manifest','registration','status','failure','result','wandb-sync-status']:
 result[name]=read(root/(name+'.json'))
result['jobs']={}
for job in (result.get('manifest') or {}).get('jobs',[]):
 key=job['id'];j={'job':read(root/'jobs'/(key+'.json'))}
 for phase in ['preflight','runs']:
  directory=root/phase/key
  j[phase]={n:read(directory/(n+'.json')) for n in ['config','progress','result','failure','checkpoint-verification','wandb','optiq-profile-verification','dacer-disabled-verification']}
 result['jobs'][key]=j
status=result.get('status') or {}
result['pid_alive']={str(x['pid']):Path('/proc/'+str(x['pid'])).exists() for x in status.get('running',[])}
print(json.dumps(result))
''' % ROOT

def collect(host):
 p=subprocess.run(['ssh',host,'python3','-'],input=CODE,text=True,capture_output=True,check=True)
 data=json.loads(p.stdout)
 (OUT/(host+'.json')).write_text(json.dumps(data,indent=2)+'\n')
 status=data['status'] or {};rows=[]
 for identifier,j in data['jobs'].items():
  run=j['runs'];pre=j['preflight']
  progress=run['progress'] or pre['progress'] or {}
  rows.append(dict(id=identifier,phase=(j['job'] or {}).get('phase'),status=(j['job'] or {}).get('status'),steps=progress.get('steps',progress.get('step')),preflight_complete=bool(pre['result']),main_started=bool(run['config']),wandb=run['wandb']))
 return dict(host=host,pending=len(status.get('pending',[])),running=len(status.get('running',[])),completed=len(status.get('completed',[])),failed=status.get('failed',[]),pid_alive=data['pid_alive'],jobs=rows)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 rows=list(pool.map(collect,['vast-heechan-180','vast-heechan-199']))
(OUT/'latest-status.json').write_text(json.dumps(dict(collected=time.time(),hosts=rows),indent=2)+'\n')
print(json.dumps(rows,indent=2))
