"""Read-only remote audit of saved direct-policy rollouts, without bulk transfer."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import subprocess,json,hashlib
ROOT=Path(__file__).resolve().parent
REMOTE=r'''
from pathlib import Path
from collections import Counter
import json,sys,hashlib,time,numpy as np
root=Path('/home/heechan/optiq-experiments/antmaze-optiq-geodesic-no-step-B0-T1-s0-20260924')
manifest=json.loads((root/'manifest.json').read_text());sys.path.insert(0,manifest['source'])
from antmaze_experiments.progress_reward import distance,maze_geometry,bonus_enabled,progress_scale,step_cost

def crossing(path,x):
 old,new=path[:-1],path[1:];ix=np.flatnonzero((old[:,0]>x)&(new[:,0]<=x))
 if not len(ix):return None
 i=ix[0];t=(x-old[i,0])/(new[i,0]-old[i,0]);return float(old[i,1]+t*(new[i,1]-old[i,1]))
def family(task,path):
 if task in ('v1','v4'):
  y=crossing(path,-4);return 'upper' if y is not None and y>2 else 'lower' if y is not None and y< -2 else 'uncommitted'
 gate=4 if task=='v2' else 8;left=(path[:,0]<-gate).any();right=(path[:,0]>gate).any()
 return 'both' if left and right else 'left' if left else 'right' if right else 'uncommitted'
result=dict(time=time.time(),source_commit=manifest['source_commit'],runs=[])
for run in sorted((root/'runs').glob('*')):
 if not (run/'config.json').exists():continue
 cfg=json.loads((run/'config.json').read_text());task=cfg['task'];profile=cfg['reward_profile'];history=[]
 assert cfg['source_commit']==manifest['source_commit']
 for f in sorted(run.glob('evaluations/*/policy-natural/summary.json')):
  summary=json.loads(f.read_text());raw=f.parent/'rollouts.npz';data=np.load(raw)
  paths=[xy[:int(n)+1] for xy,n in zip(data['xy'],data['lengths'])]
  assert len(paths)==summary['episodes'] and not summary['fixed']
  assert np.unique(data['initial_full_state'],axis=0).shape[0]>1
  assert all(np.isfinite(p).all() for p in paths)
  fam=[family(task,p) for p in paths];success=int(sum(data['goals']>0))
  assert np.isclose(success/len(paths),summary['success_rate'])
  errors=[]
  for p,n,g,ret in zip(paths,data['lengths'],data['goals'],data['returns']):
   bonus=(20 if task=='v2' and g==1 else 10) if g and bonus_enabled(profile) else 0
   expected=progress_scale(profile)*(distance(p[0],task,profile)-distance(p[-1],task,profile))-step_cost(profile)*n+bonus
   errors.append(abs(float(ret-expected)))
  assert max(errors)<.01,(f,max(errors))
  row=dict(step=summary['step'],episodes=len(paths),successes=success,
   goal_counts={str(int(k)):int(v) for k,v in Counter(data['goals']).items()},
   corridors=dict(Counter(fam)),successful_corridors=dict(Counter(c for c,g in zip(fam,data['goals']) if g)),
   max_reward_sum_error=max(errors),raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),raw_path=str(raw))
  history.append(row)
 progress=json.loads((run/'progress.json').read_text()) if (run/'progress.json').exists() else {}
 result['runs'].append(dict(id=run.name,task=task,profile=profile,progress=progress,budget=cfg['steps'],history=history,latest=history[-1] if history else None))
print(json.dumps(result))
'''

def collect(host):
 p=subprocess.run(['ssh','-o','ConnectTimeout=15',host,'/home/heechan/.venv-ddiffpg-native/bin/python','-'],input=REMOTE,text=True,capture_output=True,check=True,timeout=90)
 d=json.loads(p.stdout);d['host']=host;return d
if __name__=='__main__':
 stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');destination=ROOT/'metric_reports'/stamp;destination.mkdir(parents=True)
 with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(collect,['vast-heechan-180','vast-heechan-199']))
 for row in rows:(destination/(row['host']+'.json')).write_text(json.dumps(row,indent=2)+'\n')
 data=dict(collected_utc=stamp,remote_raw_audited=True,scope='Saved random-start direct-policy rollouts; no new evaluation/training',collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),hosts=rows)
 (destination/'analysis.json').write_text(json.dumps(data,indent=2)+'\n');(ROOT/'latest-metrics-path.txt').write_text(str(destination)+'\n')
 for row in rows:
  print(row['host'])
  for run in row['runs']:print(json.dumps({k:run[k] for k in ['id','latest']},ensure_ascii=False))
 print(destination)
