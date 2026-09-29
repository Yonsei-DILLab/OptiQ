"""Bounded local-file status read; no W&B API or training mutations."""
import csv
import json
from pathlib import Path
import time
root=Path('/home/heechan/optiq-experiments/trg-temp-beta-20260921')
result={'checked_unix':time.time(),'files':{},'jobs':[]}
for pattern in ('status-*.json','failure-*.json','selection-*.json','result-*.json'):
 for p in root.glob(pattern):result['files'][p.name]=json.loads(p.read_text())
for p in sorted((root/'jobs').glob('*.json')):
 job=json.loads(p.read_text())
 row={k:job[k] for k in ('name','task','stage','temperature','beta','seed','status','commit','gpu','timesteps','updates','wandb_url','metrics','error') if k in job}
 if job['status']=='running':
  paths=list((root/'outputs').glob(job['name']+'_*/logs/progress.csv'))
  for progress in paths:
   with progress.open() as f:
    header=f.readline().strip().split(',')
    f.seek(max(f.tell(),progress.stat().st_size-300000))
    f.readline();lines=f.readlines()
   last={}
   for line in lines:
    values=dict(zip(header,next(csv.reader([line]))))
    for k,v in values.items():
     if k in ('time/total_timesteps','train/n_updates','train/actor_loss','train/critic_loss','eval/stochastic_z/mean_reward') and v:last[k]=float(v)
   row['latest']=last;row['log_mtime']=progress.stat().st_mtime
 result['jobs'].append(row)
print(json.dumps(result))
