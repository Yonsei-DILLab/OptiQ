import csv
import json
from pathlib import Path
import sys
import numpy as np

root=Path('/home/heechan/optiq-experiments/trg-sigma-abla-20260921')
jobs=[]
for state in sorted(root.glob('*-s0.json')):
    job=json.loads(state.read_text())
    outputs=list((root/'outputs').glob(state.stem+'_*/config.json'))
    if len(outputs)==1:
        out=outputs[0].parent
        job['config']=json.loads(outputs[0].read_text())
        job['evaluations']={}
        for mode in ('zero_z','stochastic_z'):
            paths=list(out.rglob(f'evaluations_{mode}.npz'))
            if paths:
                with np.load(paths[0]) as data:
                    job['evaluations'][mode]={k:data[k].tolist() for k in
                        ('timesteps','results','ep_lengths','env_seeds','policy_seeds')}
        runs=list((out/'wandb').glob('run-*'))
        if runs:job['wandb_url']='https://wandb.ai/OptiQ/abla/runs/'+runs[0].name.rsplit('-',1)[1]
        path=out/'logs/progress.csv'
        job['last_metrics']={}
        job['sigma_history']=[]
        if path.exists():
            for row in csv.DictReader(path.open()):
                if row.get('train/actor_std_mean') and row.get('time/total_timesteps'):
                    job['sigma_history'].append({k:float(row[k]) for k in
                        ('time/total_timesteps','train/actor_std_mean','train/actor_std_at_max_fraction') if row.get(k)})
                for key,value in row.items():
                    if value:
                        try:job['last_metrics'][key]=float(value)
                        except (ValueError,TypeError):pass
        job['final_checkpoints']=[p.name for p in out.rglob('*_10000.msgpack')]
    jobs.append(job)
print(json.dumps(dict(manifest=json.loads((root/'manifest.json').read_text()),jobs=jobs)))
