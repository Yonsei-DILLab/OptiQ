"""Read-only status/config/W&B audit for either campaign host."""
import csv
import json
import os
from pathlib import Path
import re
import requests
from dotenv import load_dotenv

root=Path('/home/heechan/optiq-experiments/trg-temp-beta-20260921')
result={p.stem:json.loads(p.read_text()) for pattern in ('manifest-*.json','status-*.json','selection-*.json','result-*.json','failure-*.json') for p in root.glob(pattern)}
runs=[]
for jobpath in sorted((root/'jobs').glob('*.json')):
    job=json.loads(jobpath.read_text())
    paths=list((root/'outputs').glob(job['name']+'_*/config.json'))
    job['config_files']=len(paths)
    if len(paths)==1:
        path=paths[0];config=json.loads(path.read_text());job['config']=config
        last={};progress=path.parent/'logs/progress.csv'
        keep={'time/total_timesteps','train/n_updates','train/actor_loss','train/critic_loss','train/actor_std_mean',
              'eval/zero_z/mean_reward','eval/stochastic_z/mean_reward'}
        if progress.exists():
            with progress.open() as f:
                for row in csv.DictReader(f):
                    for k,v in row.items():
                        if k in keep and v:last[k]=float(v)
        job['latest_metrics']=last
    log=root/'logs'/(job['name']+'.log')
    if log.exists():
        with log.open() as f:
            for line in f:
                match=re.match(r'W&B: (https://wandb.ai/OptiQ/gmm-trg/runs/\w+)',line)
                if match:
                    job['wandb_url']=match.group(1);break
    runs.append(job)
result['jobs']=runs
load_dotenv('/home/heechan/.env',override=False)
online=[r for r in runs if 'wandb_url' in r]
if online:
    fields=' '.join(f'r{i}:run(name:{json.dumps(j["wandb_url"].rsplit("/",1)[-1])}){{name state summaryMetrics}}' for i,j in enumerate(online))
    response=requests.post('https://api.wandb.ai/graphql',auth=('api',os.environ['WANDB_API_KEY']),
        json={'query':'{project(name:"gmm-trg",entityName:"OptiQ"){'+fields+'}}'},timeout=40)
    response.raise_for_status();data=response.json()
    assert not data.get('errors'),data.get('errors')
    for i,j in enumerate(online):
        value=data['data']['project'][f'r{i}'];summary=json.loads(value['summaryMetrics'])
        j['cloud']={'state':value['state'],'summary':{k:v for k,v in summary.items() if k in
            ('env_steps','train/n_updates','eval/zero_z/mean_reward','eval/stochastic_z/mean_reward','completed','timesteps')}}
print(json.dumps(result))
