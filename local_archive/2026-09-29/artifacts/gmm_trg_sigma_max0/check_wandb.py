"""Read back uploaded summaries without exposing authentication material."""
import json
import os
from pathlib import Path
import requests
from dotenv import load_dotenv

load_dotenv('/home/heechan/.env',override=False)
root=Path('/home/heechan/optiq-experiments/trg-sigma-max0-20260921')
jobs=[json.loads(p.read_text()) for p in root.glob('*-s0.json')]
jobs=[j for j in jobs if j.get('wandb_url')]
fields=' '.join(f'r{i}:run(name:{json.dumps(j["wandb_url"].rsplit("/",1)[-1])}){{name state summaryMetrics}}' for i,j in enumerate(jobs))
r=requests.post('https://api.wandb.ai/graphql',auth=('api',os.environ['WANDB_API_KEY']),
    json={'query':'{project(name:"abla",entityName:"OptiQ"){'+fields+'}}'},timeout=40)
r.raise_for_status()
data=r.json()
assert not data.get('errors'), data.get('errors')
rows=[]
for i,job in enumerate(jobs):
    value=data['data']['project'][f'r{i}']
    summary=json.loads(value['summaryMetrics'])
    rows.append(dict(task=job['task'],sigma=job['sigma'],state=value['state'],
        wandb_url=job['wandb_url'],completed=summary.get('completed'),
        timesteps=summary.get('timesteps'),updates=summary.get('updates'),
        mean_sigma=summary.get('train/actor_std_mean'),
        max_log_sigma=summary.get('train/actor_log_std_max'),
        fraction_at_cap=summary.get('train/actor_std_at_max_fraction'),
        zero_z=summary.get('final_eval_return_zero_z'),
        stochastic_z=summary.get('final_eval_return_stochastic_z')))
print(json.dumps(rows))
