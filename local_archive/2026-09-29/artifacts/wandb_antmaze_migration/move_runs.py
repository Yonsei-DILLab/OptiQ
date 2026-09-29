"""Move exactly the inventoried AntMaze runs using W&B's native move mutation.

The mutation and input schema were read from authenticated server introspection.
No run is copied/recreated, no history is deleted, and both projects are private.
"""
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
import wandb
from wandb.sdk.internal.internal_api import Api

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--kind', choices=['historical', 'live'], required=True)
args = parser.parse_args()
root = args.root
inventory = json.loads((root/'inventory-before.json').read_text())['runs']
selected = [x for x in inventory if x['antmaze'] and ((x['state']=='running') == (args.kind=='live'))]
assert len(selected) == (113 if args.kind=='historical' else 4)
ids = sorted(x['id'] for x in selected)
assert len(ids) == len(set(ids))
assert all(x['group'].startswith('antmaze') for x in selected)
public = wandb.Api()
api = Api()
projects = api.execute('{source:project(name:"gmm-trg",entityName:"OptiQ"){name access} destination:project(name:"antmaze",entityName:"OptiQ"){name access}}')
assert projects['source']['access'] == projects['destination']['access'] == 'PRIVATE'
filters = {'name':{'$in':ids}}
found = list(public.runs('OptiQ/gmm-trg', filters=filters, per_page=200))
assert sorted(x.id for x in found) == ids, 'Refuse a partial or repeated migration'
before = {x.id:dict(name=x.name,group=x.group,state=x.state,storage_id=x.storage_id,
                   config=x.config,summary=dict(x.summary),last_history_step=x.lastHistoryStep) for x in found}
(root/f'{args.kind}-before.json').write_text(json.dumps(before,indent=2,default=str)+'\n')
request = dict(sourceEntityName='OptiQ',sourceProjectName='gmm-trg',
               destinationEntityName='OptiQ',destinationProjectName='antmaze',
               filters=json.dumps(filters))
response = api.execute('mutation MoveAntmaze($input:MoveRunsInput!){moveRuns(input:$input){task{id name state progress result}}}', variables={'input':request})
record = dict(at=datetime.now(timezone.utc).isoformat(),count=len(ids),ids=ids,
              request=request,response=response,method='native moveRuns; same run IDs')
(root/f'{args.kind}-move.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(count=len(ids),response=response)))
