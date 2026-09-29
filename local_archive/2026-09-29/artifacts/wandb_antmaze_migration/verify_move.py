import base64
import concurrent.futures
import json
from pathlib import Path
from datetime import datetime, timezone
import wandb
from wandb.sdk.internal.internal_api import Api

root=Path('/home/heechan/OptiQ-ops/wandb-antmaze-migration-20260923')
inventory=json.loads((root/'inventory-before.json').read_text())['runs']
expected={x['id'] for x in inventory if x['antmaze']}
others={x['id'] for x in inventory if not x['antmaze']}
public=wandb.Api()
source=list(public.runs('OptiQ/gmm-trg',per_page=200))
dest=list(public.runs('OptiQ/antmaze',per_page=200))
assert {x.id for x in source}==others
assert {x.id for x in dest}==expected
before={**json.loads((root/'historical-before.json').read_text()),**json.loads((root/'live-before.json').read_text())}
original_live=set(json.loads((root/'live-before.json').read_text()))

def verify(run):
    b=before[run.id]
    assert run.name==b['name'] and run.group==b['group'] and run.config==b['config']
    # W&B storage_id encodes its project namespace; moving changes that component.
    old=base64.b64decode(b['storage_id']).decode()
    new=base64.b64decode(run.storage_id).decode()
    assert new==old.replace(':gmm-trg:OptiQ',':antmaze:OptiQ')
    last=run.lastHistoryStep
    summary=dict(run.summary)
    if run.id not in original_live:
        assert last==b['last_history_step'],(run.id,'history length changed')
        assert summary==b['summary'],(run.id,'summary changed')
    else:
        assert last>=b['last_history_step'],(run.id,'history regressed')
    tail=list(run.scan_history(min_step=max(0,last-1),page_size=10)) if last>=0 else []
    assert tail or last<0,(run.id,'history inaccessible')
    return dict(id=run.id,name=run.name,state=run.state,url=run.url,
        history_before=b['last_history_step'],history_after=last,
        tail_rows=len(tail),tail=tail,live_at_inventory=run.id in original_live,
        id_name_group_config_preserved=True,history_accessible=True,
        summary_preserved=(summary==b['summary']))

with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    checked=list(pool.map(verify,dest))
api=Api()
tasks={}
for kind in ['historical','live']:
    move=json.loads((root/f'{kind}-move.json').read_text())
    tasks[kind]=api.execute('query($id:ID!){task(id:$id){id state progress result}}',variables={'id':move['response']['moveRuns']['task']['id']})
report=dict(at=datetime.now(timezone.utc).isoformat(),source_remaining=len(source),destination_count=len(dest),
    expected_ids_preserved=True,other_runs_preserved=True,tasks=tasks,checked=checked,
    note='Native run move preserves history; project-encoded storage ID changes namespace. Original internal wandb-history artifacts were not deleted.')
(root/'verification-after.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
print(json.dumps(dict(count=len(dest),other_runs=len(source),verified=len(checked),tasks=tasks,
    active=[{k:v for k,v in x.items() if k in ['id','state','history_before','history_after']} for x in checked if x['live_at_inventory']])))
