"""Continue an explicitly moved run from its original local W&B journal.

CPU logging only. Never imports a learner, changes a checkpoint, or restarts
training. Existing cloud rows are skipped by their original environment step.
"""
import argparse
import json
import math
import os
from pathlib import Path
import time
import traceback
import wandb
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore

def atomic(path,data):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)

def scan(path):
    rows=[]; summary={}; ended=False; partial=None
    ds=DataStore(); ds.open_for_scan(str(path))
    try:
        while True:
            try:
                blob=ds.scan_data()
                if blob is None:break
                record=wandb_internal_pb2.Record();record.ParseFromString(blob)
            except Exception as error:
                partial=f'{type(error).__name__}: {error}'
                break
            if record.HasField('history'):
                row={}
                for item in record.history.item:
                    key=item.key or '/'.join(item.nested_key)
                    value=json.loads(item.value_json)
                    if isinstance(value,(float,int)) and math.isfinite(value):row[key]=value
                if row:
                    assert '_step' in row
                    rows.append(row)
            elif record.HasField('summary'):
                for item in record.summary.update:
                    key=item.key or '/'.join(item.nested_key)
                    value=json.loads(item.value_json)
                    if key and not key.startswith('_'):summary[key]=value
            elif record.HasField('exit'):
                ended=True
    finally:
        ds.close()
    return rows,summary,ended,partial

parser=argparse.ArgumentParser()
parser.add_argument('--run',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--inspect-only',action='store_true')
args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
meta=json.loads((args.run/'wandb.json').read_text())
paths=list((args.run/'wandb').glob('run-*/run-*.wandb'))
assert len(paths)==1
journal=paths[0]
assert journal.name==f"run-{meta['id']}.wandb"
rows,summary,ended,partial=scan(journal)
if args.inspect_only:
    print(json.dumps(dict(id=meta['id'],rows=len(rows),first=rows[0]['_step'],last=rows[-1]['_step'],
        ended=ended,partial=partial,summary_keys=list(summary))))
    raise SystemExit(0)

api=wandb.Api()
existing=api.run(f"OptiQ/antmaze/{meta['id']}")
before=existing.lastHistoryStep
assert any(x['_step']==before for x in rows)
original_config=existing.config
session=wandb.init(entity='OptiQ',project='antmaze',id=meta['id'],resume='must',
    dir=str(args.output),mode='online',
    settings=wandb.Settings(console='off',x_disable_stats=True,save_code=False))
last=before; sent=0
try:
    while True:
        rows,summary,ended,partial=scan(journal)
        for row in rows:
            step=int(row['_step'])
            if step<=last:continue
            payload={k:v for k,v in row.items() if k!='_step'}
            session.log(payload,step=step,commit=True)
            last=step;sent+=1
        if summary:session.summary.update(summary)
        complete=(args.run/'result.json').exists() and ended and partial is None
        atomic(args.output/'status.json',dict(time=time.time(),id=meta['id'],
            destination='OptiQ/antmaze',journal=str(journal),before_step=before,
            last_submitted_step=last,rows_submitted=sent,source_exit_record=ended,
            source_partial_tail=partial,completed=complete,training_modified=False))
        if complete:break
        if (args.run/'failure.json').exists():
            raise RuntimeError('Training reported failure; preserve data and stop this logging sidecar')
        time.sleep(15)
    result=json.loads((args.run/'result.json').read_text())
    assert last==result['steps']
    session.summary.update(dict(completed=True,steps=result['steps'],updates=result['updates']))
    # Preserve the original journal, including post-move system events, as a run file.
    session.save(str(journal),base_path=str(journal.parent),policy='now')
    session.finish()
    checked=wandb.Api().run(f"OptiQ/antmaze/{meta['id']}")
    assert checked.lastHistoryStep==last and checked.config==original_config
    actual=list(checked.scan_history(min_step=before+1,page_size=1000))
    expected={int(x['_step']):x for x in rows if int(x['_step'])>before}
    assert len(actual)==len(expected),(len(actual),len(expected))
    for row in actual:
        exp=expected[int(row['_step'])]
        for key,value in exp.items():
            if key.startswith('_'):continue
            assert key in row and math.isclose(row[key],value,rel_tol=1e-7,abs_tol=1e-9),(row['_step'],key)
    atomic(args.output/'verification.json',dict(id=meta['id'],history_before=before,
        history_final=last,all_missing_numeric_history_verified=True,rows=len(actual),
        config_unchanged=True,original_journal_archived=True,training_modified=False))
except BaseException as error:
    atomic(args.output/'failure.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
    raise
