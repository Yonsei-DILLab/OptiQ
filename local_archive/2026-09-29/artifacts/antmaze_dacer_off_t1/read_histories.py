"""Read exact local W&B records from preserved campaigns, without API writes."""
import hashlib
import json
import math
from pathlib import Path
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2

BASE=Path('/home/heechan/optiq-experiments')
NAMES=['antmaze-dense-dacer-entropy-T1-s0-20260924',
       'antmaze-optiq-dense-off-T1-s0-20260924',
       'antmaze-dense-off-16-current-s0-20260924',
       'antmaze-dense-anneal-baselines-s0-20260924']
result={}
for name in NAMES:
    for run in sorted((BASE/name/'runs').glob('*')):
        if not (run/'config.json').exists():continue
        config=json.loads((run/'config.json').read_text())
        if name=='antmaze-dense-anneal-baselines-s0-20260924' and config['method']=='optiq':continue
        if name=='antmaze-dense-off-16-current-s0-20260924' and config['method']=='optiq':continue
        files=sorted((run/'wandb').glob('*run-*/run-*.wandb'))
        if not files:continue
        rows=[];warnings=[];hashes={}
        for path in files:
            hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            ds=DataStore();ds.open_for_scan(str(path))
            while True:
                try:blob=ds.scan_data()
                except Exception as error:
                    warnings.append(dict(path=str(path),type=type(error).__name__,message=str(error)));break
                if blob is None:break
                record=wandb_internal_pb2.Record();record.ParseFromString(blob)
                if not record.HasField('history'):continue
                row={}
                for item in record.history.item:
                    key=item.key or '/'.join(item.nested_key)
                    try:value=json.loads(item.value_json)
                    except (TypeError,ValueError):continue
                    if isinstance(value,(int,float)) and math.isfinite(value):row[key]=value
                if row:rows.append(row)
        evaluations=[json.loads(p.read_text()) for p in sorted((run/'evaluations').glob('*/policy-natural/summary.json'))]
        result[name+'/'+run.name]=dict(config=config,history=rows,source_sha256=hashes,
            read_warnings=warnings,evaluations=evaluations,
            regulator=json.loads((run/'dacer_regulator.json').read_text()) if (run/'dacer_regulator.json').exists() else None)
print(json.dumps(result,allow_nan=False))
