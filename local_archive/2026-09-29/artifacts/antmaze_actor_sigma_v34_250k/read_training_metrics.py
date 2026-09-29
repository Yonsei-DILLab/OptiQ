"""Read existing local W&B event files; no API, evaluation, or training calls."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

OUT = Path(__file__).resolve().parent / 'diagnostics'
ROOTS = {
    'vast-heechan-180': ['antmaze-optiq-v3-teacherfloor-250k-s0-20260925-r2'],
    'vast-heechan-199': ['antmaze-optiq-v4-teacherfloor-250k-s0-20260925',
                       'antmaze-optiq-sigmacap-v34-250k-s0-20260925'],
}
CODE = r'''
from pathlib import Path
import hashlib,json,tempfile
from wandb.sdk.internal.datastore import DataStore
from wandb.proto.wandb_internal_pb2 import Record
records=[]
for campaign in CAMPAIGNS:
    root=Path('/home/heechan/optiq-experiments')/campaign
    for folder in sorted((root/'runs').glob('*')):
        paths=sorted({p.resolve() for p in folder.glob('wandb/*/run-*.wandb')})
        if not paths:continue
        assert len(paths)==1,(str(folder),len(paths))
        path=paths[0];content=path.read_bytes();rows=[];tail_error=None
        with tempfile.NamedTemporaryFile(suffix='.wandb') as snapshot:
            snapshot.write(content);snapshot.flush()
            reader=DataStore();reader.open_for_scan(snapshot.name)
            while True:
                try:data=reader.scan_data()
                except Exception as exc:
                    tail_error=type(exc).__name__+': '+str(exc);break
                if data is None:break
                rec=Record();rec.ParseFromString(data)
                if not rec.HasField('history'):continue
                row={}
                for item in rec.history.item:
                    key=item.key or '/'.join(item.nested_key)
                    if key in ('step','_step') or key.startswith(('train/','exploration/')):
                        row[key]=json.loads(item.value_json)
                if any(k.startswith('train/') for k in row):rows.append(row)
        records.append(dict(campaign=campaign,job=folder.name,path=str(path),
            bytes=len(content),sha256=hashlib.sha256(content).hexdigest(),
            snapshot=True,tail_error=tail_error,rows=rows))
print(json.dumps(records))
'''


def collect(host):
    code = 'CAMPAIGNS=' + repr(ROOTS[host]) + '\n' + CODE
    p = subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host,
         '/home/heechan/.venv-ddiffpg-native/bin/python', '-'],
        input=code, text=True, capture_output=True, timeout=60, check=True)
    return host, json.loads(p.stdout.splitlines()[-1])


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        records = dict(pool.map(collect, ROOTS))
    result = dict(time_utc=datetime.now(timezone.utc).isoformat(), records=records,
                  read_only=True, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'training-metrics.json').write_text(json.dumps(result, indent=2)+'\n')
    keys = ['step','train/actor_std_mean','train/source_ess_absolute',
            'train/max_source_weight','train/gmm_component_ess_fraction',
            'train/critic_loss','train/current_q_values','train/next_q_values']
    for host, runs in records.items():
        for run in runs:
            last = run['rows'][-1] if run['rows'] else {}
            print(json.dumps(dict(host=host,job=run['job'],rows=len(run['rows']),
                                  tail_error=run['tail_error'],last={k:last[k] for k in keys if k in last})))
