"""Read existing W&B binary histories only; no training or run mutation."""
import json
import math
from pathlib import Path
import sys
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2

root = Path(sys.argv[1])
result = {}
for run in sorted((root / 'runs').iterdir()):
    if not (run / 'config.json').exists():
        continue
    files = sorted((run / 'wandb').glob('*run-*/run-*.wandb'))
    if not files:
        continue
    rows = []
    for path in files:
        ds = DataStore()
        ds.open_for_scan(str(path))
        while True:
            blob = ds.scan_data()
            if blob is None:
                break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(blob)
            if record.HasField('history'):
                row = {}
                for item in record.history.item:
                    key = item.key or '/'.join(item.nested_key)
                    try:
                        value = json.loads(item.value_json)
                    except (TypeError, ValueError):
                        continue
                    if isinstance(value, (float, int)) and math.isfinite(value):
                        row[key] = value
                if row:
                    rows.append(row)
    result[run.name] = dict(config=json.loads((run / 'config.json').read_text()),
                            history=rows, source_files=[str(p) for p in files])
print(json.dumps(result, allow_nan=False))
