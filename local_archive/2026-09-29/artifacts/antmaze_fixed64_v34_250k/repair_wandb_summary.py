"""Repair only a finished run's stale API summary from its uploaded summary file.

No run resume, history write, config mutation, training or evaluation is performed.
The original before/after values are retained in a separate provenance sidecar.
"""
from pathlib import Path
import hashlib
import json
import shlex
import subprocess

OUT = Path(__file__).resolve().parent
REMOTE_CODE = r'''
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, tempfile, wandb
root=Path('/home/heechan/optiq-experiments/antmaze-optiq-fixed64-v34-250k-s0-20260925')
runid='1727hgdu'
job='v4-optiq-geodesic-T1-teacherfloor0.5-fixed64-250k-s0'
source='9b733bad9d52ba73fad01f03521f18a211d21a1e'
result_path=root/'runs'/job/'result.json'
result=json.loads(result_path.read_text())
status=json.loads((root/'status.json').read_text())
assert result['completed'] and result['source_commit']==source
assert not status['running'] and not status['pending'] and not status['failed']
assert result['steps']==258304 and result['updates']==7816
assert result['checkpoint']['readback_verified']
api=wandb.Api(timeout=60)
run=api.run('OptiQ/antmaze/'+runid)
assert run.state=='finished'
before=dict(run.summary_metrics)
with tempfile.TemporaryDirectory() as tmp:
    f=run.file('wandb-summary.json').download(root=tmp,replace=True)
    raw=Path(f.name).read_bytes()
    uploaded=json.loads(raw)
assert uploaded['completed'] and uploaded['steps']==result['steps']
assert uploaded['updates']==result['updates']
assert uploaded['final_evaluations']==result['summaries']
patch={k:v for k,v in uploaded.items() if k.startswith('final/') or k in
       ('completed','steps','step','global_steps','updates','final_evaluations')}
for label, summary in result['summaries'].items():
    for key in ('success_rate','mean_return','episodes','goal_counts'):
        assert patch[f'final/{label}/{key}']==summary[key]
changed={k:v for k,v in patch.items() if before.get(k)!=v}
sidecar=root/'wandb-summary-repair-1727hgdu.json'
record=dict(time_utc=datetime.now(timezone.utc).isoformat(),run_id=runid,
    source_commit=source,reporting_script_sha256=SCRIPT_SHA,
    result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
    uploaded_summary_sha256=hashlib.sha256(raw).hexdigest(),before=before,
    patch=patch,changed_keys=sorted(changed),training_restarted=False,
    history_changed=False,config_changed=False,frozen_source_changed=False)
# Preserve this attempt before the one authorized metadata write.
assert not sidecar.exists(), 'Inspect prior repair instead of repeating it'
sidecar.write_text(json.dumps(record,indent=2)+'\n')
if changed:
    run.summary.update(patch)
fresh=wandb.Api(timeout=60).run('OptiQ/antmaze/'+runid)
after=fresh.summary_metrics
for key,value in patch.items():
    if key.endswith('/goal_counts'):
        # W&B can expose dictionary values as dotted scalar summary keys.
        assert after.get(key)==value or all(after.get(key+'.'+str(k))==v for k,v in value.items())
    elif key=='final_evaluations':
        # The backend similarly flattens this display-only dictionary.
        continue
    else:
        assert after.get(key)==value,(key,after.get(key),value)
assert fresh.state=='finished'
record.update(api_verified=True,after=after,state=fresh.state)
sidecar.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
'''


if __name__ == '__main__':
    code = 'SCRIPT_SHA=' + repr(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()) + '\n' + REMOTE_CODE
    command = 'bash -c ' + shlex.quote(
        'source /home/heechan/OptiQ-ops/activate.sh v5-direct-gmm >/dev/null; '
        'exec /home/heechan/.venv-ddiffpg-native/bin/python -')
    process = subprocess.run(['ssh','-o','BatchMode=yes','vast-heechan-199',command],
        input=code,text=True,capture_output=True,timeout=120)
    if process.returncode:
        raise RuntimeError(process.stderr[-2000:])
    record = json.loads(process.stdout.splitlines()[-1])
    (OUT/'wandb-summary-repair-1727hgdu.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:record[k] for k in
          ('run_id','state','api_verified','changed_keys','training_restarted','history_changed')}))
