"""Correct only the finished continuation runs' legacy W&B job-type label."""
from pathlib import Path
import json, datetime
import wandb

root = Path('/home/heechan/optiq-experiments/gmm40-mu90-256x2-500k-20260921')
records = []
for d in sorted((root / 'results').glob('mu90_256x2_*_500k')):
    audit = json.loads((d / 'update_count_audit.json').read_text())
    assert audit['status'] == 'passed' and audit['actor_updates'] == 500000
    url = json.loads((d / 'wandb_status.json').read_text())['url']
    path = 'OptiQ/gmm-trg/' + url.rsplit('/', 1)[-1]
    run = wandb.Api().run(path)
    assert run.state == 'finished' and run.name == d.name
    assert run.config['steps'] == 500000, run.config
    before = run.job_type
    run.job_type = 'gmm40-fixed-q'
    run.update()
    after = wandb.Api().run(path).job_type
    assert after == 'gmm40-fixed-q'
    records.append(dict(url=url, name=d.name, before=before, after=after))
out = dict(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
           scope='Post-launch W&B metadata only; frozen source/history unchanged', runs=records)
(root / 'wandb_metadata_correction.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out, indent=2))
