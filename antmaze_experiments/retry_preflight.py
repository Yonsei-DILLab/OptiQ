"""Explicit user-approved retry; archive failed attempts, never delete results."""
import json
from pathlib import Path
import subprocess

root=Path('/workspace/antmaze-temperature-20260924')
status=json.loads((root/'status.json').read_text())
assert status['phase']=='held_failure' and not status['running'] and not status['completed']
assert not any((root/'runs').iterdir()),'Do not overwrite real training'
for p in (root/'jobs').glob('*.json'):
    j=json.loads(p.read_text());assert j['phase']=='preflight' and j['status']=='failed'
archive=root/'attempts'/'missing-loguru'
archive.mkdir(parents=True,exist_ok=False)
for name in ('STARTED','status.json','jobs','logs','preflight'):
    (root/name).rename(archive/name)
for name in ('jobs','logs','preflight'):(root/name).mkdir()
manifest=json.loads((root/'manifest.json').read_text())
(root/'status.json').write_text(json.dumps(dict(phase='retry_authorized',pending=[j['id'] for j in manifest['jobs']])))
subprocess.run(['supervisorctl','start','antmaze-temperature-20260924'],check=True)
print('Archived failed preflight and restarted unchanged 27-run sweep shard')
