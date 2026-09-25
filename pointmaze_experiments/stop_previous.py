"""User-requested stop of known iBOLT learner/queue services; keep MaxEntDP."""
from pathlib import Path
import json
import subprocess
import re
import datetime

prefixes = ('ibolt-antmaze-env25-', 'ibolt-antmaze-env5-',
            'optiq-gmm-trg-20260921-gpu', 'optiq-nm-20260922-gpu',
            'optiq-nm256-20260925-gpu')
status = subprocess.run(['supervisorctl', 'status'], capture_output=True, text=True).stdout
targets = [line.split()[0] for line in status.splitlines()
           if line.split() and line.split()[0].startswith(prefixes)]
record = {'time': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'targets': targets, 'before': status, 'configs': []}
for path in Path('/etc/supervisor/conf.d').glob('*.conf'):
    original = path.read_text()
    matches = re.findall(r'^\[program:([^]]+)\]', original, re.M)
    if not matches or not all(name in targets for name in matches): continue
    backup = path.with_suffix('.conf.before-pointmaze-20260926')
    if not backup.exists(): backup.write_text(original)
    updated = re.sub(r'^(autostart|autorestart)\s*=.*$', r'\1=false', original, flags=re.M)
    path.write_text(updated)
    record['configs'].append(str(path))
for name in targets:
    subprocess.run(['supervisorctl', 'stop', name], check=False)
subprocess.run(['supervisorctl', 'reread'], check=False)
for name in targets:
    subprocess.run(['supervisorctl', 'update', name], check=False)
record['after'] = subprocess.run(['supervisorctl','status'],capture_output=True,text=True).stdout
Path('/tmp/ibolt-pointmaze-stop-20260926.json').write_text(json.dumps(record, indent=2))
print(json.dumps({'targets': targets, 'count': len(targets)}))
