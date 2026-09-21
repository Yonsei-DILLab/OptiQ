"""Cancel only the resolved v10 campaign; preserve data and DIPO."""
import json
from pathlib import Path
import re
import subprocess
import time

def status():
    p=subprocess.run(['supervisorctl','status'],text=True,capture_output=True)
    return {line.split()[0]:line for line in p.stdout.splitlines() if line.strip()}

root=Path('/workspace/optiq-v10-fixed-20260920')
before=status()
plan=json.loads((root/'plan.json').read_text())
names=[root.name+f'-gpu{g}' for g in plan['gpus']]
assert all(n in before and n.startswith('optiq-v10-fixed-') for n in names)
audit=root/'cancelled-for-direct-mll64.json'
assert not audit.exists(), 'Inspect previous cancellation before retrying'
audit.write_text(json.dumps(dict(time=time.time(),services=before,
    states={p.name:json.loads(p.read_text()) for p in (root/'state').glob('*.json')}),indent=2))
(root/'CANCELLED').write_text('Replaced by user-requested heejoon direct MLL 64x64. Outputs retained.\n')
for name in names:
    cfg=Path('/etc/supervisor/conf.d')/(name+'.conf')
    content=cfg.read_text();assert str(root) in content
    cfg.write_text(re.sub(r'^autostart=true$','autostart=false',content,flags=re.M))
    if before[name].split()[1] in {'RUNNING','STARTING','STOPPING','BACKOFF'}:
        subprocess.run(['supervisorctl','stop',name],check=True)
subprocess.run(['supervisorctl','reread'],check=True,stdout=subprocess.DEVNULL)
for name in names:subprocess.run(['supervisorctl','update',name],check=True,stdout=subprocess.DEVNULL)
after=status()
for name,line in before.items():
    if name.startswith('dipo') and 'RUNNING' in line:
        pid=re.search(r'pid (\d+)',line).group(1)
        assert 'RUNNING' in after[name] and f'pid {pid},' in after[name]
print(json.dumps(dict(cancelled=names,dipo_preserved=True)))
