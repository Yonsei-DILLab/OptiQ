"""Final read-only SHA256 comparison before the authorized stop-only action.

No instance lifecycle API is called here. Full payloads stay off the laptop.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import subprocess
import time

from .release_monitor import SERVERS, ready, snapshot

REMOTE_ROOT = '/workspace/heejoon-adroit-online-20261112'
STORAGE_ROOT = '/data1/heejoonorm/OptiQ/runs/adroit-online-20261112'
SSH = ['ssh', '-S', 'none', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
       '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3']
HASH_CODE = r'''
import hashlib,json,sys,time
from pathlib import Path
request=json.load(sys.stdin);root=Path(request['root']);names=request.get('names')
if names is None:
 names=[]
 for top in root.iterdir():
  if top.name in ('venv','setup'):continue
  if top.is_dir() and not (top.name.startswith('source-') or top.name in ('runs','preflight','process-logs')):continue
  for p in (top.rglob('*') if top.is_dir() else [top]):
   if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.tmp','.pyc'):
    names.append(str(p.relative_to(root)))
result={};missing=[]
for name in sorted(names):
 p=root/name
 assert not Path(name).is_absolute() and '..' not in Path(name).parts
 if not p.is_file():missing.append(name);continue
 assert p.resolve().is_relative_to(root.resolve()), ('External symlink',name)
 before=p.stat()
 if request.get('stable'):assert time.time()-before.st_mtime>=120, ('Recently changed',name)
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns), ('Changed during hash',name)
 result[name]={'bytes':after.st_size,'sha256':h.hexdigest()}
if request.get('source_manifests'):
 for folder in root.glob('source-*'):
  proof=json.loads((folder/'source-provenance.json').read_text())
  assert folder.name=='source-'+proof['commit']
  for name,digest in proof['files'].items():assert result[folder.name+'/'+name]['sha256']==digest,(folder.name,name)
print(json.dumps({'files':result,'missing':missing,'root':str(root),'finished_unix':time.time()}))
'''


def compare(source, stored):
    assert source['files'] and not source['missing'] and not stored['missing'], 'Missing archive files'
    assert source['files'] == stored['files'], 'Archive size/SHA256 mismatch'
    return dict(passed=True, files=len(source['files']),
                bytes=sum(v['bytes'] for v in source['files'].values()))


def audit_host(server, output):
    instance, host, port = server; shard = 0 if instance == 60326900 else 1
    origin = [*SSH, '-p', str(port), 'heejoonorm@' + host]
    command = 'python3 -c ' + shlex.quote(HASH_CODE)
    source = json.loads(subprocess.check_output([*origin, command],
        input=json.dumps(dict(root=REMOTE_ROOT, stable=True, source_manifests=True)), text=True))
    (output / f'shard{shard}-source-hashes.json').write_text(json.dumps(source, indent=2) + '\n')
    stored = json.loads(subprocess.check_output([*SSH, 'dildata', command],
        input=json.dumps(dict(root=f'{STORAGE_ROOT}/shard{shard}', names=list(source['files']))), text=True))
    (output / f'shard{shard}-stored-hashes.json').write_text(json.dumps(stored, indent=2) + '\n')
    receipt = dict(compare(source, stored), instance_id=instance, shard=shard,
                   destination=f'dildata:{STORAGE_ROOT}/shard{shard}', verified_unix=time.time())
    print(json.dumps(receipt), flush=True)
    return receipt


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True); a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    assert plan['final_action'] == 'stop_only' and plan['destroy_authorized'] is False
    a.output.mkdir(parents=True, exist_ok=True)
    code = Path(__file__).with_name('release_probe.py').read_text()
    with ThreadPoolExecutor(max_workers=2) as pool:
        reports = list(pool.map(lambda server: snapshot(server, code), SERVERS))
    assert ready(reports), 'Both instances must pass a fresh completion check'
    (a.output / 'completion-before-audit.json').write_text(json.dumps(reports, indent=2) + '\n')
    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(pool.map(lambda server: audit_host(server, a.output), SERVERS))
    with ThreadPoolExecutor(max_workers=2) as pool:
        final = list(pool.map(lambda server: snapshot(server, code), SERVERS))
    assert ready(final), 'Completion state changed during archival verification'
    result = dict(passed=True, final_action='stop_only', destroy_authorized=False,
                  verified_unix=time.time(), receipts=receipts, completion=final)
    (a.output / 'verified.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(event='backup_verified', **{k:v for k,v in result.items() if k!='completion'})), flush=True)


if __name__ == '__main__':
    main()
