"""Relay changed experiment outputs to dildata without transferring SSH keys.

Run on the authorized local host. SSH archive bytes stream directly to storage;
checkpoints are not materialized on the laptop. No files are deleted remotely.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

SERVERS = [('175.155.64.205', 19159), ('179.255.127.236', 31020)]
REMOTE_ROOT = '/workspace/heejoon-adroit-online-20261112'
STORAGE_ROOT = '/data1/heejoonorm/OptiQ/runs/adroit-online-20261112'


def archive_once(state_dir):
    state_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for shard, (host, port) in enumerate(SERVERS):
        ssh = ['ssh', '-p', str(port), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
               f'heejoonorm@{host}']
        inventory = '''
import json,time
from pathlib import Path
root=Path(%r); now=time.time(); files={}
for top in root.iterdir():
 if top.name=='venv' or top.name=='setup': continue
 if top.is_dir() and not (top.name.startswith('source-') or top.name in ('runs','preflight','process-logs')): continue
 for f in (top.rglob('*') if top.is_dir() else [top]):
  if not f.is_file() or '__pycache__' in f.parts or f.suffix in ('.tmp','.pyc'): continue
  s=f.stat()
  if ('checkpoints' in f.parts or f.suffix in ('.zip','.pkl','.msgpack')) and now-s.st_mtime<90: continue
  files[str(f.relative_to(root))]=[s.st_size,s.st_mtime_ns]
print(json.dumps(files))
''' % REMOTE_ROOT
        files = json.loads(subprocess.check_output([*ssh, 'python3 -c ' + shlex.quote(inventory)], text=True))
        state_file = state_dir / f'shard{shard}.json'
        old = json.loads(state_file.read_text()) if state_file.exists() else {}
        changed = sorted(name for name, stat in files.items() if old.get(name) != stat)
        if not changed:
            reports.append(dict(shard=shard, changed=0)); continue
        destination = f'{STORAGE_ROOT}/shard{shard}'
        receive = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', 'dildata',
            f'mkdir -p {shlex.quote(destination)} && tar -xf - -C {shlex.quote(destination)}'],
            stdin=subprocess.PIPE)
        send = subprocess.Popen([*ssh, f'tar -C {shlex.quote(REMOTE_ROOT)} -cf - --null -T -'],
                                stdin=subprocess.PIPE, stdout=receive.stdin)
        receive.stdin.close()
        send.stdin.write(b''.join(name.encode() + b'\0' for name in changed))
        send.stdin.close()
        sent, received = send.wait(), receive.wait()
        if sent != 0 or received != 0:
            raise RuntimeError(('archive transfer failed; inventory not advanced', shard, sent, received))
        immutable = [n for n in changed if 'checkpoints/' in n or n.endswith(('.zip', '.msgpack'))]
        if immutable:
            script = '''
import hashlib,json
from pathlib import Path
root=Path(%r); names=%r; hashes={}
for n in names:
 h=hashlib.sha256()
 with (root/n).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
 hashes[n]=h.hexdigest()
print(json.dumps(hashes))
'''
            original = json.loads(subprocess.check_output([*ssh, 'python3 -c ' +
                shlex.quote(script % (REMOTE_ROOT, immutable))], text=True))
            stored = json.loads(subprocess.check_output(['ssh', '-o', 'BatchMode=yes', 'dildata',
                'python3 -c ' + shlex.quote(script % (destination, immutable))], text=True))
            assert original == stored, 'Stored checkpoint hash mismatch'
            (state_dir / f'shard{shard}-last-checkpoint-hashes.json').write_text(json.dumps(stored, indent=2))
        tmp = state_file.with_suffix('.tmp')
        tmp.write_text(json.dumps(files)); tmp.replace(state_file)
        reports.append(dict(shard=shard, changed=len(changed), verified_checkpoints=len(immutable),
                            bytes=sum(files[n][0] for n in changed), destination=destination))
    return reports


def main():
    p = argparse.ArgumentParser(); p.add_argument('--state-dir', type=Path, required=True)
    p.add_argument('--once', action='store_true'); a = p.parse_args()
    while True:
        try:
            print(json.dumps(dict(time=time.time(), transfers=archive_once(a.state_dir))), flush=True)
        except Exception as error:
            print(json.dumps(dict(time=time.time(), error=repr(error))), flush=True)
            if a.once: raise
        if a.once: break
        finished = []
        for host, port in SERVERS:
            try:
                raw = subprocess.check_output(['ssh', '-p', str(port), '-o', 'BatchMode=yes',
                    '-o', 'ConnectTimeout=15', f'heejoonorm@{host}',
                    'python3 -c ' + shlex.quote(
                        "from pathlib import Path; p=Path(%r); f=p/'priority-queue-state.json'; "
                        "print((f if f.exists() else p/'queue-state.json').read_text())" % REMOTE_ROOT)], text=True,
                    stderr=subprocess.DEVNULL)
                jobs = json.loads(raw)['jobs']
                finished.append(len(jobs) >= 24 and all(v['status'] in
                    ('completed', 'failed', 'requires_review') for v in jobs.values()))
            except Exception:
                finished.append(False)
        if all(finished):
            # Include newly written final checkpoints after the stability interval.
            time.sleep(120)
            print(json.dumps(dict(final=True, transfers=archive_once(a.state_dir))), flush=True)
            break
        time.sleep(300)


if __name__ == '__main__':
    main()
