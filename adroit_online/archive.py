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
SSH = ['ssh', '-S', 'none', '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3']


def archive_once(state_dir, cohort=None):
    state_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for shard, (host, port) in enumerate(SERVERS):
        ssh = [*SSH, '-p', str(port), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
               f'heejoonorm@{host}']
        inventory = '''
import json,time
from pathlib import Path
root=Path(%r); now=time.time(); files={}; cohort=%r
cohort_source=json.loads((root/'temp001-plan.json').read_text())['source_commit'] if cohort else None
for top in root.iterdir():
 if top.name=='venv' or top.name=='setup': continue
 if top.is_dir() and not (top.name.startswith('source-') or top.name in ('runs','preflight','process-logs','device-checks')): continue
 if cohort and top.name.startswith('source-') and top.name!='source-'+cohort_source: continue
 for f in (top.rglob('*') if top.is_dir() else [top]):
  if not f.is_file() or '__pycache__' in f.parts or f.suffix in ('.tmp','.pyc'): continue
  rel=f.relative_to(root); parts=rel.parts
  if cohort and not (parts[0]=='source-'+cohort_source or
      (len(parts)==1 and parts[0].startswith('temp001-')) or
      (parts[0]=='runs' and parts[1].endswith('-t001')) or
      (parts[0]=='preflight' and parts[1]=='temp001') or
      (parts[0]=='process-logs' and '-t001' in f.name)): continue
  s=f.stat()
  # Live append-only logs otherwise invalidate an entire large tar transfer.
  # Defer all recently modified files; the final stable pass includes them.
  if now-s.st_mtime<120: continue
  files[str(f.relative_to(root))]=[s.st_size,s.st_mtime_ns]
print(json.dumps(files))
''' % (REMOTE_ROOT, cohort)
        files = json.loads(subprocess.check_output([*ssh, 'python3 -c ' + shlex.quote(inventory)], text=True))
        state_file = state_dir / f'shard{shard}.json'
        old = json.loads(state_file.read_text()) if state_file.exists() else {}
        changed = sorted(name for name, stat in files.items() if old.get(name) != stat)
        if not changed:
            reports.append(dict(shard=shard, changed=0)); continue
        destination = f'{STORAGE_ROOT}' + ('/temp001' if cohort else '') + f'/shard{shard}'
        receive = subprocess.Popen([*SSH, '-o', 'BatchMode=yes', 'dildata',
            f'mkdir -p {shlex.quote(destination)} && tar -xzf - -C {shlex.quote(destination)}'],
            stdin=subprocess.PIPE)
        send = subprocess.Popen([*ssh, f'tar -C {shlex.quote(REMOTE_ROOT)} -czf - --null -T -'],
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
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); names=json.load(sys.stdin); hashes={}
for n in names:
 h=hashlib.sha256()
 with (root/n).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
 hashes[n]=h.hexdigest()
print(json.dumps(hashes))
'''
            # Send the potentially large filename list over stdin, not argv.
            original = json.loads(subprocess.check_output([*ssh, 'python3 -c ' +
                shlex.quote(script) + ' ' + shlex.quote(REMOTE_ROOT)],
                input=json.dumps(immutable), text=True))
            stored = json.loads(subprocess.check_output([*SSH, '-o', 'BatchMode=yes', 'dildata',
                'python3 -c ' + shlex.quote(script) + ' ' + shlex.quote(destination)],
                input=json.dumps(immutable), text=True))
            assert original == stored, 'Stored checkpoint hash mismatch'
            (state_dir / f'shard{shard}-last-checkpoint-hashes.json').write_text(json.dumps(stored, indent=2))
        tmp = state_file.with_suffix('.tmp')
        tmp.write_text(json.dumps(files)); tmp.replace(state_file)
        reports.append(dict(shard=shard, changed=len(changed), verified_checkpoints=len(immutable),
                            bytes=sum(files[n][0] for n in changed), destination=destination))
    return reports


def main():
    p = argparse.ArgumentParser(); p.add_argument('--state-dir', type=Path, required=True)
    p.add_argument('--cohort', choices=('temp001',))
    p.add_argument('--once', action='store_true'); a = p.parse_args()
    while True:
        try:
            print(json.dumps(dict(time=time.time(), transfers=archive_once(a.state_dir, a.cohort))), flush=True)
        except Exception as error:
            print(json.dumps(dict(time=time.time(), error=repr(error))), flush=True)
            if a.once: raise
        if a.once: break
        finished = []
        for host, port in SERVERS:
            try:
                raw = subprocess.check_output([*SSH, '-p', str(port), '-o', 'BatchMode=yes',
                    '-o', 'ConnectTimeout=15', f'heejoonorm@{host}',
                    'python3 -c ' + shlex.quote(
                        "from pathlib import Path; p=Path(%r); cohort=%r; "
                        "f=p/('temp001-queue-state.json' if cohort else 'priority-queue-state.json'); "
                        "import json; q=json.loads((f if f.exists() else p/'queue-state.json').read_text()); "
                        "r=p/'recovery-queue-state.json'; "
                        "q['jobs'].update(json.loads(r.read_text())['jobs'] if r.exists() and not cohort else {}); "
                        "t=p/'temp001-queue-state.json'; "
                        "q['jobs'].update(json.loads(t.read_text())['jobs'] if t.exists() and not cohort else {}); "
                        "print(json.dumps(q))" % (REMOTE_ROOT, a.cohort))], text=True,
                    stderr=subprocess.DEVNULL)
                jobs = json.loads(raw)['jobs']
                finished.append(len(jobs) >= (4 if a.cohort else 24) and all(v['status'] in
                    ('completed', 'failed', 'requires_review') for v in jobs.values()))
            except Exception:
                finished.append(False)
        if all(finished):
            # Include newly written final checkpoints after the stability interval.
            time.sleep(120)
            print(json.dumps(dict(final=True, transfers=archive_once(a.state_dir, a.cohort))), flush=True)
            break
        time.sleep(300)


if __name__ == '__main__':
    main()
