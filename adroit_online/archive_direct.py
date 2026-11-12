"""Direct Vast-to-dildata archives using a short-lived constrained local agent.

Only authentication signatures traverse the laptop; no private key is copied.
Each pass opens an authenticated storage connection before the key expires.
No authorized_keys, SSH config, training source, or instance lifecycle changes.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import time

from .release_monitor import SERVERS

ROOT = '/workspace/heejoon-adroit-online-20261112'
STORAGE = '/data1/heejoonorm/OptiQ/runs/adroit-online-20261112'
DESTINATION = '[dil.yonsei.ac.kr]:2200'

REMOTE = r'''
import hashlib,json,os,shlex,subprocess,sys,tempfile,time
from pathlib import Path
r=json.load(sys.stdin)
assert int(os.environ['CONTAINER_ID'])==r['instance']
assert r['instance'] in (60326900,60327673)
root=Path('/workspace/heejoon-adroit-online-20261112')
shard=0 if r['instance']==60326900 else 1
destination='/data1/heejoonorm/OptiQ/runs/adroit-online-20261112/shard'+str(shard)
files={};now=time.time()
for top in root.iterdir():
 if top.name in ('venv','setup'):continue
 if top.is_dir() and not (top.name.startswith('source-') or top.name in ('runs','preflight','process-logs')):continue
 for f in (top.rglob('*') if top.is_dir() else [top]):
  if not f.is_file() or '__pycache__' in f.parts or f.suffix in ('.tmp','.pyc'):continue
  if r.get('essential') and not top.name.startswith('source-') and 'checkpoints' in f.relative_to(root).parts:
   if f.name not in ('policy-003000064.zip','actor_state_3000064.msgpack','critic_state_3000064.msgpack','auxiliary-state-003000064.pkl'):continue
  assert f.resolve().is_relative_to(root.resolve())
  s=f.stat()
  if now-s.st_mtime<120:continue
  files[str(f.relative_to(root))]=[s.st_size,s.st_mtime_ns]
changed=sorted(n for n,s in files.items() if r['old'].get(n)!=s)
hashes={};start=time.time()
with tempfile.TemporaryDirectory(prefix='adroit-direct-') as td:
 known=Path(td)/'known_hosts';known.write_text(r['known_hosts'])
 ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15',
      '-o','ServerAliveCountMax=3','-o','StrictHostKeyChecking=yes',
      '-o','UserKnownHostsFile='+str(known),'-o','ControlMaster=auto',
      '-o','ControlPersist=600','-o','ControlPath='+td+'/master',
      '-p','2200','heejoonorm@dil.yonsei.ac.kr']
 subprocess.run([*ssh,'true'],check=True,timeout=30)
 try:
  if r.get('probe'):
   with (root/('runs/door-sac-s'+str(4+shard)+'-e256-u025/checkpoints/final-replay.pkl')).open('rb') as f:
    payload=f.read(16*1024*1024)
   t=time.time();subprocess.run([*ssh,'cat > /dev/null'],input=payload,check=True,timeout=90)
   print(json.dumps(dict(probe=True,instance=r['instance'],bytes=len(payload),seconds=time.time()-t)),flush=True)
   sys.exit(0)
  reused={}
  if changed:
   salvage="""import hashlib,json,sys
from pathlib import Path
r=json.load(sys.stdin);root=Path(r['root']);out={}
for n,size in r['names'].items():
 assert not Path(n).is_absolute() and '..' not in Path(n).parts
 p=root/n
 if not p.is_file() or p.stat().st_size!=size:continue
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 out[n]=h.hexdigest()
print(json.dumps(out))
"""
   candidates=json.loads(subprocess.check_output([*ssh,'python3 -c '+shlex.quote(salvage)],
       input=json.dumps(dict(root=destination,names={n:files[n][0] for n in changed})),text=True))
   for n,digest in candidates.items():
    assert n in changed
    h=hashlib.sha256();p=root/n;before=p.stat()
    with p.open('rb') as f:
     for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    after=p.stat()
    if [after.st_size,after.st_mtime_ns]==files[n] and (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns) and h.hexdigest()==digest:
     reused[n]=digest
   changed=[n for n in changed if n not in reused]
  if changed:
   receive=subprocess.Popen([*ssh,'mkdir -p '+shlex.quote(destination)+' && tar -xzf - -C '+shlex.quote(destination)],stdin=subprocess.PIPE)
   send=subprocess.Popen(['tar','-C',str(root),'-czf','-','--null','-T','-'],stdin=subprocess.PIPE,stdout=receive.stdin)
   receive.stdin.close()
   send.stdin.write(b''.join(n.encode()+b'\0' for n in changed));send.stdin.close()
   sent,received=send.wait(),receive.wait()
   assert (sent,received)==(0,0), ('Archive failed; inventory unchanged',sent,received)
   names=[n for n in changed if 'checkpoints/' in n or n.endswith(('.zip','.msgpack'))]
   for n in names:
    h=hashlib.sha256()
    with (root/n).open('rb') as f:
     for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    hashes[n]=h.hexdigest()
   code="""import hashlib,json,sys
from pathlib import Path
r=json.load(sys.stdin);root=Path(r['root']);out={}
for n in r['names']:
 assert not Path(n).is_absolute() and '..' not in Path(n).parts
 h=hashlib.sha256()
 with (root/n).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 out[n]=h.hexdigest()
print(json.dumps(out))
"""
   stored=json.loads(subprocess.check_output([*ssh,'python3 -c '+shlex.quote(code)],input=json.dumps(dict(root=destination,names=names)),text=True))
   assert stored==hashes, 'Stored checkpoint hash mismatch'
  hashes.update({n:h for n,h in reused.items() if 'checkpoints/' in n})
  print(json.dumps(dict(instance=r['instance'],shard=shard,files=files,hashes=hashes,
    scope='essential' if r.get('essential') else 'full',
    changed=len(changed),bytes=sum(files[n][0] for n in changed),destination=destination,
    verified_reused=len(reused),reused_bytes=sum(files[n][0] for n in reused),
    seconds=time.time()-start,finished_unix=time.time())),flush=True)
 finally:
  subprocess.run([*ssh[:-1],'-O','exit',ssh[-1]],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
'''


def constraint_args(names):
    result = []
    for name in names:
        result += ['-h', 'heejoonorm@' + name,
                   '-h', name + '>heejoonorm@' + DESTINATION]
    return result


def archive_one(server, state_dir, env, destination_keys, probe=False, essential=False):
    instance, host, port = server
    shard = 0 if instance == 60326900 else 1
    state = state_dir / f'shard{shard}.json'
    old = json.loads(state.read_text()) if state.exists() else {}
    request = dict(instance=instance, old=old, known_hosts=destination_keys, probe=probe,
                   essential=essential)
    command = ['ssh', '-S', 'none', '-A', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
               '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3',
               '-o', 'IdentityAgent=' + env['SSH_AUTH_SOCK'], '-p', str(port),
               'heejoonorm@' + host, 'python3 -c ' + shlex.quote(REMOTE)]
    process = subprocess.run(command, env=env, input=json.dumps(request), text=True, capture_output=True)
    if process.returncode:
        raise RuntimeError(f'Direct archive {instance} failed: ' + process.stderr[-3000:])
    result = json.loads(process.stdout)
    if not probe:
        temporary = state.with_suffix('.tmp')
        temporary.write_text(json.dumps(result.pop('files'))); temporary.replace(state)
        hashes = result.pop('hashes')
        if hashes:
            (state_dir / f'shard{shard}-last-checkpoint-hashes.json').write_text(json.dumps(hashes, indent=2))
        (state_dir / f'shard{shard}-direct-receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument('--state-dir', type=Path, required=True)
    p.add_argument('--probe', action='store_true'); p.add_argument('--shard', type=int, choices=(0, 1))
    p.add_argument('--essential', action='store_true')
    a = p.parse_args(); a.state_dir.mkdir(parents=True, exist_ok=True)
    servers = SERVERS if a.shard is None else [SERVERS[a.shard]]
    names = [f'[{s[1]}]:{s[2]}' for s in servers]
    known = Path.home() / '.ssh/known_hosts'
    keys = {n: subprocess.check_output(['ssh-keygen', '-F', n, '-f', str(known)], text=True)
            for n in [*names, DESTINATION]}
    with tempfile.TemporaryDirectory(prefix='adroit-agent-') as td:
        sock = str(Path(td) / 'agent.sock'); kh = Path(td) / 'known_hosts'
        kh.write_text(''.join(keys.values()))
        agent = subprocess.Popen(['ssh-agent', '-D', '-a', sock],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            for _ in range(50):
                if Path(sock).exists(): break
                time.sleep(.1)
            env = {**os.environ, 'SSH_AUTH_SOCK': sock, 'SSH_ASKPASS_REQUIRE': 'never'}
            constraints = constraint_args(names)
            subprocess.run(['ssh-add', '-q', '-t', '120', '-H', str(kh),
                            *constraints, str(Path.home() / '.ssh/id_ed25519')],
                           env=env, check=True, stdin=subprocess.DEVNULL, timeout=10)
            with ThreadPoolExecutor(max_workers=len(servers)) as pool:
                list(pool.map(lambda s: archive_one(s, a.state_dir, env, keys[DESTINATION], a.probe, a.essential), servers))
        finally:
            agent.terminate(); agent.wait(timeout=10)


if __name__ == '__main__':
    main()
