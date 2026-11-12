"""Pull immutable evidence, verify SHA256, then acknowledge moving older checkpoint copies."""
from pathlib import Path
import subprocess,json,hashlib,time,concurrent.futures,shlex,traceback
ROOT=Path('/data1/heechan/iBOLT/20261112-f1-a1-temp');ROOT.mkdir(parents=True,exist_ok=True)
HOSTS={2:('114.32.64.6',41188),1:('122.117.118.68',41261),3:('211.72.13.201',42866),4:('175.155.64.199',19280),5:('211.72.13.201',42757)}
NEW='heechan-unified-f1-a1-20261112'
def ssh(n):
 ip,port=HOSTS[n];return (['ssh','-o','KexAlgorithms=curve25519-sha256'] if n==2 else ['ssh']) + ['-i',str(Path.home()/'.ssh/ibolt_archive_20261112'),'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','UserKnownHostsFile='+str(Path.home()/'.ssh/ibolt_archive_known_hosts'),'-p',str(port)]
def remote(n,command,data=None):
 r=subprocess.run(ssh(n)+['heechan@'+HOSTS[n][0],command],input=None if data is None else json.dumps(data),text=True,capture_output=True,timeout=240)
 if r.returncode:raise RuntimeError(r.stderr[-1000:])
 return r.stdout

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()

def sync(n,folder):
 dest=ROOT/f'hosts/host{n}/workspace'/folder;dest.mkdir(parents=True,exist_ok=True)
 cmd=['rsync','-a','--partial','--timeout=180','--exclude=venv*','--exclude=.venv*','--exclude=__pycache__','--exclude=.git','--exclude=*.tmp','--exclude=*.lock','-e',shlex.join(ssh(n)),'heechan@'+HOSTS[n][0]+':'+folder+'/',str(dest)+'/']
 with (ROOT/f'rsync-host{n}.log').open('a') as log:
  r=subprocess.run(cmd,stdout=log,stderr=log,timeout=7200)
 if r.returncode not in (0,24):raise RuntimeError(f'rsync {folder} exit {r.returncode}')
 return dest

def once(n,initial=False):
 out=dict(host=n,started=time.time(),initial=initial)
 try:
  folders=[NEW]
  if initial:folders+=['heechan-adroit']+(['heechan-fetch-dacer/outputs'] if n==3 else [])
  for folder in folders:sync(n,folder)
  # These candidates are old immutable checkpoints, never the two newest step groups.
  manifest=json.loads(remote(n,'checkpoint-manifest'))
  eligible=[]
  for row in manifest:
   p=ROOT/f'hosts/host{n}/workspace'/row['path']
   if p.is_file() and p.stat().st_size==row['size'] and digest(p)==row['sha256']:eligible.append(row)
  receipt=dict(at=time.time(),files=eligible,bytes=sum(r['size'] for r in eligible))
  (ROOT/f'host{n}-sha256-verified-{int(time.time())}.json').write_text(json.dumps(receipt))
  if eligible:out['moved']=json.loads(remote(n,'archive-prune',eligible))
  out.update(passed=True,finished=time.time(),verified_files=len(eligible))
 except Exception as e:out.update(passed=False,error=repr(e),traceback=traceback.format_exc())
 (ROOT/f'collector-host{n}-status.json').write_text(json.dumps(out,indent=2))
 with (ROOT/'collector-history.jsonl').open('a') as f:f.write(json.dumps(out)+'\n')
 print(json.dumps(out),flush=True);return out

def host_loop(n):
 initial=not (ROOT/f'initial-archive-host{n}.json').exists()
 while True:
  if (ROOT/'PAUSE_COLLECTOR').exists():time.sleep(20);continue
  r=once(n,initial)
  if initial and r['passed']:(ROOT/f'initial-archive-host{n}.json').write_text(json.dumps(r));initial=False
  time.sleep(60)
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as e:
  list(e.map(host_loop,[5,3,4]))
