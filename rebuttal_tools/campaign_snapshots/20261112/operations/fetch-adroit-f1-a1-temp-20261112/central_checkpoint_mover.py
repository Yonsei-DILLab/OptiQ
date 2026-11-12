from central_collector import *

def cycle(n):
 try:
  rows=json.loads(remote(n,'checkpoint-manifest'));groups={}
  for row in rows:
   p=ROOT/f'hosts/host{n}/workspace'/row['path']
   if p.is_file() and p.stat().st_size==row['size'] and digest(p)==row['sha256']:continue
   first,rest=row['path'].split('/',1);groups.setdefault(first,[]).append(rest)
  batches=[]
  for first,paths in groups.items():
   for i in range(4):
    subset=paths[i::4]
    if subset:batches.append((first,subset))
  def pull(batch):
   first,paths=batch;dest=ROOT/f'hosts/host{n}/workspace'/first;dest.mkdir(parents=True,exist_ok=True)
   cmd=['rsync','-a','--relative','--files-from=-','--timeout=180','-e',shlex.join(ssh(n)),'heechan@'+HOSTS[n][0]+':'+first+'/',str(dest)+'/']
   r=subprocess.run(cmd,input='\n'.join(paths)+'\n',text=True,capture_output=True,timeout=3600)
   if r.returncode not in (0,24):raise RuntimeError(r.stderr[-1000:])
  # Parallel streams allow space recovery while the full evidence mirror progresses.
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as e:list(e.map(pull,batches))
  ack=[]
  for row in rows:
   p=ROOT/f'hosts/host{n}/workspace'/row['path']
   if p.is_file() and p.stat().st_size==row['size'] and digest(p)==row['sha256']:ack.append(row)
  if ack:
   receipt=ROOT/f'checkpoint-transfer-host{n}-{int(time.time())}.json';receipt.write_text(json.dumps(dict(at=time.time(),files=ack)))
   result=json.loads(remote(n,'archive-prune',ack))
  else:result=dict(moved=0,bytes=0)
  out=dict(host=n,at=time.time(),passed=True,**result)
 except Exception as e:out=dict(host=n,at=time.time(),passed=False,error=repr(e))
 (ROOT/f'checkpoint-mover-host{n}.json').write_text(json.dumps(out));print(json.dumps(out),flush=True)

def loop(n):
 while True:cycle(n);time.sleep(90)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as e:list(e.map(loop,[5,3,4]))
