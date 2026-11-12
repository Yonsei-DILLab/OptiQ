from central_collector import *

def ack(n):
 try:
  rows=json.loads(remote(n,'checkpoint-manifest'));verified=[]
  for row in rows:
   p=ROOT/f'hosts/host{n}/workspace'/row['path']
   if p.is_file() and p.stat().st_size==row['size'] and digest(p)==row['sha256']:verified.append(row)
  result={}
  if verified:
   (ROOT/f'incremental-verified-host{n}-{int(time.time())}.json').write_text(json.dumps(dict(files=verified,at=time.time())))
   result=json.loads(remote(n,'archive-prune',verified))
  out=dict(host=n,at=time.time(),passed=True,**result)
 except Exception as e:out=dict(host=n,at=time.time(),passed=False,error=repr(e))
 (ROOT/f'incremental-ack-host{n}.json').write_text(json.dumps(out));print(json.dumps(out),flush=True)
while True:
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as e:list(e.map(ack,[3,4,5]))
 time.sleep(45)
