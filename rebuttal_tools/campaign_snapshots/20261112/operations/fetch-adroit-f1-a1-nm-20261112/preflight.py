from pathlib import Path
import json,subprocess,concurrent.futures,base64,zlib,time
P=Path(__file__).resolve().parent
ROOT='/workspace/heechan-unified-f1-a1-20261112'
def ssh(h,code,timeout=55):
 blob=zlib.compress(code.encode());packed=base64.b64encode(blob).decode()
 # Vast proxy on host1/2 drops oversized exec requests; stage bounded chunks.
 if h in (1,2) and len(packed)>1000:
  import hashlib
  directory=ROOT+'/nm-ablation-20261112/command-payloads/'+hashlib.sha256(blob).hexdigest()
  chunks=[blob[i:i+384] for i in range(0,len(blob),384)]
  def put(item):
   i,data=item
   small='from pathlib import Path\nimport base64\np=Path('+repr(directory+'/'+str(i))+');p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode('+repr(base64.b64encode(data).decode())+'));print("{}")'
   return ssh(h,small,timeout=timeout)
  with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(put,enumerate(chunks)))
  code='from pathlib import Path\nimport zlib\np=Path('+repr(directory)+');exec(zlib.decompress(b"".join((p/str(i)).read_bytes() for i in range('+str(len(chunks))+'))))'
  packed=base64.b64encode(zlib.compress(code.encode())).decode()
 cmd='python3 -c '+__import__('shlex').quote("import base64,zlib;exec(zlib.decompress(base64.b64decode("+repr(packed)+")))")
 r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','KexAlgorithms=curve25519-sha256','-o','ConnectTimeout=15',f'ibolt-server-{h}',cmd],capture_output=True,text=True,timeout=timeout)
 if r.returncode: raise RuntimeError(r.stdout+r.stderr)
 return json.loads(r.stdout)
CODE=r'''
import json,subprocess,sys,hashlib,os,shutil
from pathlib import Path
p=Path('/workspace/heechan-unified-f1-a1-20261112');sys.path.insert(0,str(p));import controller
plan=json.loads((p/'plan.json').read_text());active=[j for j in plan['jobs'] if j['status']=='running']
g=controller.gpu_state()
for x in g:x['pids']=list(x['pids'])
print(json.dumps(dict(plan=plan,controller=(p/'controller.py').read_text(),hashes={n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in ['controller.py','archive_gateway.py']},gpu=g,free_cores=len(controller.cores(active)),meminfo=Path('/proc/meminfo').read_text().splitlines()[:4],free_gib=shutil.disk_usage(p).free/2**30,excluded=plan.get('excluded_gpus',[]),pause=(p/'PAUSE').exists(),selects={x.stem:json.loads(x.read_text()) for x in (p/'selections').glob('*.json')},archive_ready=json.loads((p/'archive-ready.json').read_text()))))
'''
def one(h):
 d=ssh(h,CODE);(P/f'preflight-host{h}.json').write_text(json.dumps(d,indent=2));print(json.dumps(dict(host=h,active=d['plan']['counts'].get('running',0),gpu=d['gpu'],free_cores=d['free_cores'],memory=d['meminfo'],disk=d['free_gib'],disk_gate=d['plan']['min_free_gib'],paused=d['pause'])),flush=True)
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:list(pool.map(one,range(1,6)))
