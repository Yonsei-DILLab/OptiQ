from pathlib import Path
import json,base64,hashlib,concurrent.futures,sys,time
from preflight import ssh,ROOT
P=Path(__file__).resolve().parent;h=int(sys.argv[1]);raw=(P/f'stage-host{h}.tgz').read_bytes();parts=[raw[i:i+48000] for i in range(0,len(raw),48000)]
def one(item):
 i,b=item
 code='from pathlib import Path\nimport base64,json\np=Path('+repr(ROOT+'/nm-ablation-20261112/transfer')+');p.mkdir(parents=True,exist_ok=True)\n(p/'+repr(f'part{i:03d}')+').write_bytes(base64.b64decode('+repr(base64.b64encode(b).decode())+'))\nprint("{}")'
 for attempt in range(3):
  try:return ssh(h,code,timeout=45)
  except Exception:
   if attempt==2:raise
   time.sleep(1)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(one,enumerate(parts)))
code='ROOT='+repr(ROOT)+'\nHASH='+repr(hashlib.sha256(raw).hexdigest())+'\n'+r'''
from pathlib import Path
import hashlib,tarfile,io,json
p=Path(ROOT);raw=b''.join(x.read_bytes() for x in sorted((p/'nm-ablation-20261112/transfer').glob('part*')))
assert hashlib.sha256(raw).hexdigest()==HASH
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as t:t.extractall(p,filter='data')
print(json.dumps(dict(passed=True,bytes=len(raw))))
'''
print(ssh(h,code))
