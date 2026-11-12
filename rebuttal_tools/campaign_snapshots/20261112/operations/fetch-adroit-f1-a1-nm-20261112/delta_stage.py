from pathlib import Path
import json,zlib,base64,hashlib,concurrent.futures,time,sys
from preflight import ssh,ROOT
P=Path(__file__).resolve().parent;h=int(sys.argv[1]);proto=json.loads((P/'protocol.json').read_text())
files={'nm-protocol.json':(P/'protocol.json').read_text(),'nm-ablation-20261112/additions.json':(P/f'additions-host{h}.json').read_text(),'nm-ablation-20261112/controller.py':(P/f'controller-host{h}.py').read_text()}
for v,r in proto['sources'].items():
 for name in r['changed_files']+['source-sha256.json']:
  files['nm-'+v+'-source/'+name]=(P/('nm-'+v+'-source')/name).read_text()
raw=zlib.compress(json.dumps(files).encode());parts=[raw[i:i+384] for i in range(0,len(raw),384)]
print('DELTA',h,len(raw),len(parts),flush=True)
def one(pair):
 i,b=pair;dest=ROOT+'/nm-ablation-20261112/delta/'+f'part{i:04d}'
 code='from pathlib import Path\nimport base64\np=Path('+repr(dest)+');p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode('+repr(base64.b64encode(b).decode())+'));print("{}")'
 for attempt in range(3):
  try:return ssh(h,code,timeout=15)
  except Exception:
   if attempt==2:raise
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(one,enumerate(parts)))
code='ROOT='+repr(ROOT)+'\nHASH='+repr(hashlib.sha256(raw).hexdigest())+'\n'+r'''
from pathlib import Path
import json,hashlib,zlib
p=Path(ROOT);raw=b''.join(x.read_bytes() for x in sorted((p/'nm-ablation-20261112/delta').glob('part*')))
assert hashlib.sha256(raw).hexdigest()==HASH
for f,text in json.loads(zlib.decompress(raw)).items():
 q=p/f;q.parent.mkdir(exist_ok=True,parents=True);q.write_text(text)
print(json.dumps(dict(passed=True,bytes=len(raw))))
'''
print(ssh(h,code),flush=True)
