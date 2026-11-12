from pathlib import Path
import subprocess,json,shlex
P=Path(__file__).resolve().parent
for h in [1,2,3]:
 code='HOST='+str(h)+'\n'+r'''
import sys,tarfile,io,json,hashlib,time
from pathlib import Path
root=Path('/data1/heechan/iBOLT/20261112-f1-a1-temp');base=root/f'hosts/host{HOST}/workspace/heechan-unified-f1-a1-20261112'
base.mkdir(parents=True,exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as t:
 for member in t.getmembers():
  assert not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
 t.extractall(base)
proto=json.loads((base/'nm-protocol.json').read_text());proof={}
for variant,s in proto['sources'].items():
 p=base/('nm-'+variant+'-source');m=p/'source-sha256.json'
 assert hashlib.sha256(m.read_bytes()).hexdigest()==s['manifest_sha256']
 for f,digest in json.loads(m.read_text()).items():assert hashlib.sha256((p/f).read_bytes()).hexdigest()==digest,f
 proof[p.name]=s['manifest_sha256']
record=dict(passed=True,host=HOST,at=time.time(),central=str(base),source_manifests=proof,protocol_sha256=hashlib.sha256((base/'nm-protocol.json').read_bytes()).hexdigest())
admin=root/'administration/nm-ablation-20261112';admin.mkdir(exist_ok=True,parents=True)
(admin/f'host{HOST}-source-receipt.json').write_text(json.dumps(record,indent=2))
(admin/'protocol.json').write_text(json.dumps(proto,indent=2))
print(json.dumps(record))
'''
 with (P/f'stage-host{h}.tgz').open('rb') as f:
  r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-p','2200','heechan@dil.yonsei.ac.kr','python3 -c '+shlex.quote(code)],stdin=f,capture_output=True,timeout=55)
 assert r.returncode==0,r.stderr.decode();d=json.loads(r.stdout);(P/f'central-source-receipt-host{h}.json').write_text(json.dumps(d,indent=2));print('CENTRAL_SOURCE_VERIFIED',h,flush=True)
