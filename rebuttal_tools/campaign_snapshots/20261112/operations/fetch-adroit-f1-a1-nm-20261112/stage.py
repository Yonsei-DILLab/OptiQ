from pathlib import Path
import json,subprocess,hashlib,tarfile,shlex,time,sys
from preflight import ssh,ROOT
P=Path(__file__).resolve().parent;TAG='nm-ablation-20261112'
protocol=json.loads((P/'protocol.json').read_text())
for h in ([int(sys.argv[1])] if len(sys.argv)>1 else [1,2,3]):
 archive=P/f'stage-host{h}.tgz'
 with tarfile.open(archive,'w:gz') as t:
  t.add(P/'protocol.json',arcname='nm-protocol.json')
  t.add(P/f'additions-host{h}.json',arcname=TAG+'/additions.json')
  t.add(P/f'controller-host{h}.py',arcname=TAG+'/controller.py')
  for variant in protocol['sources']:
   t.add(P/('nm-'+variant+'-source'),arcname='nm-'+variant+'-source')
 cmd=['ssh','-T','-o','BatchMode=yes','-o','KexAlgorithms=curve25519-sha256',f'ibolt-server-{h}','tar -xz -C '+ROOT]
 if '--verify-only' not in sys.argv:
  with archive.open('rb') as f:r=subprocess.run(cmd,stdin=f,capture_output=True,timeout=55)
  assert r.returncode==0,r.stderr
 code='ROOT='+repr(ROOT)+'\n'+r'''
from pathlib import Path
import json,hashlib,subprocess
p=Path(ROOT);proto=json.loads((p/'nm-protocol.json').read_text());proof={}
for variant,record in proto['sources'].items():
 s=p/('nm-'+variant+'-source');m=s/'source-sha256.json'
 assert hashlib.sha256(m.read_bytes()).hexdigest()==record['manifest_sha256']
 for f,h in json.loads(m.read_text()).items():assert hashlib.sha256((s/f).read_bytes()).hexdigest()==h,f
 if not (s/'.git').exists():
  subprocess.run(['git','init','-q','-b','heechan-rebuttal',str(s)],check=True,stdout=subprocess.DEVNULL)
  subprocess.run(['git','add','.'],cwd=s,check=True)
  subprocess.run(['git','-c','user.name=Experiment Snapshot','-c','user.email=experiment@local','commit','-q','-m','Freeze authorized N and M ablation with DACER baseline'],cwd=s,check=True)
 assert subprocess.check_output(['git','branch','--show-current'],cwd=s,text=True).strip()=='heechan-rebuttal'
 proof[s.name]=record['manifest_sha256']
for j in json.loads((p/'nm-ablation-20261112/additions.json').read_text()):
 if j['family']=='adroit':
  assert hashlib.sha256(Path(j['demo_path']).read_bytes()).hexdigest()==j['demo_sha256']
  assert hashlib.sha256(Path(j['demonstration_manifest']).read_bytes()).hexdigest()==j['demonstration_manifest_sha256']
d=dict(passed=True,source_manifests=proof,protocol_sha256=hashlib.sha256((p/'nm-protocol.json').read_bytes()).hexdigest())
(p/'nm-transfer-proof.json').write_text(json.dumps(d,indent=2));print(json.dumps(d))
'''
 d=ssh(h,code);(P/f'staged-host{h}.json').write_text(json.dumps(d,indent=2));print('STAGED',h,flush=True)
