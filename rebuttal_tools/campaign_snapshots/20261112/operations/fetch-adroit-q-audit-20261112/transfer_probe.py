from pathlib import Path
import json,subprocess,time
P=Path(__file__).resolve().parent
e=json.loads((P/'evidence.json').read_text());jobs=[r['job'] for r in e['runs'] if r['job']['family']=='adroit']
code='JOBS='+repr(jobs)+'\n'+r'''
from pathlib import Path
import json,tarfile,sys,io,hashlib
root=Path('/data1/heechan/iBOLT/20261112-f1-a1-temp/hosts');manifest={};rows=[]
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:
 for j in JOBS:
  p=root/f"host{j['host']}/workspace/heechan-unified-f1-a1-20261112/runs"/j['id'];steps=[]
  files=[p/'ibolt-config.json']
  for step in [1000960,2001920,5000192]:
   f=p/'checkpoints'/f'critic_state_{step}.msgpack'
   if f.exists():files.append(f);steps.append(step)
  for f in files:
   name=j['id']+'/'+str(f.relative_to(p));manifest[name]=hashlib.sha256(f.read_bytes()).hexdigest();t.add(f,arcname=name)
  rows.append(dict(id=j['id'],task=j['task'],seed=j['seed'],N=j.get('num_policy_samples',64),M=j.get('num_reference_actions',64),steps=steps,demo_path=j['demo_path'],demo_sha256=j['demo_sha256']))
 for name,obj in [('checkpoint-sha256.json',manifest),('probe-spec.json',dict(rows=rows))]:
  data=json.dumps(obj,indent=2).encode();info=tarfile.TarInfo(name);info.size=len(data);t.addfile(info,io.BytesIO(data))
'''
with (P/'probe-checkpoints.tgz').open('wb') as out:
 r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-p','2200','heechan@dil.yonsei.ac.kr','python3 -'],input=code.encode(),stdout=out,stderr=subprocess.PIPE,timeout=90)
 assert r.returncode==0,r.stderr
remote='/workspace/heechan-unified-f1-a1-20261112/diagnostics/adroit-q-audit-20261112'
r=subprocess.run(['ssh','-T','-o','BatchMode=yes','ibolt-server-3','mkdir -p '+remote],capture_output=True,timeout=25);assert not r.returncode,r.stderr
with (P/'probe-checkpoints.tgz').open('rb') as data:
 r=subprocess.run(['ssh','-T','-o','BatchMode=yes','ibolt-server-3','tar -xz -C '+remote],stdin=data,capture_output=True,timeout=90);assert not r.returncode,r.stderr
print('Transferred MiB',round((P/'probe-checkpoints.tgz').stat().st_size/2**20,1),flush=True)
