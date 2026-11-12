from pathlib import Path
import json,subprocess
P=Path(__file__).resolve().parent
code=r'''
from pathlib import Path
import json,time,csv,hashlib
root=Path('/data1/heechan/iBOLT/20261112-f1-a1-temp/hosts');out=[]
ids={f'fetch-slide-r20cm-ibolt-s{s}-t0p1-pilot' for s in [5,6]}|{f'fetch-slide-r20cm-ibolt-n64-m256-s{s}-pilot' for s in [5,6]}
for h in range(1,6):
 b=root/f'host{h}/workspace/heechan-unified-f1-a1-20261112';p=b/'plan.json'
 if not p.exists():continue
 for j in json.loads(p.read_text())['jobs']:
  if j['id'] not in ids:continue
  o=b/'runs'/j['id'];checkpoints=[]
  for f in (o/'checkpoints').glob('*'):
   checkpoints.append(dict(name=f.name,size=f.stat().st_size))
  data={}
  for n in ['config.json','mujoco-config.json','progress.json','dacer_regulator.json','initial-state.json','verification.json']:
   if (o/n).exists():data[n]=json.loads((o/n).read_text())
  data['root_files']=[f.name for f in o.iterdir()];data['checkpoints']=checkpoints
  data['csv']=(o/'learner/progress.csv').read_text();data['evals']=(o/'evaluations.jsonl').read_text()
  data.update(job=j,central_path=str(o))
  out.append(data)
print(json.dumps(dict(at=time.time(),runs=out)))
'''
r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-p','2200','heechan@dil.yonsei.ac.kr','python3 -'],input=code,text=True,capture_output=True,timeout=50);assert not r.returncode,r.stderr
s=json.loads(r.stdout);(P/'evidence.json').write_text(json.dumps(s,indent=2))
for r in s['runs']:
 folder=P/r['job']['id'];folder.mkdir(exist_ok=True)
 for k,v in r.items():
  if k.endswith('.json'):(folder/k).write_text(json.dumps(v,indent=2))
 (folder/'progress.csv').write_text(r['csv']);(folder/'evaluations.jsonl').write_text(r['evals'])
 print(r['job']['id'],r['job']['status'],len(r['csv'].splitlines()),'checkpoints',len(r['checkpoints']),r['checkpoints'][:6],r['root_files'])
