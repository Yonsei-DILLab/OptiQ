from pathlib import Path
import subprocess,json
P=Path(__file__).resolve().parent
remote=r'''
from pathlib import Path
import json,time
root=Path('/data1/heechan/iBOLT/20261112-f1-a1-temp');rows=[]
def read(p):return json.loads(p.read_text()) if p.exists() else None
for host in [1,3,4,5]:
 base=root/f'hosts/host{host}/workspace/heechan-unified-f1-a1-20261112';plan=read(base/'plan.json')
 if not plan:continue
 for job in plan['jobs']:
  if job['family']!='adroit' or job['phase']!='pilot':continue
  run=base/'runs'/job['id'];evaluations=[]
  if (run/'evaluations.jsonl').exists():
   for line in (run/'evaluations.jsonl').read_text().splitlines():
    try:evaluations.append(json.loads(line))
    except json.JSONDecodeError:pass
  rows.append(dict(job=job,config=read(run/'config.json'),ibolt_config=read(run/'ibolt-config.json'),evaluations=evaluations,progress=read(run/'progress.json')))
print(json.dumps(dict(at=time.time(),rows=rows)))
'''
r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','2200','heechan@dil.yonsei.ac.kr','python3 -'],input=remote,text=True,capture_output=True,timeout=60);assert not r.returncode,r.stderr
d=json.loads(r.stdout);assert len(d['rows'])==40;(P/'evidence.json').write_text(json.dumps(d,indent=2));subprocess.run(['python3',str(P/'plot.py')],check=True)
