"""Read-only operational health checks for the explicitly approved sixteen runs."""
from pathlib import Path
import concurrent.futures,json,math,subprocess,sys,time
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'fetch-adroit-f1-a1-nm-20261112'))
from preflight import ssh
proto=json.loads((P/'protocol.json').read_text());ids={r['id'] for r in proto['rows']}
status=json.loads((P/'status.json').read_text())
previous=json.loads((P/'heartbeat-previous-status.json').read_text())
old={r['id']:r for r in previous['runs']};issues=[]
assert ids=={r['id'] for r in status['runs']} and len(ids)==16
for r in status['runs']:
    if not r['verified'] or r['failure']:issues.append([r['id'],'configuration or failure',r['failure']])
    if r['status']!='completed' and not r['live']:issues.append([r['id'],'not live'])
    if r['status']!='completed' and r['steps']<=old[r['id']]['steps']:issues.append([r['id'],'no progress since previous check'])
    if any(not math.isfinite(v) for v in r['metrics'].values() if isinstance(v,(float,int))):issues.append([r['id'],'nonfinite current metric'])
CODE='IDS='+repr(ids)+'\n'+r'''
from pathlib import Path
import json,time,shutil,csv,math,hashlib
p=Path('/workspace/heechan-unified-f1-a1-20261112');d=json.loads((p/'plan.json').read_text());out=[]
for j in d['jobs']:
 if j['id'] not in IDS:continue
 o=Path(j['output']);pr=o/'progress.json';f=o/'learner/progress.csv';bad=[];count=0;hashes={}
 if f.exists():
  with f.open() as z:
   for row in csv.DictReader(z):
    count+=1
    for k,v in row.items():
     if not v:continue
     try:
      if not math.isfinite(float(v)):bad.append([count,k,v])
     except ValueError:pass
 if j['status']=='completed':
  for f in (o/'checkpoints').iterdir():
   if f.is_file():hashes[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
 out.append(dict(id=j['id'],status=j['status'],progress_age_s=time.time()-pr.stat().st_mtime if pr.exists() else None,csv_rows=count,nonfinite=bad[:10],checkpoint_hashes=hashes))
print(json.dumps(dict(host=d['host'],free_gib=shutil.disk_usage(p).free/2**30,disk_gate=d.get('min_free_gib'),runs=out,controller_failure=json.loads((p/'controller-failure.json').read_text()) if (p/'controller-failure.json').exists() else None)))
'''
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:hosts=list(pool.map(lambda h:ssh(h,CODE),[3,4]))
for h in hosts:
    if h['free_gib']<=h['disk_gate']:issues.append([h['host'],'disk below gate',h['free_gib']])
    if h['controller_failure']:issues.append([h['host'],'controller failure',h['controller_failure']])
    for r in h['runs']:
        if r['nonfinite']:issues.append([r['id'],'nonfinite logged metric',r['nonfinite']])
        if r['status']!='completed' and (r['progress_age_s'] is None or r['progress_age_s']>300):issues.append([r['id'],'stale progress',r['progress_age_s']])
completed=[dict(host=h['host'],id=r['id'],hashes=r['checkpoint_hashes']) for h in hosts for r in h['runs'] if r['status']=='completed']
code='RUNS='+repr(completed)+'\n'+r'''
from pathlib import Path
import hashlib,json
root=Path('/data1/heechan/iBOLT/20261112-f1-a1-temp');rows=[]
for r in RUNS:
 p=root/f"hosts/host{r['host']}/workspace/heechan-unified-f1-a1-20261112/runs"/r['id'];checks={}
 for n,h in r['hashes'].items():
  f=p/'checkpoints'/n;checks[n]=f.exists() and hashlib.sha256(f.read_bytes()).hexdigest()==h
 rows.append(dict(id=r['id'],checks=checks,passed=bool(checks) and all(checks.values())))
print(json.dumps(rows))
'''
r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-p','2200','heechan@dil.yonsei.ac.kr','python3 -'],input=code.encode(),capture_output=True,timeout=50)
assert not r.returncode,r.stderr
receipt=json.loads(r.stdout)
(P/'completed-checkpoint-central-verification.json').write_text(json.dumps(dict(at=time.time(),rows=receipt),indent=2)+'\n')
for r in receipt:
    if not r['passed']:issues.append([r['id'],'completed checkpoint archive pending or mismatch',r['checks']])
snap=json.loads((P/'snapshot.json').read_text())
sync=[dict(host=h['host'],passed=h['sync']['passed'],age_s=snap['at']-h['sync']['at']) for h in snap['hosts'] if h['host'] in [3,4]]
for h in sync:
    if not h['passed'] or h['age_s']>300:issues.append([h['host'],'central sync stale or failed',h])
report=dict(at=time.time(),counts=status['counts'],issues=issues,hosts=hosts,central_sync=sync,completed_checkpoint_hashes_verified=sum(r['passed'] for r in receipt),note='Adroit historical progress CSV is empty; current metrics and completion metadata are checked instead.')
(P/'heartbeat-health.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='hosts'},indent=2))
print('HOSTS',[(h['host'],round(h['free_gib'],2),len(h['runs'])) for h in hosts])
print('PROGRESS',[(r['id'],r['status'],r['steps']) for r in status['runs']])
