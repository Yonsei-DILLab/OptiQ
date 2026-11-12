from pathlib import Path
import json,subprocess,concurrent.futures,collections,time
P=Path(__file__).resolve().parent
code=r'''
from pathlib import Path
import json,time,shutil
p=Path('/workspace/heechan-unified-f1-a1-20261112');d=json.loads((p/'plan.json').read_text());rs=[]
for j in d['jobs']:
 if j['status']!='running':continue
 try:
  a=Path('/proc',str(j['pid']),'stat').read_text().rsplit(')',1)[1].split();live=a[0]!='Z' and int(a[19])==j['start_ticks']
 except FileNotFoundError:live=False
 o=Path(j['output']);progress=json.loads((o/'progress.json').read_text()) if (o/'progress.json').exists() else None
 done_path=o/('completed.json' if j['family']=='adroit' else 'verified-completion.json')
 done=json.loads(done_path.read_text()) if not live and done_path.exists() else {}
 expected=(13568 if j['family']=='adroit' else 13312) if j['phase']=='smoke' else 5000192
 finished_normally=bool(done.get('completed') and done.get('steps')==expected and done.get('updates')==int((expected-12800)*(.125 if j['family']=='fetch' else .25)))
 rs.append(dict(id=j['id'],family=j['family'],task=j['task'],method=j['method'],phase=j['phase'],seed=j['seed'],temperature=j['temperature'],live=live,finished_normally=finished_normally,gpu=j['gpu'],progress=progress))
print(json.dumps(dict(at=time.time(),plan=d,running=rs,free_gib=shutil.disk_usage(p).free/2**30,controller_failure=(p/'controller-failure.json').exists())))
'''
def one(h):
 r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','KexAlgorithms=curve25519-sha256','-o','ConnectTimeout=15',f'ibolt-server-{h}','python3 -'],input=code,text=True,capture_output=True,timeout=35);assert not r.returncode,r.stderr;d=json.loads(r.stdout);d['host']=h;(P/f'expanded-host{h}.json').write_text(json.dumps(d,indent=2));return d
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as e:hosts=list(e.map(one,[1,2,3,4,5]))
rows=[j for h in hosts for j in h['plan']['jobs']];assert len(rows)==len({j['id'] for j in rows})
scientific=[j for j in rows if j['phase']!='smoke']
old=json.loads((P/'protocol.json').read_text());expected={j['id'] for j in old['rows'] if j['phase']!='smoke'}
entropy_protocol=P.parent/'fetch-adroit-f1-a1-entropy-20261112/protocol.json'
if any(j['phase'].startswith('entropy_') for j in scientific):
 assert entropy_protocol.exists()
 expected.update(j['id'] for j in json.loads(entropy_protocol.read_text())['rows'])
nm_protocol=P.parent/'fetch-adroit-f1-a1-nm-20261112/protocol.json'
if any(j['phase']=='nm_pilot' for j in scientific):
 assert nm_protocol.exists()
 expected.update(j['id'] for j in json.loads(nm_protocol.read_text())['rows'])
assert len(scientific)==len(expected) and {j['id'] for j in scientific}==expected
summary=[]
for h in hosts:
 assert not h['controller_failure'];assert all(r['live'] or r['finished_normally'] for r in h['running']);assert all(j['status']!='failed' for j in h['plan']['jobs'])
 groups=collections.defaultdict(list)
 for r in h['running']:
  if r['phase']!='smoke':groups[(r['family'],r['method'])].append(r)
 summary.append(dict(host=h['host'],free_gib=h['free_gib'],counts=h['plan']['counts'],running=[dict(family=k[0],method=k[1],n=len(rr),steps=[min((r['progress']or{}).get('steps',0) for r in rr),max((r['progress']or{}).get('steps',0) for r in rr)],tasks=dict(collections.Counter(r['task'] for r in rr))) for k,rr in groups.items()]))
result=dict(at=time.time(),unique_scientific_plan_entries=len(scientific),all_original_jobs_preserved=True,duplicate_job_ids=False,hosts=summary);(P/'host1-expansion-verified.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
