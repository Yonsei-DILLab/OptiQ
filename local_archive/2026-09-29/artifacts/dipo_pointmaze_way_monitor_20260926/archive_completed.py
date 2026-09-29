"""Read-only, resumable archival of the 36 approved jobs; no remote mutation."""
import concurrent.futures, hashlib, json, subprocess, sys, time
from pathlib import Path
BASE=Path(__file__).resolve().parent
WORK=BASE.parents[1]/'tmp/pointmaze-multiseed-worktree'
sys.path.insert(0,str(WORK))
from maze_benchmarks.deadline_queue import verify
SNAP=Path((BASE/'latest-snapshot.txt').read_text())
OUT=BASE/'archive';OUT.mkdir(exist_ok=True)
PLANS=['POINTMAZE_DIPO_UPSTREAM_U32_PLAN.json','DIPO_UPSTREAM_U32_REMAINDER_PLAN.json','POINTMAZE_BASELINE_FOUR_SEED_PLAN.json','POINTMAZE_MEOW_ALPHA05_PLAN.json']
expected={}
for pn in PLANS:
 p=json.loads((WORK/'maze_benchmarks'/pn).read_text())
 for j in p['jobs']:
  assert j['name'] not in expected
  expected[j['name']]=j
assert len(expected)==36
owners={}; transfers=[]
for file in SNAP.glob('vast-*.json'):
 d=json.loads(file.read_text());host=d['host']
 for campaign,c in d['campaigns'].items():
  q=c['metadata']['queue.json'];commit=q['source_commit']
  for j in q['jobs']:
   name=j['name'];assert name in expected,name
   if j['state']=='transferred':transfers.append(dict(host=host,campaign=campaign,job=j));continue
   assert name not in owners,('duplicate owner',name)
   for k,v in expected[name].items():
    if k!='host':assert j.get(k)==v,(name,k,j.get(k),v)
   owners[name]=dict(host=host,campaign=campaign,job=j,commit=commit,root=c['root'],data=c)
assert set(owners)==set(expected)
for tr in transfers:
 dst=owners[tr['job']['name']]
 assert dst['host']=='vast-heechan-6'
 assert dst['campaign']=='pointmaze-baselines-idle-backfill-20260926'
(OUT/'ownership.json').write_text(json.dumps(dict(snapshot=str(SNAP),owners={n:{k:v for k,v in x.items() if k!='data'} for n,x in owners.items()},transfers=transfers),indent=2))
def collect(item):
 name,x=item;j=x['job'];c=x['data'];local=OUT/'runs'/name;local.mkdir(parents=True,exist_ok=True)
 proof=c['proofs'][name+'-runs']
 meta=OUT/'metadata'/x['host']/x['campaign'];meta.mkdir(parents=True,exist_ok=True)
 for fn,v in c['metadata'].items():(meta/fn).write_text(json.dumps(v,indent=2))
 for fn,v in c['transfers'].items():(meta/fn).write_text(json.dumps(v,indent=2))
 (meta/(name+'-proof.json')).write_text(json.dumps(proof,indent=2))
 prior=OUT/'verified'/(name+'.json')
 already=prior.exists() and json.loads(prior.read_text()).get('sha256')==proof['sha256']
 if not already:
  subprocess.run(['rsync','-az','--exclude','learner/',x['host']+':'+x['root']+'/runs/'+name+'/',str(local)+'/'],check=True,capture_output=True)
 checked=verify(local,j,x['commit'],False)
 assert checked['sha256']==proof['sha256'],('remote SHA mismatch',name)
 assert checked['record']==proof['record']
 record=checked['record'];mode=record.get('primary_evaluation_mode','policy');primary=record[mode]
 result=dict(name=name,host=x['host'],campaign=x['campaign'],source_commit=x['commit'],task=j['task'],method=j['method'],seed=j.get('seed',0),temperature=j['temperature'],verified_at=time.time(),steps=checked['steps'],updates=checked['updates'],mode=mode,primary=primary,record=record,sha256=checked['sha256'])
 (OUT/'verified').mkdir(exist_ok=True)
 (OUT/'verified'/(name+'.json')).write_text(json.dumps(result,indent=2))
 print(json.dumps(dict(verified=name,success=primary['success'],goals=primary['goals'])),flush=True)
 return result
jobs=[(n,x) for n,x in owners.items() if x['job']['state']=='complete']
results={};failures={}
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 fs={pool.submit(collect,item):item[0] for item in jobs}
 for f in concurrent.futures.as_completed(fs):
  name=fs[f]
  try:results[name]=f.result()
  except Exception as e:failures[name]=repr(e);print('ARCHIVE_ERROR '+name+' '+repr(e),flush=True)
manifest=dict(checked_at=time.time(),snapshot=str(SNAP),expected_count=36,completed_at_snapshot=len(jobs),verified_count=len(results),results=results,failures=failures,incomplete=[n for n in expected if n not in results])
(OUT/'archive-manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({k:v for k,v in manifest.items() if k!='results'}),flush=True)
if failures:sys.exit(1)
