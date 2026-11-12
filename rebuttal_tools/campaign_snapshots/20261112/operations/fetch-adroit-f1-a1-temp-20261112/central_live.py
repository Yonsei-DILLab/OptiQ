from central_collector import *
from statistics import mean,stdev
TEMPS=[.05,.1,.25,.5,1.]

def sync_live(n):
 try:
  base=ROOT/f'hosts/host{n}/workspace'/NEW
  # Metadata and immutable protocol files go first, independent of checkpoint volume.
  base.mkdir(parents=True,exist_ok=True)
  cmd=['rsync','-a','--partial','--timeout=90','--exclude=checkpoints/','--exclude=.git','--exclude=__pycache__','--exclude=*.tmp','--exclude=*.lock','--exclude=jax-cache/','--exclude=mpl-cache/','--exclude=*.tgz','--exclude=datasets/','--exclude=fetch-source/','--exclude=fetch-slide-20cm-source/','--exclude=adroit-source/','--exclude=runtime/','-e',shlex.join(ssh(n)),'heechan@'+HOSTS[n][0]+':'+NEW+'/',str(base)+'/']
  r=subprocess.run(cmd,capture_output=True,text=True,timeout=600)
  if r.returncode not in (0,24):raise RuntimeError(r.stderr[-1000:])
  proof=json.loads((base/'transfer-proof.json').read_text())
  assert all(digest(base/f)==h for f,h in proof.items())
  ready=dict(passed=True,hashes=proof,central=str(base),verified_unix=time.time())
  if not (base/'archive-ready.json').exists() or json.loads((base/'archive-ready.json').read_text()).get('hashes') != proof:remote(n,'archive-ready',ready)
  out=dict(host=n,passed=True,at=time.time())
 except Exception as e:out=dict(host=n,passed=False,error=repr(e),at=time.time())
 (ROOT/f'live-host{n}.json').write_text(json.dumps(out));return out

def read_json(p):
 try:return json.loads(p.read_text())
 except FileNotFoundError:return None

def collect():
 rows=[]
 for h in HOSTS:
  base=ROOT/f'hosts/host{h}/workspace'/NEW;plan=read_json(base/'plan.json')
  if not plan:continue
  for j in plan['jobs']:
   out=base/('preflights' if j['phase']=='smoke' else 'runs')/j['id']
   done=read_json(out/('completed.json' if j['family']=='adroit' else 'verified-completion.json'))
   rows.append(dict(job=j,output=out,done=done))
 return rows

def coordinate():
 rows=collect();gates={}
 for r in rows:
  j=r['job'];d=r['done']
  if j['phase']=='smoke' and d and d.get('completed'):
   assert d['steps']==13312 and d['updates']==64 and d['actor_parameters_changed'] and d['backend']=='gpu'
   if j['task']=='slide':
    config=read_json(r['output']/'config.json')
    assert j.get('protocol_variant')=='F1-slide-20cm' and config.get('distance_threshold_m')==.20
   gates[j['task']+'-'+j['method']]=True
 if gates:
  for n in HOSTS:remote(n,'smoke-gates',gates)
 selections=ROOT/'selections';selections.mkdir(exist_ok=True)
 for family,tasks in dict(fetch=['reach','push','pick-and-place','slide'],adroit=['door','hammer','pen','relocate']).items():
  for task in tasks:
   dest=selections/(family+'-'+task+'.json')
   if dest.exists():
    selection=read_json(dest)
    for n in HOSTS:remote(n,'selection',selection)
    continue
   candidates=[r for r in rows if r['job']['family']==family and r['job']['task']==task and r['job']['phase']=='pilot']
   if len(candidates)!=10 or any(not r['done'] or not r['done'].get('completed') or r['done']['steps']!=5000192 for r in candidates):continue
   if family=='fetch' and task=='slide':
    assert all(r['job'].get('protocol_variant')=='F1-slide-20cm' and r['job'].get('distance_threshold_m')==.20 for r in candidates)
    assert all(read_json(r['output']/'config.json').get('distance_threshold_m')==.20 for r in candidates)
   curves={}
   for r in candidates:
    j=r['job'];curve={}
    for line in (r['output']/'evaluations.jsonl').read_text().splitlines():
     e=json.loads(line);mode=e['primary_mode'];eps=e[mode]['episodes'];n=len(eps) if isinstance(eps,list) else eps
     if n==10 and e['steps']>0:curve[e['steps']]=(e['updates'],e[mode]['success_rate'])
    curves[j['temperature'],j['seed']]=curve
   common=sorted(set.intersection(*(set(c) for c in curves.values())))
   assert len(common)>=5 and common[-1]>4_900_000
   last=common[-5:];scores=[]
   for temp in TEMPS:
    vals=[mean(curves[temp,s][t][1] for t in last) for s in [5,6]]
    scores.append(dict(temperature=temp,mean=mean(vals),per_seed=vals))
   best=max(scores,key=lambda r:(r['mean'],-r['temperature']))
   selection=dict(family=family,task=task,temperature=best['temperature'],pilot_seeds=[5,6],expected_steps=5000192,common_steps=last,scores=scores,rule='mean of last five regular primary success evaluations across two pilot seeds; ties ascending temperature',at=time.time())
   if family=='fetch' and task=='slide':selection.update(protocol_variant='F1-slide-20cm',distance_threshold_m=.20)
   dest.write_text(json.dumps(selection,indent=2))
   for n in HOSTS:remote(n,'selection',selection)
   print('SELECTED',json.dumps(selection),flush=True)
 summary=dict(at=time.time(),smoke_gates=gates,counts={s:sum(r['job']['status']==s for r in rows) for s in set(r['job']['status'] for r in rows)},selected=[p.stem for p in selections.glob('*.json')])
 (ROOT/'live-summary.json').write_text(json.dumps(summary,indent=2))
 print(json.dumps(summary),flush=True)

def host_live_loop(n):
 while True:
  sync_live(n)
  time.sleep(45)

import threading
for n in HOSTS:threading.Thread(target=host_live_loop,args=(n,),daemon=True).start()
while True:
 try:coordinate()
 except Exception as e:(ROOT/'selector-error.json').write_text(json.dumps(dict(at=time.time(),error=repr(e),traceback=traceback.format_exc())))
 time.sleep(45)
