"""User-authorized early T freeze and occupancy-balanced entropy admission."""
from pathlib import Path
import sys,json,subprocess,tarfile,hashlib,copy,collections,time,concurrent.futures
P=Path(__file__).resolve().parent;BASE=P.parent;sys.path.insert(0,str(BASE));from deploy import freeze,resume
R='/workspace/heechan-unified-f1-a1-20261112';TAG='capacity-current-step-20261112';HOSTS=[1,2,3,4,5]
def remote(h,code,**values):
 text='\n'.join(f'{k}={v!r}' for k,v in dict(ROOT=R,TAG=TAG,HOST=h,**values).items())+'\n'+code
 r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15',f'ibolt-server-{h}','python3 -'],input=text,text=True,capture_output=True,timeout=60)
 assert r.returncode==0,r.stdout+r.stderr
 return json.loads(r.stdout)
def save(name,x):(P/name).write_text(json.dumps(x,indent=2))
def stage(h):
 remote(h,"from pathlib import Path\n(Path(ROOT)/TAG).mkdir(exist_ok=True)\nprint('{}')")
 with (P/'stage.tgz').open('rb') as f:r=subprocess.run(['ssh','-T',f'ibolt-server-{h}','tar -xz -C '+R+'/'+TAG],stdin=f,capture_output=True,timeout=30)
 assert r.returncode==0,r.stderr
 return h,remote(h,"from pathlib import Path\nimport hashlib,json\np=Path(ROOT);print(json.dumps(dict(gateway=hashlib.sha256((p/'archive_gateway.py').read_bytes()).hexdigest())))")
if __name__=='__main__':
 with tarfile.open(P/'stage.tgz','w:gz') as t:
  t.add(P/'controller.balanced.py',arcname='controller.py')
  for n in ['entropy_dependencies.py','archive_gateway.py']:t.add(BASE/n,arcname=n)
  t.add(P/'protocol-amendment.json',arcname='entropy-protocol-amendment.json')
  t.add(P/'entropy-temperature-selections',arcname='entropy-temperature-selections')
 with concurrent.futures.ThreadPoolExecutor(max_workers=5) as e:staged=dict(e.map(stage,HOSTS))
 before={};applied=[]
 try:
  def hold(h):
   plan=remote(h,freeze,OLD_CONTROLLER=hashlib.sha256((BASE/'controller.py').read_bytes()).hexdigest(),OLD_GATEWAY=staged[h]['gateway']);save(f'plan-before-host{h}.json',plan);return h,plan
  with concurrent.futures.ThreadPoolExecutor(max_workers=5) as e:
   futures={e.submit(hold,h):h for h in HOSTS}
   for f in concurrent.futures.as_completed(futures):
    h,plan=f.result();before[h]=plan;print('ADMISSION_HELD',h,flush=True)
  after=copy.deepcopy(before);pool=[];assignment={h:[] for h in HOSTS}
  for h in HOSTS:
   pending=[j for j in after[h]['jobs'] if j['phase']=='entropy_pilot' and j['status'] in ['waiting_temperature','queued'] and not j.get('pid')]
   ids={j['id'] for j in pending};after[h]['jobs']=[j for j in after[h]['jobs'] if j['id'] not in ids]
   pool += [(h,j) for j in pending]
  assert len(pool)==56
  active={h:sum(j['status']=='running' for j in after[h]['jobs']) for h in HOSTS};limits={h:(15 if h==5 else 20) for h in HOSTS};room={h:max(0,limits[h]-active[h]) for h in HOSTS}
  for h,family in [(1,'fetch'),(5,'adroit')]:
   selected=[x for x in pool if x[1]['family']==family][:room[h]];assignment[h]+=selected;ids={x[1]['id'] for x in selected};pool=[x for x in pool if x[1]['id'] not in ids]
  for oldhost,j in pool:
   eligible=[h for h in [2,3,4] if len(assignment[h])<room[h]]
   h=min(eligible or [2,3,4],key=lambda x:((active[x]+len(assignment[x]))/4,len(assignment[x]),x));assignment[h].append((oldhost,j))
  records={q.stem:json.loads(q.read_text()) for q in (P/'entropy-temperature-selections').glob('*.json')};moves=[]
  for h in HOSTS:
   for oldhost,j in assignment[h]:
    m=copy.deepcopy(j);t=records[j['family']+'-'+j['task']];m.update(host=h,status='queued',temperature=t['temperature'],temperature_selection_sha256=hashlib.sha256(json.dumps(t,sort_keys=True).encode()).hexdigest())
    if h!=oldhost:m.setdefault('placement_history',[]).append(dict(from_host=oldhost,to_host=h,at=time.time(),reason=TAG))
    after[h]['jobs'].append(m);moves.append(dict(id=j['id'],from_host=oldhost,to_host=h,temperature=t['temperature']))
   # New entropy pilots first; active and completed jobs remain untouched.
   after[h]['jobs'].sort(key=lambda j:0 if j['phase']=='entropy_pilot' and j['status']=='queued' else 1)
   after[h]['wait_for_low_gpu_util']=False
   if h==5:after[h].update(min_free_gib=4,disk_reserve_mib_per_job=128)
   assert [j for j in before[h]['jobs'] if j['status']=='running']==[j for j in after[h]['jobs'] if j['status']=='running']
  orig=[j['id'] for d in before.values() for j in d['jobs']];new=[j['id'] for d in after.values() for j in d['jobs']];assert len(new)==len(set(new)) and set(orig)==set(new)
  movement=dict(at=time.time(),reason=TAG,only_unstarted=True,active_before=active,assigned_pilots={h:len(v) for h,v in assignment.items()},moves=moves);save('movement.json',movement)
  mutation=r'''
from pathlib import Path
import json,fcntl,shutil,time,collections
p=Path(ROOT);stage=p/TAG;assert (p/'PAUSE').read_text().startswith(TAG)
lock=(p/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
old=json.loads((p/'plan.json').read_text());assert old==BEFORE
for j in AFTER['jobs']:
 if j['phase']=='entropy_pilot' and j['status']=='queued':assert not Path(j['output']).exists() and not (p/'specs'/(j['id']+'.json')).exists()
for n in ['controller.py','entropy_dependencies.py','archive_gateway.py']:
 shutil.copy2(p/n,stage/(n+'.before'));shutil.copy2(stage/n,p/n)
shutil.copytree(stage/'entropy-temperature-selections',p/'entropy-temperature-selections',dirs_exist_ok=True)
shutil.copy2(stage/'entropy-protocol-amendment.json',p/'entropy-protocol-amendment.json')
AFTER['counts']=dict(collections.Counter(j['status'] for j in AFTER['jobs']));AFTER['updated_unix']=time.time()
tmp=p/'plan.capacity.tmp';tmp.write_text(json.dumps(AFTER,indent=2));tmp.replace(p/'plan.json');(stage/'movement.json').write_text(json.dumps(MOVEMENT,indent=2))
print(json.dumps(dict(host=HOST,counts=AFTER['counts'],new_entropy_pilots=sum(j['status']=='queued' and j['phase']=='entropy_pilot' for j in AFTER['jobs']))))
'''
  for h in HOSTS:
   result=remote(h,mutation,BEFORE=before[h],AFTER=after[h],MOVEMENT=movement);applied.append(h);print(result,flush=True)
  save('committed.json',dict(passed=True,at=time.time(),hosts=HOSTS,unique_jobs=len(new),released=56,learner_pids_preserved=True))
 except BaseException:
  for h in applied:
   remote(h,"from pathlib import Path\nimport shutil,json\np=Path(ROOT);s=p/TAG\nfor n in ['controller.py','entropy_dependencies.py','archive_gateway.py']:shutil.copy2(s/(n+'.before'),p/n)\n(p/'plan.json').write_text(json.dumps(BEFORE,indent=2));print('{}')",BEFORE=before[h])
  raise
 finally:
  with concurrent.futures.ThreadPoolExecutor(max_workers=5) as e:
   for result in e.map(lambda h:remote(h,resume),sorted(before)):print(result,flush=True)
