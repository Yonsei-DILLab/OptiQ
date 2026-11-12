"""Move only unstarted, uniquely owned jobs; preserve every live learner PID."""
from pathlib import Path
import sys,json,subprocess,hashlib,collections,copy,time
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P.parent))
from deploy import call,freeze,resume
TAG='server2-redistribution-20261112';R='/workspace/heechan-unified-f1-a1-20261112'
def remote(h,code,**v):
 text='\n'.join(f'{k}={x!r}' for k,x in dict(ROOT=R,TAG=TAG,HOST=h,**v).items())+'\n'+code
 result=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15',f'ibolt-server-{h}','python3 -'],input=text,text=True,capture_output=True,timeout=55)
 assert result.returncode==0,result.stdout+result.stderr
 return json.loads(result.stdout)
def save(name,d):(P/name).write_text(json.dumps(d,indent=2))
if __name__=='__main__':
 assert json.loads((P/'preflight.json').read_text())['passed']
 assert json.loads((P/'runtime-checksum.json').read_text())['passed']
 before={};applied=[];target_applied=False
 try:
  for h in [1,3,4,5]:
   remote(h,"from pathlib import Path\nimport json\n(Path(ROOT)/TAG).mkdir(exist_ok=True)\nprint('{}')")
   before[h]=remote(h,freeze,OLD_CONTROLLER=hashlib.sha256((P.parent/'controller.py').read_bytes()).hexdigest(),OLD_GATEWAY=hashlib.sha256((P.parent/'archive_gateway.py').read_bytes()).hexdigest())
   save(f'plan-before-host{h}.json',before[h]);print('ADMISSION_HELD',h,flush=True)
  target=remote(2,"from pathlib import Path\nimport json\np=Path(ROOT);assert (p/'PAUSE').exists();d=json.loads((p/'plan.json').read_text());assert not d['jobs'];assert json.loads((p/'preflight.json').read_text())['passed'];print(json.dumps(d))")
  after=copy.deepcopy(before);moved=[]
  for h,np,nc in [(1,4,2),(3,4,2),(4,8,4),(5,4,2)]:
   for phase,n,status in [('entropy_pilot',np,'waiting_temperature'),('entropy_confirm',nc,'waiting_entropy')]:
    candidates=[j for j in before[h]['jobs'] if j['phase']==phase and j['status']==status and not j.get('pid')]
    # Spread work over tasks, then across seed/entropy values.
    groups=collections.defaultdict(list)
    for j in candidates:groups[(j['family'],j['task'])].append(j)
    selected=[]
    while len(selected)<n:
     for key in sorted(groups):
      if groups[key] and len(selected)<n:selected.append(groups[key].pop(0))
     assert any(groups.values()) or len(selected)==n
    ids={j['id'] for j in selected};after[h]['jobs']=[j for j in after[h]['jobs'] if j['id'] not in ids]
    for j in selected:
     m=copy.deepcopy(j);m.update(host=2,moved_from_host=h,moved_at=time.time(),movement=TAG);moved.append(m)
  # Move five complete conditional seed groups from the most constrained hosts.
  groups=collections.defaultdict(list)
  for h,d in after.items():
   for j in d['jobs']:
    if j['phase']=='confirm':groups[(j['family'],j['task'],j['seed'])].append((h,j))
  preferred=[('adroit','door',0),('adroit','hammer',0),('adroit','relocate',0),('fetch','slide',1),('fetch','slide',4)]
  eligible=[k for k in preferred+sorted(groups) if k in groups and len(groups[k])==5 and all(j['status']=='waiting_selection' and not j.get('pid') for h,j in groups[k])]
  keys=list(dict.fromkeys(eligible))[:5];assert len(keys)==5
  for key in keys:
   for h,j in groups[key]:
    after[h]['jobs']=[x for x in after[h]['jobs'] if x['id']!=j['id']]
    m=copy.deepcopy(j);m.update(host=2,moved_from_host=h,moved_at=time.time(),movement=TAG);moved.append(m)
  assert len(moved)==55 and len({j['id'] for j in moved})==55
  target['jobs']=moved;target['host']=2;target['max_jobs_per_gpu']=5;target['wait_for_low_gpu_util']=False
  allbefore=[j for d in before.values() for j in d['jobs']];allafter=[j for d in after.values() for j in d['jobs']]+moved
  assert len(allbefore)==len(allafter)==len({j['id'] for j in allafter}) and {j['id'] for j in allbefore}=={j['id'] for j in allafter}
  for h in before:
   assert [j for j in before[h]['jobs'] if j['status']=='running']==[j for j in after[h]['jobs'] if j['status']=='running']
  save('movement.json',dict(at=time.time(),passed=True,only_unstarted=True,running_learners_preserved=True,moved=[dict(id=j['id'],from_host=j['moved_from_host'],to_host=2,phase=j['phase'],family=j['family'],task=j['task']) for j in moved]))
  mutation=r'''
from pathlib import Path
import json,time,fcntl,collections
p=Path(ROOT);assert (p/'PAUSE').exists()
lock=(p/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
old=json.loads((p/'plan.json').read_text());assert old==BEFORE
removed={j['id'] for j in old['jobs']}-{j['id'] for j in AFTER['jobs']}
added={j['id'] for j in AFTER['jobs']}-{j['id'] for j in old['jobs']}
for j in old['jobs']:
 if j['id'] in removed:assert j['status'] in ['waiting_temperature','waiting_entropy','waiting_selection'] and not j.get('pid') and not Path(j['output']).exists() and not (p/'specs'/(j['id']+'.json')).exists()
for j in AFTER['jobs']:
 if j['id'] in added:assert not Path(j['output']).exists() and not (p/'specs'/(j['id']+'.json')).exists()
AFTER['counts']=dict(collections.Counter(j['status'] for j in AFTER['jobs']));AFTER['updated_unix']=time.time()
(p/TAG).mkdir(exist_ok=True);(p/TAG/'movement.json').write_text(json.dumps(MOVEMENT,indent=2));tmp=p/'plan.moving.tmp';tmp.write_text(json.dumps(AFTER,indent=2));tmp.replace(p/'plan.json');print(json.dumps(dict(host=HOST,removed=len(removed),added=len(added))))
'''
  movement=json.loads((P/'movement.json').read_text())
  for h in [1,3,4,5]:
   result=remote(h,mutation,BEFORE=before[h],AFTER=after[h],MOVEMENT=movement);applied.append(h);print(result,flush=True)
  result=remote(2,mutation,BEFORE=remote(2,"from pathlib import Path\nprint((Path(ROOT)/'plan.json').read_text())"),AFTER=target,MOVEMENT=movement);target_applied=True;print(result,flush=True)
  save('committed.json',dict(passed=True,at=time.time(),moved=len(moved),unique_entries=len(allafter)))
 except BaseException:
  if not target_applied:
   for h in applied:remote(h,"from pathlib import Path\nimport json\np=Path(ROOT);(p/'plan.json').write_text(json.dumps(BEFORE,indent=2));print('{}')",BEFORE=before[h])
  raise
 finally:
  for h in before:print(remote(h,resume),flush=True)
