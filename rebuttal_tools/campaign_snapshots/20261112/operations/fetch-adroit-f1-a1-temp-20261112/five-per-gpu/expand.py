from pathlib import Path
import json,subprocess,concurrent.futures,collections,copy,ast,time
P=Path(__file__).resolve().parent;BASE=P.parent;R='/workspace/heechan-unified-f1-a1-20261112';HOSTS=[1,3,4,5]
controller=(BASE/'controller.py').read_text();ast.parse(controller)
# The placement change preserves four collector processes and does not alter learning code.
ns={};exec(compile(ast.Module(body=[n for n in ast.parse(controller).body if isinstance(n,ast.FunctionDef) and n.name=='placement'],type_ignores=[]),'<placement>','exec'),ns)
a,b=ns['placement']([[4,68],[5,69],[6,70]],{'physical_cores_per_job':3});assert a==[4,68] and b==[[5],[69],[6],[70]]
a,b=ns['placement']([[i,i+64] for i in range(6)],{});assert len(a)==4 and len(b)==4

def call(h,code,timeout=90):
 r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15',f'ibolt-server-{h}','python3 -'],input=code,capture_output=True,text=True,timeout=timeout)
 if r.returncode:raise RuntimeError(f'host{h}\n'+r.stdout+r.stderr)
 return json.loads(r.stdout)
freeze=r'''
from pathlib import Path
import subprocess,json,time,os,signal,fcntl
p=Path(R);pause=p/'PAUSE';assert not pause.exists();stamp=time.time();pause.write_text('five-per-gpu operational reconfiguration; retain every learner\n')
for _ in range(45):
 plan=json.loads((p/'plan.json').read_text())
 if plan['updated_unix']>stamp:break
 time.sleep(1)
else:raise RuntimeError('Controller admission hold not acknowledged')
name=f'ibolt-f1-a1-temp-20261112-host{HOST}';pid=int(subprocess.check_output(['sudo','-n','supervisorctl','pid',name],text=True).strip());assert str(p/'controller.py').encode() in Path('/proc',str(pid),'cmdline').read_bytes().split(b'\0')
admission=Path('/workspace/.ibolt-capacity-admission.lock').open('a');fcntl.flock(admission,fcntl.LOCK_EX);os.kill(pid,signal.SIGKILL);lock=(p/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
plan=json.loads((p/'plan.json').read_text());folder=p/'five-per-gpu';folder.mkdir(exist_ok=True);assert not (folder/'plan.before.json').exists();(folder/'plan.before.json').write_text(json.dumps(plan,indent=2));(folder/'controller.before.py').write_bytes((p/'controller.py').read_bytes())
print(json.dumps(dict(host=HOST,at=time.time(),controller_pid=pid,plan=plan)))
'''
def stop(h):
 d=call(h,'R='+repr(R)+'\nHOST='+repr(h)+'\n'+freeze);(P/f'before-host{h}.json').write_text(json.dumps(d,indent=2));print('SCHEDULER_HELD',h,flush=True);return h,d
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as e:before=dict(e.map(stop,HOSTS))
plans={h:copy.deepcopy(d['plan']) for h,d in before.items()}
# Server4 needs five additional queued jobs to occupy its twenty safe slots.
moved=[j for j in plans[5]['jobs'] if j['family']=='adroit' and j['phase']=='pilot' and j['status']=='queued'][:5];assert len(moved)==5
ids={j['id'] for j in moved};plans[5]['jobs']=[j for j in plans[5]['jobs'] if j['id'] not in ids]
for j in moved:j['previous_host']=5;j['host']=4
pos=next((i for i,j in enumerate(plans[4]['jobs']) if j['phase']=='confirm'),len(plans[4]['jobs']));plans[4]['jobs'][pos:pos]=moved
for h,plan in plans.items():
 plan['max_jobs_per_gpu']=5;plan['physical_cores_per_job']=3 if h in [1,4] else 6;plan['wait_for_low_gpu_util']=False
 plan['disk_reserve_mib_per_job']=256 if h==5 else 512
 plan['counts']=dict(collections.Counter(j['status'] for j in plan['jobs']))
 plan['five_per_gpu_authorized_unix']=time.time();(P/f'plan-host{h}.json').write_text(json.dumps(plan,indent=2))
alljobs=[j for plan in plans.values() for j in plan['jobs']];assert len(alljobs)==len({j['id'] for j in alljobs});assert len([j for j in alljobs if j['phase']!='smoke'])==320
(P/'transfer.json').write_text(json.dumps(dict(at=time.time(),from_host=5,to_host=4,jobs=moved,reason='Five learners per available GPU; transfer only queued unstarted runs'),indent=2))
apply=r'''
from pathlib import Path
import subprocess,json,time,os,collections,hashlib
p=Path(R);assert (p/'PAUSE').read_text().startswith('five-per-gpu');current=json.loads((p/'plan.json').read_text());old_ids={j['id'] for j in current['jobs']};new_ids={j['id'] for j in PLAN['jobs']}
for j in PLAN['jobs']:
 if j['id'] not in old_ids:assert j['status']=='queued' and not Path(j['output']).exists()
for j in current['jobs']:
 if j['id'] not in new_ids:assert j['status']=='queued' and not Path(j['output']).exists()
def identity(pid):
 try:
  a=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split();return int(a[19]) if a[0]!='Z' else None
 except FileNotFoundError:return None
active=[j for j in PLAN['jobs'] if j['status']=='running'];pgids={j['pid'] for j in active};proc_rows=[]
for q in Path('/proc').iterdir():
 if not q.name.isdigit():continue
 try:
  a=(q/'stat').read_text().rsplit(')',1)[1].split();proc_rows.append(dict(pid=int(q.name),pgid=int(a[2]),uid=q.stat().st_uid,affinity=set(os.sched_getaffinity(int(q.name)))))
 except (ProcessLookupError,FileNotFoundError,PermissionError):pass
report=dict(at=time.time(),host=HOST,retained=[],affinity=[])
for j in active:
 if identity(j['pid'])==j['start_ticks']:report['retained'].append(dict(id=j['id'],pid=j['pid'],start_ticks=j['start_ticks']))
if PLAN['physical_cores_per_job']==3:
 groups={}
 for c in sorted(os.sched_getaffinity(0)):
  q=Path(f'/sys/devices/system/cpu/cpu{c}/topology');k=((q/'physical_package_id').read_text(),(q/'core_id').read_text());groups.setdefault(k,[]).append(c)
 reserved=set(sum(list(groups.values())[:4],[]))
 for q in proc_rows:
  if q['pgid'] not in pgids and len(q['affinity'])<=16:reserved.update(q['affinity'])
 lanes=[v for v in groups.values() if not reserved.intersection(v)];assert len(lanes)>=60,(HOST,len(lanes))
 for j in active:
  if identity(j['pid'])!=j['start_ticks']:continue
  prog=json.loads((Path(j['output'])/'progress.json').read_text());assert prog['steps']>=12800
  selected=lanes[:3];lanes=lanes[3:];learner=selected[0];worker_threads=selected[1]+selected[2];assert len(worker_threads)==4;workers=[[c] for c in worker_threads]
  oldlearner=j['learner_cpus'];oldworkers=j['worker_cpus'];changes=[]
  for q in proc_rows:
   if q['pgid']!=j['pid']:continue
   assert q['uid']==os.getuid()
   ix=next((i for i,a in enumerate(oldworkers) if q['affinity'] and q['affinity'].issubset(set(a))),None)
   target=workers[ix] if ix is not None else learner
   # The initial worker pinning has already happened; all native threads follow their process placement.
   taskdir=Path('/proc',str(q['pid']),'task')
   if not taskdir.exists():continue
   tids=[]
   for t in list(taskdir.iterdir()):
    try:os.sched_setaffinity(int(t.name),target);tids.append(int(t.name))
    except ProcessLookupError:pass
   changes.append(dict(pid=q['pid'],worker_index=ix,old=sorted(q['affinity']),new=target,threads=len(tids)))
  assert sum(c['worker_index'] is not None for c in changes)==4,(j['id'],changes)
  event=dict(at=time.time(),steps=prog['steps'],old_learner=oldlearner,old_workers=oldworkers,new_learner=learner,new_workers=workers,processes=changes)
  j.setdefault('affinity_history',[]).append(event);j['learner_cpus']=learner;j['worker_cpus']=workers;report['affinity'].append(dict(id=j['id'],**event))
PLAN['counts']=dict(collections.Counter(j['status'] for j in PLAN['jobs']));PLAN['updated_unix']=time.time()
(p/'five-per-gpu/affinity-and-preservation.json').write_text(json.dumps(report,indent=2));(p/'five-per-gpu/queue-transfer.json').write_text(json.dumps(TRANSFER,indent=2));tmp=p/'plan.five.tmp';tmp.write_text(json.dumps(PLAN,indent=2));tmp.replace(p/'plan.json');(p/'controller.py').write_text(CONTROLLER)
print(json.dumps(dict(host=HOST,retained=report['retained'],affinity_changed=len(report['affinity']),counts=PLAN['counts'])))
'''
transfer=json.loads((P/'transfer.json').read_text())
def configure(h):
 d=call(h,'R='+repr(R)+'\nHOST='+repr(h)+'\nPLAN='+repr(plans[h])+'\nCONTROLLER='+repr(controller)+'\nTRANSFER='+repr(transfer)+'\n'+apply);(P/f'applied-host{h}.json').write_text(json.dumps(d,indent=2));print('CONFIGURED',h,'retained',len(d['retained']),'affinity',d['affinity_changed'],flush=True);return d
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as e:list(e.map(configure,HOSTS))
start=r'''
from pathlib import Path
import subprocess,json,time
p=Path(R);name=f'ibolt-f1-a1-temp-20261112-host{HOST}';r=subprocess.run(['sudo','-n','supervisorctl','start',name],capture_output=True,text=True);assert r.returncode==0,r.stdout+r.stderr;(p/'PAUSE').unlink();print(json.dumps(dict(host=HOST,started=True)))
'''
def resume(h):
 d=call(h,'R='+repr(R)+'\nHOST='+repr(h)+'\n'+start);print('RESUMED',h,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as e:list(e.map(resume,HOSTS))
print('ALL_FOUR_SCHEDULERS_RESUMED_WITH_EXISTING_LEARNERS',flush=True)
