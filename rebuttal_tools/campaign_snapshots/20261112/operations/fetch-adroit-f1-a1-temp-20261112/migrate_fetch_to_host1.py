from pathlib import Path
import subprocess,json,time
P=Path(__file__).resolve().parent;R='/workspace/heechan-unified-f1-a1-20261112'
def call(host,code,timeout=90):
 r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15',host,'python3 -'],input=code,capture_output=True,text=True,timeout=timeout)
 if r.returncode:raise RuntimeError(r.stdout+r.stderr)
 return json.loads(r.stdout)
# Host1 has only its completed integration checks; stop its empty controller normally.
pre=call('ibolt-server-1','R='+repr(R)+'\n'+r'''
from pathlib import Path
import subprocess,json,time
p=Path(R)
for _ in range(25):
 plan=json.loads((p/'plan.json').read_text())
 if all(j['status']=='completed' for j in plan['jobs']):break
 time.sleep(1)
assert len(plan['jobs'])==8 and all(j['phase']=='smoke' and j['status']=='completed' for j in plan['jobs'])
for j in plan['jobs']:
 d=json.loads((Path(j['output'])/'verified-completion.json').read_text());assert d['completed'] and d['backend']=='gpu' and d['actor_parameters_changed'] and d['steps']==13312 and d['updates']==64
r=subprocess.run(['sudo','-n','supervisorctl','stop','ibolt-f1-a1-temp-20261112-host1'],capture_output=True,text=True);assert r.returncode==0,r.stdout+r.stderr
print(json.dumps(dict(passed=True,smokes=8)))
''')
# Replace only the scheduler process, retaining every active training PID and checkpoint state.
receipt=call('ibolt-server-4','R='+repr(R)+'\n'+r'''
from pathlib import Path
import subprocess,json,time,os,signal,fcntl,collections
p=Path(R);pause=p/'PAUSE';assert not pause.exists();stamp=time.time();pause.write_text('Temporary scheduler-only transfer to server1; running learners remain active.\n')
for _ in range(40):
 plan=json.loads((p/'plan.json').read_text())
 if plan['updated_unix']>stamp:break
 time.sleep(1)
else:raise RuntimeError('Controller did not acknowledge admissions hold')
name='ibolt-f1-a1-temp-20261112-host4';pid=int(subprocess.check_output(['sudo','-n','supervisorctl','pid',name],text=True).strip());argv=Path('/proc',str(pid),'cmdline').read_bytes().split(b'\0');assert str(p/'controller.py').encode() in argv
before=[dict(id=j['id'],pid=j['pid'],start_ticks=j['start_ticks']) for j in plan['jobs'] if j['status']=='running']
def ident(pid):
 try:
  a=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split();return int(a[19]) if a[0]!='Z' else None
 except FileNotFoundError:return None
assert all(ident(j['pid'])==j['start_ticks'] for j in before)
admission=Path('/workspace/.ibolt-capacity-admission.lock').open('a');fcntl.flock(admission,fcntl.LOCK_EX)
os.kill(pid,signal.SIGKILL)
lock=(p/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
plan=json.loads((p/'plan.json').read_text());backup=p/f'plan.before-host1-{int(time.time())}.json';backup.write_text(json.dumps(plan,indent=2))
moved=[j for j in plan['jobs'] if j['family']=='fetch' and j['status'] in ['queued','waiting_selection']]
assert moved
assert all(not Path(j['output']).exists() for j in moved)
ids={j['id'] for j in moved};plan['jobs']=[j for j in plan['jobs'] if j['id'] not in ids];plan['counts']=dict(collections.Counter(j['status'] for j in plan['jobs']))
record=dict(at=time.time(),reason='User explicitly authorized server1 for current F1/A1 campaign',from_host=4,to_host=1,jobs=moved,preserved_running=before,backup=str(backup))
(p/'host1-transfer.json').write_text(json.dumps(record,indent=2));tmp=p/'plan.transfer.tmp';tmp.write_text(json.dumps(plan,indent=2));tmp.replace(p/'plan.json')
fcntl.flock(lock,fcntl.LOCK_UN);lock.close();fcntl.flock(admission,fcntl.LOCK_UN);admission.close()
for _ in range(10):
 status=subprocess.run(['sudo','-n','supervisorctl','status',name],capture_output=True,text=True).stdout
 if 'EXITED' in status:break
 time.sleep(1)
r=subprocess.run(['sudo','-n','supervisorctl','start',name],capture_output=True,text=True);assert r.returncode==0,r.stdout+r.stderr
pause.unlink();record['live_after']=[dict(id=j['id'],same_pid=ident(j['pid'])==j['start_ticks']) for j in before]
assert all(j['same_pid'] for j in record['live_after']);(p/'host1-transfer.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
''',timeout=100)
(P/'host1-expansion.json').write_text(json.dumps(receipt,indent=2))
result=call('ibolt-server-1','R='+repr(R)+'\nRECEIPT='+repr(receipt)+'\n'+r'''
from pathlib import Path
import subprocess,json,collections,itertools,time
p=Path(R);plan=json.loads((p/'plan.json').read_text());moved=RECEIPT['jobs']
for j in moved:j['previous_host']=4;j['host']=1
pilots=[j for j in moved if j['phase']=='pilot'];sac=[j for j in moved if j['phase']=='baseline'];wait=[j for j in moved if j['phase']=='confirm']
queued=[j for pair in itertools.zip_longest(pilots,sac) for j in pair if j]
existing={j['id'] for j in plan['jobs']};assert not existing.intersection(j['id'] for j in moved)
plan['jobs']+=queued+wait;plan['status']='prepared';plan['counts']=dict(collections.Counter(j['status'] for j in plan['jobs']));plan['expanded_unix']=time.time()
(p/'host1-expansion.json').write_text(json.dumps(RECEIPT,indent=2));tmp=p/'plan.expand.tmp';tmp.write_text(json.dumps(plan,indent=2));tmp.replace(p/'plan.json')
r=subprocess.run(['sudo','-n','supervisorctl','start','ibolt-f1-a1-temp-20261112-host1'],capture_output=True,text=True);assert r.returncode==0,r.stdout+r.stderr
print(json.dumps(dict(started=True,transferred=len(moved),counts=plan['counts'])))
''')
print(json.dumps(dict(transfer=result,preserved_learners=receipt['live_after']),indent=2))
