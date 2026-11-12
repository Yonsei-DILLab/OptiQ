"""Move two unstarted Fetch pilots to verified spare host3 slots; retain learners."""
from pathlib import Path
import collections, copy, json, subprocess, time

P = Path(__file__).resolve().parent
ROOT = '/workspace/heechan-unified-f1-a1-20261112'
TAG = 'rebalance-fetch-20261112-0516'
DEST = P / TAG
DEST.mkdir(exist_ok=False)

def call(host, code, **values):
    payload = '\n'.join(f'{k}={v!r}' for k,v in dict(ROOT=ROOT,TAG=TAG,HOST=host,**values).items())+'\n'+code
    result = subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15',f'ibolt-server-{host}','python3 -'],input=payload,text=True,capture_output=True,timeout=55)
    if result.returncode: raise RuntimeError(result.stdout+result.stderr)
    return json.loads(result.stdout)

freeze = '''
from pathlib import Path
import json,time,os,signal,fcntl,subprocess
p=Path(ROOT); assert not (p/'PAUSE').exists()
stamp=time.time(); (p/'PAUSE').write_text(TAG+'; hold admission only; preserve learners\\n')
for _ in range(40):
 plan=json.loads((p/'plan.json').read_text())
 if plan['updated_unix']>stamp: break
 time.sleep(1)
else: raise RuntimeError('Admission hold not acknowledged')
pid=int(subprocess.check_output(['sudo','-n','supervisorctl','pid',f'ibolt-f1-a1-temp-20261112-host{HOST}'],text=True).strip())
assert pid>1 and Path('/proc',str(pid)).stat().st_uid==os.getuid()
assert str(p/'controller.py').encode() in Path('/proc',str(pid),'cmdline').read_bytes().split(b'\\0')
admission=Path('/workspace/.ibolt-capacity-admission.lock').open('a');fcntl.flock(admission,fcntl.LOCK_EX)
os.kill(pid,signal.SIGKILL)
lock=(p/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
plan=json.loads((p/'plan.json').read_text());folder=p/TAG;folder.mkdir(exist_ok=False)
(folder/'plan.before.json').write_text(json.dumps(plan,indent=2))
print(json.dumps(plan))
'''
apply = '''
from pathlib import Path
import json,time,collections,hashlib,fcntl
p=Path(ROOT);assert (p/'PAUSE').read_text().startswith(TAG)
lock=(p/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
current=json.loads((p/'plan.json').read_text());old={j['id']:j for j in current['jobs']};new={j['id']:j for j in PLAN['jobs']}
for jid in old.keys()^new.keys():
 j=old.get(jid,new.get(jid));assert j['status']=='queued' and j['phase']=='pilot' and j['family']=='fetch' and j['task'] in ['push','pick-and-place']
 assert not any((p/sub/name).exists() for sub,name in [('runs',jid),('logs',jid+'.log'),('specs',jid+'.json')])
for jid in old.keys()&new.keys():assert old[jid]==new[jid]
assert hashlib.sha256((p/'fetch-source/source-sha256.json').read_bytes()).hexdigest()=='b1286171495a26117e353a6f1ed75252a038909b4e7532aef305056ced1ad3c2'
assert all(json.loads((p/'fetch-smoke-gates.json').read_text()).get(t+'-ibolt') for t in ['push','pick-and-place'])
PLAN['counts']=dict(collections.Counter(j['status'] for j in PLAN['jobs']));PLAN['updated_unix']=time.time()
(p/TAG/'transfer.json').write_text(json.dumps(RECEIPT,indent=2))
tmp=p/'plan.rebalance.tmp';tmp.write_text(json.dumps(PLAN,indent=2));tmp.replace(p/'plan.json')
print(json.dumps(dict(host=HOST,counts=PLAN['counts'])))
'''
resume = '''
from pathlib import Path
import json,subprocess
p=Path(ROOT);assert (p/'PAUSE').read_text().startswith(TAG)
result=subprocess.run(['sudo','-n','supervisorctl','start',f'ibolt-f1-a1-temp-20261112-host{HOST}'],capture_output=True,text=True)
assert result.returncode==0,result.stdout+result.stderr
(p/'PAUSE').unlink();print(json.dumps(dict(host=HOST,resumed=True)))
'''

before = {}
applied = []
try:
    for host in [1,3]:
        before[host]=call(host,freeze)
        (DEST/f'before-host{host}.json').write_text(json.dumps(before[host],indent=2))
        print('ADMISSION_HELD',host,flush=True)
    plans=copy.deepcopy(before)
    assert sum(j['status']=='running' for j in plans[3]['jobs'])<=18
    moved=[j for j in plans[1]['jobs'] if j['status']=='queued' and j['phase']=='pilot' and j['task'] in ['push','pick-and-place']][:2]
    assert len(moved)==2
    ids={j['id'] for j in moved}
    plans[1]['jobs']=[j for j in plans[1]['jobs'] if j['id'] not in ids]
    for j in moved:
        j['previous_host']=1;j['host']=3
    pos=next((i for i,j in enumerate(plans[3]['jobs']) if j['phase']=='confirm'),len(plans[3]['jobs']))
    plans[3]['jobs'][pos:pos]=moved
    alljobs=[j for plan in plans.values() for j in plan['jobs']]
    assert len(alljobs)==len({j['id'] for j in alljobs})
    receipt=dict(at=time.time(),from_host=1,to_host=3,jobs=moved,reason='Fill two verified spare GPU slots using only queued unstarted jobs; preserve all running learner PIDs and scientific settings')
    (DEST/'transfer.json').write_text(json.dumps(receipt,indent=2))
    for host in [1,3]:
        result=call(host,apply,PLAN=plans[host],RECEIPT=receipt)
        applied.append(host);(DEST/f'applied-host{host}.json').write_text(json.dumps(result,indent=2))
        print('APPLIED',host,result['counts'],flush=True)
except BaseException:
    # Restore original plans before releasing admission if a transaction did not finish.
    for host in applied:
        call(host,"from pathlib import Path\nimport json\np=Path(ROOT);tmp=p/'plan.rollback.tmp';tmp.write_text(json.dumps(PLAN,indent=2));tmp.replace(p/'plan.json');print('{}')",PLAN=before[host])
    raise
finally:
    for host in before:
        result=call(host,resume)
        (DEST/f'resumed-host{host}.json').write_text(json.dumps(result,indent=2))
        print('ADMISSION_RESUMED',host,flush=True)
