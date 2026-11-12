"""Append the authorized entropy jobs without interrupting running learner groups."""
from pathlib import Path
import subprocess,json,hashlib,tarfile,concurrent.futures,time
P=Path(__file__).resolve().parent;OLD=P.parent/'fetch-adroit-f1-a1-temp-20261112'
ROOT='/workspace/heechan-unified-f1-a1-20261112';TAG='entropy-grid-20261112'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def call(h,code,**values):
    text='\n'.join(f'{k}={v!r}' for k,v in dict(ROOT=ROOT,TAG=TAG,HOST=h,**values).items())+'\n'+code
    r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15',f'ibolt-server-{h}','python3 -'],input=text,text=True,capture_output=True,timeout=55)
    if r.returncode:raise RuntimeError(r.stdout+r.stderr)
    return json.loads(r.stdout)

def stage(h):
    archive=P/f'stage-host{h}.tgz'
    with tarfile.open(archive,'w:gz') as t:
        for name in ['controller.py','archive_gateway.py','entropy_dependencies.py']:
            t.add(P/name,arcname=name)
        t.add(P/'protocol.json',arcname='entropy-protocol.json')
        t.add(P/f'additions-host{h}.json',arcname='additions.json')
        for name in ['entropy-fetch-source','entropy-fetch-slide-20cm-source','entropy-adroit-source']:
            t.add(P/name,arcname=name,filter=lambda x: None if '__pycache__' in Path(x.name).parts or '.git' in Path(x.name).parts else x)
    remote=ROOT+'/'+TAG
    r=subprocess.run(['ssh','-T','-o','BatchMode=yes',f'ibolt-server-{h}','mkdir -p '+remote],capture_output=True,timeout=20);assert r.returncode==0,r.stderr
    with archive.open('rb') as f:
        r=subprocess.run(['ssh','-T','-o','BatchMode=yes',f'ibolt-server-{h}','tar -xz -C '+remote],stdin=f,capture_output=True,timeout=50);assert r.returncode==0,r.stderr
    out=call(h,r'''
from pathlib import Path
import json,hashlib,subprocess,shutil
p=Path(ROOT);stage=p/TAG;protocol=json.loads((stage/'entropy-protocol.json').read_text());manifests={}
for variant,record in protocol['sources'].items():
 source=stage/('entropy-'+variant+'-source');m=source/'source-sha256.json'
 assert hashlib.sha256(m.read_bytes()).hexdigest()==record['manifest_sha256']
 for name,h in json.loads(m.read_text()).items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==h,name
 assert not (p/source.name).exists(),source
 shutil.move(str(source),p/source.name);source=p/source.name
 subprocess.run(['git','init','-q','-b','heechan-rebuttal',str(source)],check=True)
 subprocess.run(['git','add','.'],cwd=source,check=True)
 subprocess.run(['git','-c','user.name=Experiment Snapshot','-c','user.email=experiment@local','commit','-q','-m','Freeze authorized DACER entropy parameter sweep adapter'],cwd=source,check=True)
 manifests[source.name]=record['manifest_sha256']
shutil.copy2(stage/'entropy-protocol.json',p/'entropy-protocol.json')
proof=dict(host=HOST,source_manifests=manifests,protocol_sha256=hashlib.sha256((p/'entropy-protocol.json').read_bytes()).hexdigest())
(p/'entropy-transfer-proof.json').write_text(json.dumps(proof,indent=2))
print(json.dumps(dict(passed=True,host=HOST,proof=proof)))
''')
    (P/f'staged-host{h}.json').write_text(json.dumps(out,indent=2));return out

freeze=r'''
from pathlib import Path
import json,time,os,signal,fcntl,subprocess,hashlib
p=Path(ROOT);assert not (p/'PAUSE').exists()
assert hashlib.sha256((p/'controller.py').read_bytes()).hexdigest()==OLD_CONTROLLER
assert hashlib.sha256((p/'archive_gateway.py').read_bytes()).hexdigest()==OLD_GATEWAY
stamp=time.time();(p/'PAUSE').write_text(TAG+'; admission hold only, preserve learners\n')
for _ in range(35):
 plan=json.loads((p/'plan.json').read_text())
 if plan['updated_unix']>stamp:break
 time.sleep(1)
else:raise RuntimeError('Controller did not acknowledge admission pause')
pid=int(subprocess.check_output(['sudo','-n','supervisorctl','pid',f'ibolt-f1-a1-temp-20261112-host{HOST}'],text=True).strip())
assert pid>1 and Path('/proc',str(pid)).stat().st_uid==os.getuid()
assert str(p/'controller.py').encode() in Path('/proc',str(pid),'cmdline').read_bytes().split(b'\0')
lock=Path('/workspace/.ibolt-capacity-admission.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
os.kill(pid,signal.SIGKILL)
control=(p/'controller.lock').open('a');fcntl.flock(control,fcntl.LOCK_EX)
plan=json.loads((p/'plan.json').read_text());(p/TAG/'plan.before.json').write_text(json.dumps(plan,indent=2))
print(json.dumps(plan))
'''
apply=r'''
from pathlib import Path
import json,shutil,fcntl,collections,time,hashlib
p=Path(ROOT);assert (p/'PAUSE').read_text().startswith(TAG)
lock=(p/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
stage=p/TAG;plan=json.loads((p/'plan.json').read_text());before=json.loads((stage/'plan.before.json').read_text());assert plan==before
jobs=json.loads((stage/'additions.json').read_text());ids={j['id'] for j in plan['jobs']}
assert all(j['id'] not in ids and j['host']==HOST and j['phase'] in ['entropy_pilot','entropy_confirm'] and j['temperature'] is None for j in jobs)
assert all(not (p/'runs'/j['id']).exists() and not (p/'specs'/(j['id']+'.json')).exists() for j in jobs)
for name in ['controller.py','archive_gateway.py']:
 shutil.copy2(p/name,stage/(name+'.before'));shutil.copy2(stage/name,p/name)
shutil.copy2(stage/'entropy_dependencies.py',p/'entropy_dependencies.py')
for folder,h in json.loads((p/'entropy-transfer-proof.json').read_text())['source_manifests'].items():plan['source_manifests'][folder.removesuffix('-source')]=h
plan['jobs'].extend(jobs);assert plan['jobs'][:len(before['jobs'])]==before['jobs']
if HOST==5:plan['excluded_gpus']=sorted(set(plan.get('excluded_gpus',[]))|{3})
plan['entropy_protocol']=str(p/'entropy-protocol.json');plan['counts']=dict(collections.Counter(j['status'] for j in plan['jobs']));plan['updated_unix']=time.time()
tmp=p/'plan.entropy.tmp';tmp.write_text(json.dumps(plan,indent=2));tmp.replace(p/'plan.json')
print(json.dumps(dict(host=HOST,added=len(jobs),counts=plan['counts'],all_old_jobs_preserved=True)))
'''
resume=r'''
from pathlib import Path
import json,subprocess
p=Path(ROOT);assert (p/'PAUSE').read_text().startswith(TAG)
r=subprocess.run(['sudo','-n','supervisorctl','start',f'ibolt-f1-a1-temp-20261112-host{HOST}'],text=True,capture_output=True)
assert r.returncode==0,r.stdout+r.stderr
(p/'PAUSE').unlink();print(json.dumps(dict(host=HOST,resumed=True)))
'''

if __name__=='__main__':
    assert json.loads((P/'queue-check.json').read_text())['passed']
    for h in [1,3,4,5]:print('STAGED',h,stage(h)['passed'],flush=True)
    held={};applied=[]
    try:
        for h in [1,3,4,5]:
            held[h]=call(h,freeze,OLD_CONTROLLER=sha(P/'controller-host5.before.py') if h==5 else sha(OLD/'controller.py'),OLD_GATEWAY=sha(OLD/'archive_gateway.py'))
            (P/f'before-host{h}.json').write_text(json.dumps(held[h],indent=2));print('ADMISSION_HELD',h,flush=True)
        for h in held:
            d=call(h,apply);applied.append(h);(P/f'applied-host{h}.json').write_text(json.dumps(d,indent=2));print('APPENDED',d,flush=True)
    except BaseException:
        for h in applied:
            call(h,"from pathlib import Path\nimport shutil,json\np=Path(ROOT);s=p/TAG\nfor name in ['controller.py','archive_gateway.py']:shutil.copy2(s/(name+'.before'),p/name)\nshutil.copy2(s/'plan.before.json',p/'plan.json')\nprint('{}')")
        raise
    finally:
        for h in held:
            d=call(h,resume);(P/f'resumed-host{h}.json').write_text(json.dumps(d));print('ADMISSION_RESUMED',h,flush=True)
