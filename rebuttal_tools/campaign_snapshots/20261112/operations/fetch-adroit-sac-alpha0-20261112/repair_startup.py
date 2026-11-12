from pathlib import Path
import json,hashlib,shutil,sys,subprocess,tarfile,shlex,time
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'fetch-adroit-f1-a1-nm-20261112'));from preflight import ssh,ROOT
sys.path.insert(0,str(P.parent/'fetch-adroit-f1-a1-entropy-20261112'));from deploy import freeze,resume
TAG='sac-alpha0-20261112';proto=json.loads((P/'protocol.json').read_text())
shutil.copy2(P/'protocol.json',P/'protocol.v1.json')
for variant in ['fetch','fetch-slide-20cm']:
 old=P/('alpha0-'+variant+'-source');new=P/('alpha0-'+variant+'-v2-source');shutil.copytree(old,new)
 worker=new/'rebuttal_tools/fetch_jax_sac_worker.py';s=worker.read_text()
 text="entropy='fixed', fixed_alpha=ent_coef, automatic_entropy_tuning=False"
 assert s.count(text)==1;s=s.replace(text,"entropy='fixed', fixed_alpha=ent_coef");worker.write_text(s)
 m=json.loads((new/'source-sha256.json').read_text());m['rebuttal_tools/fetch_jax_sac_worker.py']=hashlib.sha256(worker.read_bytes()).hexdigest();(new/'source-sha256.json').write_text(json.dumps(m,indent=2,sort_keys=True)+'\n')
 digest=hashlib.sha256((new/'source-sha256.json').read_bytes()).hexdigest()
 proto['sources'][variant]['manifest_sha256']=digest;proto['sources'][variant]['directory']=new.name
 for j in proto['rows']:
  if j['source'].endswith(old.name):
   j.update(source=ROOT+'/'+new.name,source_manifest=ROOT+'/'+new.name+'/source-sha256.json',source_manifest_sha256=digest)
proto['startup_repair']=dict(reason='Duplicate automatic_entropy_tuning keyword in metadata dict before first environment/update',source_change='Only remove redundant metadata keyword; JAX module and validated numerical path unchanged',preserve='All failed attempt logs/configs/specs and original sources',affected='Fetch eight runs, zero optimizer updates')
(P/'protocol.json').write_text(json.dumps(proto,indent=2)+'\n')
with tarfile.open(P/'repair.tgz','w:gz') as t:
 for v in ['fetch','fetch-slide-20cm']:t.add(P/('alpha0-'+v+'-v2-source'),arcname='alpha0-'+v+'-v2-source')
 t.add(P/'protocol.json',arcname=TAG+'/protocol.v2.json')
with (P/'repair.tgz').open('rb') as f:
 r=subprocess.run(['ssh','-T','-o','BatchMode=yes','ibolt-server-4','tar -xz -C '+ROOT],stdin=f,capture_output=True,timeout=50);assert not r.returncode,r.stderr
def call(code,**v):return ssh(4,'\n'.join(f'{k}={x!r}' for k,x in dict(ROOT=ROOT,TAG=TAG,HOST=4,**v).items())+'\n'+code)
stage=call(r'''
from pathlib import Path
import json,hashlib,subprocess,shutil
p=Path(ROOT)
for v in ['fetch','fetch-slide-20cm']:
 s=p/('alpha0-'+v+'-v2-source')
 for f,h in json.loads((s/'source-sha256.json').read_text()).items():assert hashlib.sha256((s/f).read_bytes()).hexdigest()==h
 assert hashlib.sha256((s/'baselines/jax_sac.py').read_bytes()).hexdigest()==json.loads((p/TAG/'validation-fetch.json').read_text())['module_sha256']
 subprocess.run(['git','init','-q','-b','heechan-rebuttal',str(s)],check=True);subprocess.run(['git','add','.'],cwd=s,check=True)
 subprocess.run(['git','-c','user.name=Experiment Snapshot','-c','user.email=experiment@local','commit','-q','-m','Correct alpha zero startup metadata duplicate key'],cwd=s,check=True)
print(json.dumps({n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in ['controller.py','archive_gateway.py']}))
''')
held=False;mutated=False
try:
 before=call(freeze,OLD_CONTROLLER=stage['controller.py'],OLD_GATEWAY=stage['archive_gateway.py']);held=True
 result=call(r'''
from pathlib import Path
import json,fcntl,hashlib,os,sys,shutil,time,collections
p=Path(ROOT);stage=p/TAG;sys.path.insert(0,str(p));import controller
lock=(p/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
plan=controller.read(p/'plan.json');old=controller.read(stage/'plan.before.json');assert plan==old
new=controller.read(stage/'protocol.v2.json');jobs={j['id']:j for j in new['rows'] if j['host']==4};receipt=[]
failed=stage/'startup-failures';failed.mkdir(exist_ok=True)
for j in plan['jobs']:
 if j['id'] not in jobs:continue
 assert controller.identity(j['pid']) is None,(j['id'],'still live')
 o=Path(j['output']);pr=controller.read(o/'progress.json');err=controller.read(o/'failure.json')
 assert pr['steps']==pr['updates']==0 and 'multiple values' in err['error'] and 'automatic_entropy_tuning' in err['error']
 dst=failed/j['id'];dst.mkdir(exist_ok=False);shutil.move(str(o),dst/'output')
 for folder,suffix in [('specs','.json'),('logs','.log')]:shutil.move(str(p/folder/(j['id']+suffix)),dst/(folder+suffix))
 previous=dict(j);j.clear();j.update(jobs[previous['id']]);j['startup_repair']=dict(at=time.time(),previous_pid=previous['pid'],previous_start_ticks=previous['start_ticks'],archived_attempt=str(dst),optimizer_updates=0,reason=err['error'])
 plan['source_manifests'][Path(j['source']).name.removesuffix('-source')]=j['source_manifest_sha256'];receipt.append(j['startup_repair'])
assert len(receipt)==8
for i,j in enumerate(plan['jobs']):
 if j['id'] not in jobs:assert j==old['jobs'][i]
shutil.copy2(stage/'protocol.json',stage/'protocol.v1.json');shutil.copy2(stage/'protocol.v2.json',stage/'protocol.json')
plan['counts']=dict(collections.Counter(j['status'] for j in plan['jobs']));controller.write(p/'plan.json',plan)
controller.write(stage/'startup-repair.json',dict(passed=True,runs=8,only_zero_update_attempts=True,repairs=receipt))
print(json.dumps(dict(passed=True,runs=8,existing_learners_preserved=True)))
''');mutated=True;(P/'startup-repair.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
finally:
 if held:print(call(resume),flush=True)
