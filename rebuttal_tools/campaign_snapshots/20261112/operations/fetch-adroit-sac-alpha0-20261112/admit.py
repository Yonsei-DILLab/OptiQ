from pathlib import Path
import json,sys,hashlib,subprocess,shlex,time
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'fetch-adroit-f1-a1-nm-20261112'));from preflight import ssh,ROOT
sys.path.insert(0,str(P.parent/'fetch-adroit-f1-a1-entropy-20261112'));from deploy import freeze,resume
TAG='sac-alpha0-20261112'
proto=json.loads((P/'protocol.json').read_text())
def call(h,code,**vals):
 return ssh(h,'\n'.join(f'{k}={v!r}' for k,v in dict(ROOT=ROOT,TAG=TAG,HOST=h,**vals).items())+'\n'+code)
def save(name,d):(P/name).write_text(json.dumps(d,indent=2)+'\n')
hosts=[int(x) for x in sys.argv[1:]] or [3,4]
for fam in [('adroit' if h==3 else 'fetch') for h in hosts]:
 d=json.loads((P/f'validation-{fam}.json').read_text());assert d['passed']
 assert d['module_sha256']==hashlib.sha256((P/('alpha0-'+fam+'-source')/'baselines/jax_sac.py').read_bytes()).hexdigest()
# The existing restricted collector must already have preserved every frozen source byte.
code='EXPECTED='+repr(proto['sources'])+'\n'+r'''
from pathlib import Path
import json,hashlib
p=Path('/data1/heechan/iBOLT/20261112-f1-a1-temp');rows=[]
for variant,v in EXPECTED.items():
 h=3 if variant=='adroit' else 4;s=p/f'hosts/host{h}/workspace/heechan-unified-f1-a1-20261112'/('alpha0-'+variant+'-source')
 m=s/'source-sha256.json';assert m.exists(),str(m)
 assert hashlib.sha256(m.read_bytes()).hexdigest()==v['manifest_sha256']
 for name,digest in json.loads(m.read_text()).items():assert hashlib.sha256((s/name).read_bytes()).hexdigest()==digest,name
 rows.append(dict(host=h,source=str(s),manifest_sha256=v['manifest_sha256']))
print(json.dumps(dict(passed=True,sources=rows)))
'''
r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-p','2200','heechan@dil.yonsei.ac.kr','python3 -c '+shlex.quote(code)],capture_output=True,text=True,timeout=40)
assert r.returncode==0,r.stderr
receipt=json.loads(r.stdout);save('central-source-receipt.json',receipt)
mutation=r'''
from pathlib import Path
import json,hashlib,fcntl,collections,time,shutil,ast
p=Path(ROOT);stage=p/TAG;assert (p/'PAUSE').read_text().startswith(TAG)
lock=(p/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
plan=json.loads((p/'plan.json').read_text());before=json.loads((stage/'plan.before.json').read_text());assert plan==before
jobs=json.loads((stage/'additions.json').read_text());ids={j['id'] for j in plan['jobs']}
assert len(jobs)==8 and all(j['id'] not in ids and j['host']==HOST and j['phase']=='sac_alpha0_pilot' and j['sac_ent_coef']==0. and j['status']=='queued' for j in jobs)
assert all(not Path(j['output']).exists() for j in jobs)
for j in jobs:
 v=json.loads(Path(j['validation_manifest']).read_text());assert v['passed'] and v['module_sha256']==hashlib.sha256((Path(j['source'])/'baselines/jax_sac.py').read_bytes()).hexdigest()
 assert hashlib.sha256(Path(j['source_manifest']).read_bytes()).hexdigest()==j['source_manifest_sha256']
 if j['family']=='adroit':
  assert hashlib.sha256(Path(j['demo_path']).read_bytes()).hexdigest()==j['demo_sha256']
  assert hashlib.sha256(Path(j['demonstration_manifest']).read_bytes()).hexdigest()==j['demonstration_manifest_sha256']
 plan['source_manifests'][Path(j['source']).name.removesuffix('-source')]=j['source_manifest_sha256']
s=(p/'controller.py').read_text();shutil.copy2(p/'controller.py',stage/'controller.before.py')
marker=" j['config_verified']=True"
assert s.count(marker)==1
s=s.replace(marker," if j['phase']=='sac_alpha0_pilot' and 'entropy_coefficient' in c:\n  assert c['entropy_coefficient']==0. and not c['automatic_entropy_tuning']\n"+marker)
marker='  def gate(j):\n';assert s.count(marker)==1
s=s.replace(marker,marker+"   if j['phase']=='sac_alpha0_pilot':\n    proof=read(Path(j['validation_manifest']),{})\n    if not proof.get('passed') or proof['module_sha256']!=hashlib.sha256((Path(j['source'])/'baselines/jax_sac.py').read_bytes()).hexdigest():return False\n")
ast.parse(s);(p/'controller.py').write_text(s)
(stage/'central-source-receipt.json').write_text(json.dumps(RECEIPT,indent=2))
plan['jobs'].extend(jobs);assert plan['jobs'][:len(before['jobs'])]==before['jobs']
plan['counts']=dict(collections.Counter(j['status'] for j in plan['jobs']));plan['updated_unix']=time.time()
t=p/'plan.alpha0.tmp';t.write_text(json.dumps(plan,indent=2));t.replace(p/'plan.json')
print(json.dumps(dict(host=HOST,added=8,counts=plan['counts'],existing_learners_preserved=True)))
'''
for h in hosts:
 hashes=call(h,"from pathlib import Path\nimport json,hashlib\np=Path(ROOT);print(json.dumps({n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in ['controller.py','archive_gateway.py']}))")
 held=False;committed=False
 try:
  before=call(h,freeze,OLD_CONTROLLER=hashes['controller.py'],OLD_GATEWAY=hashes['archive_gateway.py']);held=True;save(f'plan-before-host{h}.json',before)
  result=call(h,mutation,RECEIPT=receipt);committed=True;save(f'admitted-host{h}.json',result);print(result,flush=True)
 finally:
  if held and not committed:
   call(h,"from pathlib import Path\nimport shutil\np=Path(ROOT);s=p/TAG;shutil.copy2(s/'plan.before.json',p/'plan.json')\nif (s/'controller.before.py').exists():shutil.copy2(s/'controller.before.py',p/'controller.py')\nprint('{}')")
  if held:
   r=call(h,resume);save(f'resumed-host{h}.json',r);print(r,flush=True)
