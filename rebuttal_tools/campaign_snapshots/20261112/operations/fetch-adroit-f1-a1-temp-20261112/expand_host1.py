from pathlib import Path
import json,subprocess,tarfile,copy,shlex,sys
P=Path(__file__).resolve().parent;R='/workspace/heechan-unified-f1-a1-20261112'
def ssh(host,code,timeout=180):
 r=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15',host,'python3 -'],input=code,text=True,capture_output=True,timeout=timeout)
 if r.returncode:raise RuntimeError(r.stdout+r.stderr)
 print(r.stdout,flush=True);return r.stdout
base=json.loads((P/'plan-host4.json').read_text());base.update(host=1,python={'fetch':'/workspace/heechan-adroit/runtime/bin/python','adroit':'/workspace/heechan-adroit/runtime/bin/python'},min_free_gib=8,status='prepared')
base['jobs']=[]
for method in ['ibolt','sac']:
 template=json.loads((P/('plan-host3.json' if method=='ibolt' else 'plan-host4.json')).read_text())
 for j in template['jobs']:
  if j['phase']=='smoke':
   j=copy.deepcopy(j);j.update(host=1,seed=205,id=f"fetch-{j['task']}-{method}-s205-smoke-host1",status='queued');j['output']=R+'/preflights/'+j['id'];base['jobs'].append(j)
(P/'plan-host1.json').write_text(json.dumps(base,indent=2))
with tarfile.open(P/'deploy-host1.tgz','w:gz') as t:
 for name in ['sources.tgz','datasets.tgz','controller.py','preflight.py','fetch-reference.json','jax-validation.json','protocol.json','archive_gateway.py']:t.add(P/name,arcname=name)
 t.add(P/'plan-host1.json',arcname='plan.json')
if '--resume' not in sys.argv:
 ssh('ibolt-server-1',f"from pathlib import Path\np=Path({R!r});assert not p.exists();p.mkdir()")
files=['sources.tgz','datasets.tgz','controller.py','preflight.py','fetch-reference.json','jax-validation.json','protocol.json','archive_gateway.py','plan-host1.json']
r=subprocess.run(['rsync','-a','--partial','--timeout=90','-e','ssh -o BatchMode=yes']+[str(P/n) for n in files]+['ibolt-server-1:'+R+'/'],capture_output=True,timeout=600);assert not r.returncode,r.stderr
ssh('ibolt-server-1',f"from pathlib import Path\np=Path({R!r});assert not (p/'plan.initial.json').exists();(p/'plan.json').write_bytes((p/'plan-host1.json').read_bytes())")
pub='ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMi26dumpQqk58iMoZwWKR+/4UloJBzGd65UJoRugiHu ibolt-central-archive-20261112'
code='R='+repr(R)+'\nPUB='+repr(pub)+'\n'+r'''
from pathlib import Path
import json,subprocess,tarfile,hashlib
p=Path(R)
for n in ['sources','datasets']:
 with tarfile.open(p/(n+'.tgz')) as t:t.extractall(p,filter='data')
for n in ['fetch-source','adroit-source']:
 src=p/n;subprocess.run(['git','init','-q','-b','heechan-rebuttal',str(src)],check=True);subprocess.run(['git','add','.'],cwd=src,check=True);subprocess.run(['git','-c','user.name=Experiment Snapshot','-c','user.email=experiment@local','commit','-q','-m','Freeze identical F1 A1 source for server1 expansion'],cwd=src,check=True)
for n in ['runs','preflights','logs','specs','selections']:(p/n).mkdir(exist_ok=True)
(p/'plan.initial.json').write_bytes((p/'plan.json').read_bytes());proof={n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in ['sources.tgz','datasets.tgz','protocol.json','plan.initial.json']};(p/'transfer-proof.json').write_text(json.dumps(proof,indent=2))
(p/'archive-access.json').write_text(json.dumps(dict(allowed_roots=[p.name],checkpoint_roots=[p.name+'/runs'],central='dil.yonsei.ac.kr:/data1/heechan/iBOLT/20261112-f1-a1-temp'),indent=2));(p/'archive_gateway.py').chmod(0o700)
auth=Path.home()/'.ssh/authorized_keys';old=auth.read_text();line='restrict,command="/usr/bin/python3 '+str(p/'archive_gateway.py')+'" '+PUB
assert PUB not in old
with auth.open('a') as f:f.write(('\n' if not old.endswith('\n') else '')+line+'\n')
auth.chmod(0o600)
name='ibolt-f1-a1-temp-20261112-host1';config="""[program:{name}]
command=/usr/bin/python3 -u {root}/controller.py
directory={root}
user=heechan
autostart=false
autorestart=false
startsecs=2
startretries=0
stopasgroup=true
killasgroup=true
stopwaitsecs=30
redirect_stderr=true
stdout_logfile={root}/controller.log
stdout_logfile_maxbytes=10485760
""".format(name=name,root=R)
(p/'supervisor.conf').write_text(config);subprocess.run(['sudo','-n','install','-m','644',str(p/'supervisor.conf'),'/etc/supervisor/conf.d/'+name+'.conf'],check=True)
subprocess.run(['sudo','-n','supervisorctl','reread'],check=True);subprocess.run(['sudo','-n','supervisorctl','update',name],check=True)
print('DEPLOYED_HOST1')
'''
ssh('ibolt-server-1',code)
code='R='+repr(R)+'\n'+r'''
from pathlib import Path
import subprocess,json,os
py='/workspace/heechan-adroit/runtime/bin/python';env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',JAX_PLATFORMS='cuda',CUDA_VISIBLE_DEVICES='0',XLA_PYTHON_CLIENT_PREALLOCATE='false')
r=subprocess.run([py,R+'/preflight.py'],env=env,capture_output=True,text=True,timeout=180);Path(R+'/preflight.log').write_text(r.stdout+r.stderr);assert not r.returncode,r.stderr;print(r.stdout)
r=subprocess.run([py,'-c',"import jax,json;print(json.dumps({'backend':jax.default_backend(),'devices':[str(d) for d in jax.devices()]}));assert jax.default_backend()=='gpu'"],env=env,capture_output=True,text=True,timeout=60);assert not r.returncode,r.stderr;print(r.stdout)
'''
ssh('ibolt-server-1',code,timeout=240)
