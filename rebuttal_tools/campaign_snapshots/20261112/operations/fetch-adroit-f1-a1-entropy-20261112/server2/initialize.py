from pathlib import Path
import json,subprocess,hashlib,tarfile,shutil
R='/workspace/heechan-unified-f1-a1-20261112'
P=Path(__file__).resolve().parent;OLD=P.parent.parent/'fetch-adroit-f1-a1-temp-20261112'
plan=json.loads((OLD/'expanded-host1.json').read_text())['plan']
plan.update(host=2,jobs=[],counts={},status='prepared',python={'fetch':R+'/runtime/bin/python','adroit':R+'/adroit-runtime/bin/python'},
            physical_cores_per_job=3,min_free_gib=12,disk_reserve_mib_per_job=512,max_jobs_per_gpu=5,excluded_gpus=[])
plan.pop('updated_unix',None);plan.pop('free_gib',None)
result=subprocess.run(['ssh','-T','-o','BatchMode=yes','-p','2200','heechan@dil.yonsei.ac.kr','cat /home/heechan/.ssh/ibolt_archive_20261112.pub'],text=True,capture_output=True,timeout=20);assert result.returncode==0,result.stderr
pub=result.stdout.strip();assert pub.startswith('ssh-ed25519 ')
code='PLAN='+repr(plan)+'\nPUBLIC_KEY='+repr(pub)+'\n'+r'''
from pathlib import Path
import json,subprocess,hashlib,tarfile,shutil,os
p=Path('/workspace/heechan-unified-f1-a1-20261112')
assert not (p/'plan.json').exists()
with tarfile.open(p/'datasets.tgz') as t:t.extractall(p,filter='data')
manifests={}
for source in sorted(p.glob('*-source')):
 m=source/'source-sha256.json'
 for name,h in json.loads(m.read_text()).items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==h,name
 manifests[source.name.removesuffix('-source')]=hashlib.sha256(m.read_bytes()).hexdigest()
 subprocess.run(['git','init','-q','-b','heechan-rebuttal',str(source)],check=True)
 subprocess.run(['git','add','.'],cwd=source,check=True)
 subprocess.run(['git','-c','user.name=Experiment Snapshot','-c','user.email=experiment@local','commit','-q','-m','Freeze identical validated F1 A1 source on authorized replacement server2'],cwd=source,check=True)
PLAN['source_manifests']=manifests
for name in ['runs','preflights','logs','specs','selections','entropy-selections','slide-20cm-20261112']:(p/name).mkdir(exist_ok=True)
(p/'plan.json').write_text(json.dumps(PLAN,indent=2));(p/'plan.initial.json').write_bytes((p/'plan.json').read_bytes())
proof={name:hashlib.sha256((p/name).read_bytes()).hexdigest() for name in ['datasets.tgz','protocol.json','plan.initial.json']}
(p/'transfer-proof.json').write_text(json.dumps(proof,indent=2))
eproof=dict(host=2,source_manifests={k+'-source':v for k,v in manifests.items() if k.startswith('entropy-')},protocol_sha256=hashlib.sha256((p/'entropy-protocol.json').read_bytes()).hexdigest())
(p/'entropy-transfer-proof.json').write_text(json.dumps(eproof,indent=2))
(p/'archive-access.json').write_text(json.dumps(dict(allowed_roots=[p.name],checkpoint_roots=[p.name],central='heechan@dil.yonsei.ac.kr:2200:/data1/heechan/iBOLT/20261112-f1-a1-temp')))
folder=Path.home()/'.ssh';folder.mkdir(mode=0o700,exist_ok=True);auth=folder/'authorized_keys'
line='restrict,command="/usr/bin/python3 '+str(p/'archive_gateway.py')+'" '+PUBLIC_KEY
prior=auth.read_text() if auth.exists() else '';assert PUBLIC_KEY.split()[1] not in prior
auth.write_text(prior.rstrip()+'\n'+line+'\n');auth.chmod(0o600)
(p/'PAUSE').write_text('server2 bootstrap: wait for environment and replay validation\n')
print(json.dumps(dict(passed=True,source_manifests=manifests,archive_key_restricted=True)))
'''
r=subprocess.run(['ssh','-T','-o','BatchMode=yes','ibolt-server-2','python3 -'],input=code,text=True,capture_output=True,timeout=55);assert r.returncode==0,r.stdout+r.stderr
(P/'initialized.json').write_text(r.stdout);print(r.stdout)
