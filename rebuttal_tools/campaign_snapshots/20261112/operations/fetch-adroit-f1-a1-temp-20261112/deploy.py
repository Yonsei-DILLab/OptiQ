from pathlib import Path
import json,subprocess,tarfile,io,concurrent.futures,hashlib
P=Path(__file__).resolve().parent;R='/workspace/heechan-unified-f1-a1-20261112'
preflight=r'''
from pathlib import Path
import os,sys,json,hashlib,importlib.metadata
root=Path(__file__).resolve().parent;plan=json.loads((root/'plan.json').read_text())
sys.path.insert(0,str(root/'adroit-source'))
from adroit_rebuttal.verified_demos import resolve
proofs={}
for task,data in plan['datasets'].items():
 path,proof=resolve(task,data);assert proof['passed'] and proof['replayed'] and proof['successful_trajectories']==25;proofs[task]=proof
for family,digest in plan['source_manifests'].items():
 source=root/(family+'-source');manifest=source/'source-sha256.json';assert hashlib.sha256(manifest.read_bytes()).hexdigest()==digest
 for name,h in json.loads(manifest.read_text()).items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==h
report=dict(passed=True,demos=proofs,packages={n:importlib.metadata.version(n) for n in ['gymnasium','gymnasium-robotics','mujoco','jax','torch','stable-baselines3','flax']})
(root/'preflight.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(passed=True,host=plan['host'],demos={t:p['successful_trajectories'] for t,p in proofs.items()},packages=report['packages'])))
'''
(P/'preflight.py').write_text(preflight)
# Source changes are already frozen and source-manifest checked; controller is outside those snapshots.
for h in [3,4,5]:
 with tarfile.open(P/f'deploy-host{h}.tgz','w:gz') as t:
  for name in ['sources.tgz','datasets.tgz','controller.py','preflight.py','fetch-reference.json','jax-validation.json','protocol.json']:t.add(P/name,arcname=name)
  t.add(P/f'plan-host{h}.json',arcname='plan.json')

def one(h):
 cmd=['ssh','-T','-o','BatchMode=yes',f'ibolt-server-{h}']
 with (P/f'deploy-host{h}.tgz').open('rb') as f:
  r=subprocess.run(cmd+['tar -xz -C '+R],stdin=f,capture_output=True,timeout=180);assert not r.returncode,r.stderr
 code='ROOT='+repr(R)+'\nHOST='+repr(h)+'\n'+r'''
from pathlib import Path
import tarfile,subprocess,json,hashlib,os,shutil
root=Path(ROOT)
for name in ['sources','datasets']:
 with tarfile.open(root/(name+'.tgz')) as t:t.extractall(root,filter='data')
for name in ['fetch-source','adroit-source']:
 src=root/name
 if not (src/'.git').exists():
  subprocess.run(['git','init','-q','-b','heechan-rebuttal',str(src)],check=True)
  subprocess.run(['git','add','.'],cwd=src,check=True)
  subprocess.run(['git','-c','user.name=Experiment Snapshot','-c','user.email=experiment@local','commit','-q','-m','Freeze F1 A1 source for authorized temperature comparison'],cwd=src,check=True)
for d in ['runs','preflights','logs','specs','selections']:(root/d).mkdir(exist_ok=True)
plan=json.loads((root/'plan.json').read_text())
for j in plan['jobs']:
 if j['status']=='reused':
  old=Path(j['reuse_output']);dst=Path(j['output']);assert not dst.exists()
  cfg=json.loads((old/'mujoco-config.json').read_text());assert cfg['seed']==j['seed'] and cfg['temperature']==j['temperature'] and cfg['dacer']['enabled']
  assert cfg['alg']['actor']['hidden_dims']==cfg['alg']['critic']['hs']==[256,256] and cfg['alg']['utd']==.125 and cfg['alg']['gamma']==.99
  shutil.copytree(old,dst,copy_function=os.link)
  (dst/'REUSED.json').write_text(json.dumps(dict(original=str(old),reason='Original matching completed F1 pilot; not selected by outcome',original_config_sha256=hashlib.sha256((old/'mujoco-config.json').read_bytes()).hexdigest())))
(root/'plan.initial.json').write_bytes((root/'plan.json').read_bytes())
proof={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['sources.tgz','datasets.tgz','protocol.json','plan.initial.json']}
(root/'transfer-proof.json').write_text(json.dumps(proof,indent=2))
print(json.dumps(dict(host=HOST,deployed=True,free_gib=shutil.disk_usage(root).free/2**30)))
'''
 r=subprocess.run(cmd+['python3 -'],input=code,text=True,capture_output=True,timeout=90);assert not r.returncode,r.stderr;print(r.stdout,flush=True)
 py=json.loads((P/f'plan-host{h}.json').read_text())['python']['adroit']
 command=f'cd {R}/adroit-source && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 {py} {R}/preflight.py'
 r=subprocess.run(cmd+[command],text=True,capture_output=True,timeout=120);(P/f'preflight-host{h}.log').write_text(r.stdout+r.stderr);assert not r.returncode,r.stderr[-2000:];return r.stdout
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as e:
 for r in e.map(one,[3,4,5]):print(r,flush=True)
