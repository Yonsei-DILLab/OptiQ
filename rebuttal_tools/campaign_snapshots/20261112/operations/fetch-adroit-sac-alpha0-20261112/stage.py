from pathlib import Path
import sys,json,subprocess,tarfile,concurrent.futures
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'fetch-adroit-f1-a1-nm-20261112'));from preflight import ssh,ROOT
TAG='sac-alpha0-20261112'
def one(h):
 variants=['adroit'] if h==3 else ['fetch','fetch-slide-20cm']
 with tarfile.open(P/f'stage-host{h}.tgz','w:gz') as t:
  for v in variants:t.add(P/('alpha0-'+v+'-source'),arcname='alpha0-'+v+'-source')
  for name in ['protocol.json','validate.py']:t.add(P/name,arcname=TAG+'/'+name)
  t.add(P/f'additions-host{h}.json',arcname=TAG+'/additions.json')
 with (P/f'stage-host{h}.tgz').open('rb') as f:
  r=subprocess.run(['ssh','-T','-o','BatchMode=yes',f'ibolt-server-{h}','tar -xz -C '+ROOT],stdin=f,capture_output=True,timeout=55)
  assert r.returncode==0,r.stderr
 code='ROOT='+repr(ROOT)+'\nTAG='+repr(TAG)+'\nHOST='+str(h)+'\nVARIANTS='+repr(variants)+'\n'+r'''
from pathlib import Path
import sys,json,hashlib,subprocess,os,fcntl
p=Path(ROOT);stage=p/TAG;sys.path.insert(0,str(p));import controller
plan=controller.read(p/'plan.json');proto=controller.read(stage/'protocol.json')
for v in VARIANTS:
 s=p/('alpha0-'+v+'-source');m=s/'source-sha256.json'
 assert hashlib.sha256(m.read_bytes()).hexdigest()==proto['sources'][v]['manifest_sha256']
 for f,h in controller.read(m).items():assert hashlib.sha256((s/f).read_bytes()).hexdigest()==h,f
 subprocess.run(['git','init','-q','-b','heechan-rebuttal',str(s)],check=True)
 subprocess.run(['git','add','.'],cwd=s,check=True)
 subprocess.run(['git','-c','user.name=Experiment Snapshot','-c','user.email=experiment@local','commit','-q','-m','Freeze authorized fixed alpha zero SAC ablation'],cwd=s,check=True)
family='adroit' if HOST==3 else 'fetch';source=p/('alpha0-'+family+'-source')
active=[j for j in plan['jobs'] if j['status']=='running'];own={j['pid'] for j in active}
lock=Path('/workspace/.ibolt-capacity-admission.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
g=next(g for g in sorted(controller.gpu_state(),key=lambda g:(len(g['pids']),g['util'])) if not g['pids']-own and len(g['pids'])<5 and g['free']>10000)
cpus=sum(controller.cores(active)[:3],[]);assert len(cpus)>=3
env=dict(os.environ,PYTHONPATH=str(source),CUDA_VISIBLE_DEVICES=str(g['index']),JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl')
with (stage/('validation-'+family+'.log')).open('x') as log:
 proc=subprocess.Popen(['nice','-n','10','taskset','-c',','.join(map(str,cpus)),plan['python'][family],'-u',str(stage/'validate.py'),family,str(stage/('numeric-'+family))],cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
result=dict(passed=True,host=HOST,validation_pid=proc.pid,gpu=g['index'],cpus=cpus)
(stage/'validation-launch.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
'''
 result=ssh(h,code);(P/f'staged-host{h}.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(one,[3,4]))
