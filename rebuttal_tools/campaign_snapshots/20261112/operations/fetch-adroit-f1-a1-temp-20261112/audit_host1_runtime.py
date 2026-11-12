from pathlib import Path
import subprocess,json,concurrent.futures
P=Path(__file__).resolve().parent
code=r'''
from pathlib import Path
import hashlib,json,importlib.metadata as m,numpy as np
import gymnasium as gym,gymnasium_robotics as robotics
gym.register_envs(robotics);p=Path(robotics.__file__).parent
manifest={str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(p.rglob('*')) if f.is_file() and '__pycache__' not in f.parts and f.suffix not in ['.pyc','.pyo']}
rollouts={}
for name in ['FetchReach-v4','FetchPush-v4','FetchPickAndPlace-v4','FetchSlide-v4']:
 e=gym.make(name,reward_type='sparse');o,_=e.reset(seed=123);h=hashlib.sha256()
 for i in range(20):
  for key in sorted(o):h.update(key.encode());h.update(np.asarray(o[key]).tobytes())
  o,r,te,tr,info=e.step(np.zeros(e.action_space.shape,dtype=e.action_space.dtype));h.update(np.asarray([r,te,tr,info['is_success']],dtype=np.float64).tobytes())
 e.close();rollouts[name]=h.hexdigest()
print(json.dumps(dict(module=str(p),files=manifest,rollouts=rollouts,packages={n:m.version(n) for n in ['gymnasium','gymnasium-robotics','mujoco','jax','torch','stable-baselines3','flax']},adroit_v2='AdroitHandDoorSparse-v2' in gym.envs.registry)))
'''
def one(h,py):
 r=subprocess.run(['ssh','-T','-o','BatchMode=yes',f'ibolt-server-{h}',py+' -'],input=code,text=True,capture_output=True,timeout=90);assert not r.returncode,r.stderr;d=json.loads(r.stdout);(P/f'environment-host{h}.json').write_text(json.dumps(d,indent=2));return d
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as e:
 fs=[e.submit(one,1,'/workspace/heechan-unified-f1-a1-20261112/runtime/bin/python'),e.submit(one,4,'/workspace/heechan-fetch-sigma/venv-sac-cu128-20261110/bin/python')];a,b=[f.result() for f in fs]
print('PACKAGES_EQUAL',a['packages']==b['packages']);print('ROLLOUTS_EQUAL',a['rollouts']==b['rollouts']);print('SOURCE_FILES_EQUAL',a['files']==b['files']);print('FILE_DIFF',[k for k in set(a['files'])|set(b['files']) if a['files'].get(k)!=b['files'].get(k)]);assert a['adroit_v2']==b['adroit_v2'] and a['packages']==b['packages'] and a['rollouts']==b['rollouts'] and a['files']==b['files']
