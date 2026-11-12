from pathlib import Path
import subprocess, json, time
P=Path(__file__).resolve().parent
R='/workspace/heechan-unified-f1-a1-20261112'
code=r'''
from pathlib import Path
import os,json,subprocess,time,hashlib
p=Path('/workspace/heechan-unified-f1-a1-20261112');plan=json.loads((p/'plan.json').read_text())
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl',CUDA_VISIBLE_DEVICES='0',JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false')
reports={}
for family in ['fetch','adroit']:
 env['PYTHONPATH']=str(p/(family+'-source'))
 test="import json,jax,jax.numpy as jnp,importlib.metadata as m; x=jax.jit(lambda a:a@a)(jnp.ones((32,32))); assert bool(jnp.isfinite(x).all()); assert jax.default_backend()=='gpu'; print(json.dumps(dict(backend=jax.default_backend(),devices=[str(d) for d in jax.devices()],packages={n:m.version(n) for n in ['jax','jaxlib','mujoco','gymnasium','gymnasium-robotics','torch','flax','stable-baselines3']})))"
 r=subprocess.run([plan['python'][family],'-c',test],env=env,capture_output=True,text=True,timeout=90)
 assert r.returncode==0,r.stdout+r.stderr
 reports[family]=json.loads(r.stdout.strip().splitlines()[-1])
 print(family,'GPU validation passed',flush=True)
env['PYTHONPATH']=str(p/'fetch-slide-20cm-source')
r=subprocess.run([plan['python']['fetch'],str(p/'slide-20cm-20261112/validate_environment.py')],env=env,capture_output=True,text=True,timeout=180)
(p/'preflights/fetch-environment.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stdout+r.stderr
proof=json.loads((p/'slide-20cm-20261112/environment-verification.json').read_text());assert proof['passed'];assert proof['physical_rollout_sha256']=='0e5b3b8000328a814451ea6d7b84342bf34cd0df23b0b65aef00ceaaedce1887'
print('Fetch environment, physics, vec256, HER and evaluation passed',flush=True)
env['PYTHONPATH']=str(p/'adroit-source')
demo_code="from pathlib import Path; import json; from adroit_rebuttal.verified_demos import resolve; p=Path("+repr(str(p))+"); plan=json.loads((p/'plan.json').read_text()); reports={};\nfor task,spec in plan['datasets'].items():\n path,proof=resolve(task,spec); reports[task]=proof; print(task,proof['successful_trajectories'],flush=True)\n(p/'preflights/adroit-replay.json').write_text(json.dumps(reports,indent=2))"
r=subprocess.run([plan['python']['adroit'],'-c',demo_code],env=env,capture_output=True,text=True,timeout=180)
(p/'preflights/adroit-replay.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stdout+r.stderr
demonstrations=json.loads((p/'preflights/adroit-replay.json').read_text());assert len(demonstrations)==4 and all(d['passed'] and d['replayed'] and d['successful_trajectories']==25 for d in demonstrations.values())
for source,digest in plan['source_manifests'].items():
 folder=p/(source+'-source');m=folder/'source-sha256.json';assert hashlib.sha256(m.read_bytes()).hexdigest()==digest
 for name,h in json.loads(m.read_text()).items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==h,name
report=dict(passed=True,at=time.time(),host=2,endpoint='heechan@114.32.64.6:41188',runtime=reports,source_manifests=plan['source_manifests'],fetch_environment=proof,adroit_demonstrations={t:dict(passed=d['passed'],trajectories=d['successful_trajectories'],dataset_sha256=d['dataset_sha256'],max_observation_error=max(e['max_observation_error'] for e in d['episodes'])) for t,d in demonstrations.items()})
(p/'preflight.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
'''
r=subprocess.run(['ssh','-T','-o','BatchMode=yes','ibolt-server-2','python3 -'],input=code,text=True,capture_output=True,timeout=600)
(P/'validation.log').write_text(r.stdout+r.stderr)
print(r.stdout,flush=True)
assert r.returncode==0,r.stdout+r.stderr
(P/'preflight.json').write_text(r.stdout.strip().splitlines()[-1])
