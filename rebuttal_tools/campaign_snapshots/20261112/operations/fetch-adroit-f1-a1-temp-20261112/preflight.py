
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
