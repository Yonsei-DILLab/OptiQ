from pathlib import Path
import json,time,subprocess,hashlib
base=Path('/home/heechan/optiq-experiments')
campaigns=['pointmaze-dipo-upstream-u32-s0to2-1m-20260926','maze-dipo-upstream-u32-s0-1m-20260926','pointmaze-baselines-s1to3-1m-20260926','pointmaze-baselines-idle-backfill-20260926','pointmaze-simple-meow-alpha05-1m-20260926']
out={'time':time.time(),'campaigns':{}}
for c in campaigns:
 r=base/c
 if not r.exists():continue
 d={'root':str(r),'metadata':{},'progress':{},'jobs':{},'proofs':{},'transfers':{},'failures':{}}
 for f in r.glob('*.json'):
  if 'failure' in f.name:d['failures'][f.name]=json.loads(f.read_text())
  elif 'transfer' in f.name:d['transfers'][f.name]=json.loads(f.read_text())
  elif f.name in ['queue.json','manifest.json','registration.json','result.json','status.json']:d['metadata'][f.name]=json.loads(f.read_text())
 for f in (r/'jobs').glob('*.json'):d['jobs'][f.stem]=json.loads(f.read_text())
 for f in (r/'proofs').glob('*-runs.json'):d['proofs'][f.stem]=json.loads(f.read_text())
 for f in (r/'runs').glob('*/progress.json'):d['progress'][f.parent.name]=json.loads(f.read_text())
 for f in (r/'runs').glob('*/failure.json'):d['failures'][str(f.relative_to(r))]=json.loads(f.read_text())
 out['campaigns'][c]=d
ps=subprocess.run(['ps','-eo','pid,etimes,args'],capture_output=True,text=True).stdout
out['learners']=[x for x in ps.splitlines() if 'python' in x and '-m maze_benchmarks.run ' in x]
sv=subprocess.run(['supervisorctl','-c','/home/heechan/OptiQ-ops/supervisor/supervisord.conf','status'],capture_output=True,text=True)
out['supervisor_returncode']=sv.returncode
out['supervisor_stderr']=sv.stderr
out['supervisors']=[x for x in sv.stdout.splitlines() if any(c in x for c in campaigns)]
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],capture_output=True,text=True)
out['gpu_processes']=gpu.stdout
out['guide_sha256']=hashlib.sha256(Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()
print(json.dumps(out))

