"""CPU-only evaluation of new checkpoints; never launches/restarts training."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CAMPAIGN='antmaze-matched-start-eval-20260924-r2'
OLD='antmaze-optiq-progress100-2x2-T1-s0-20260924'
NEW='antmaze-optiq-geodesic-no-step-B0-T1-s0-20260924'
OPS=Path('/home/heechan/OptiQ-ops')
BASE=Path('/home/heechan/optiq-experiments')
HOST_JOBS={
 'vast-heechan-180':[(NEW,f'{task}-optiq-geodesic-no-step-B0-T1-s0') for task in ('v3','v4')],
 'vast-heechan-199':[(NEW,'v2-optiq-geodesic-no-step-B0-T1-s0')]+
     [(OLD,f'{task}-optiq-progress100_geodesic_no_bonus-T1-s0') for task in ('v3','v4')],
 'vast1':[(OLD,'v2-optiq-progress100_geodesic-T1-s0'),(OLD,'v2-optiq-progress100_geodesic_no_bonus-T1-s0')],
}


def read(path):return json.loads(path.read_text())
def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
 tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def candidates(run):
 found=[]
 for proof in run.glob('policy-checkpoints/*/verification.json'):
  data=read(proof);checkpoint=proof.parent/'policy.pt'
  if checkpoint.exists() and data['readback_verified']:found.append((data['step'],checkpoint))
 if (run/'result.json').exists() and (run/'checkpoint-verification.json').exists():
  proof=read(run/'checkpoint-verification.json')
  if proof['readback_verified']:found.append((proof['steps'],run/'checkpoint-final.pt'))
 return sorted(found)


def register(host):
 source=Path(__file__).resolve().parents[1]
 sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
 assert source==OPS/'sources'/sha
 root=BASE/CAMPAIGN;assert not root.exists();root.mkdir();(root/'logs').mkdir()
 jobs=[]
 for campaign,key in HOST_JOBS[host]:
  run=BASE/campaign/'runs'/key;cfg=read(run/'config.json')
  assert cfg['task']!='v1' and cfg['method']=='optiq'
  available=candidates(run);floor=available[-1][0] if available else 0
  jobs.append(dict(id=key,run=str(run),source_commit=cfg['source_commit'],floor_step=floor,task=cfg['task']))
  write(run/'evaluation-policy-20260924.json',dict(training_source=cfg['source_commit'],evaluation_source=sha,
      change='Primary evaluation now matches training: original fixed origin/full state for v2-v4; v1 remains random',
      primary_evaluation_root=str(root/'runs'/key),initial_fixed_checkpoint_step=floor,
      original_running_evaluator='Frozen process retains prior random-start outputs as supplementary; no training restart or mutation',
      v1_unchanged=True,time=time.time()))
 write(root/'manifest.json',dict(evaluation_source=sha,source=str(source),host=host,jobs=jobs,
       max_parallel=2,episodes_intermediate=40,episodes_final=100,training_unchanged=True,
       failure_holds_pending=True,automatic_restart=False,project='OptiQ/antmaze'))
 conf=OPS/'supervisor/jobs'/(CAMPAIGN+'.conf');assert not conf.exists()
 conf.write_text(f'''[program:{CAMPAIGN}]
command={sys.executable} -m antmaze_experiments.watch_fixed_evaluations --root {root}
directory={source}
environment=PYTHONDONTWRITEBYTECODE="1"
autostart=false
autorestart=false
startsecs=2
startretries=0
stopasgroup=true
killasgroup=true
stopwaitsecs=20
stdout_logfile={root}/controller.log
stderr_logfile={root}/controller.err
stdout_logfile_maxbytes=0
stderr_logfile_maxbytes=0
''')
 ctl=['/usr/local/bin/supervisorctl','-c',str(OPS/'supervisor/supervisord.conf')]
 for args in (['reread'],['update',CAMPAIGN],['start',CAMPAIGN]):subprocess.run(ctl+args,check=True)
 write(root/'registration.json',dict(time=time.time(),service=CAMPAIGN,manifest=read(root/'manifest.json')))
 print(json.dumps(dict(root=str(root),jobs=[j['id'] for j in jobs],evaluation_source=sha)))


def watch(root):
 import fcntl
 lock=(root/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (root/'status.json').exists(),'Do not automatically restart evaluation controller'
 manifest=read(root/'manifest.json');source=Path(manifest['source'])
 live={};attempted=set();completed=[];failed=[]
 while True:
  for lane,(proc,item,handle) in list(live.items()):
   code=proc.poll()
   if code is None:continue
   handle.close();del live[lane]
   result=Path(item['output'])/'result.json'
   verification=Path(item['output'])/'verification.json'
   if code==0 and result.exists() and verification.exists() and read(verification)['passed']:
    completed.append(dict(**item,completed_at=time.time()))
   else:
    failed.append(dict(**item,returncode=code));write(root/'failure.json',dict(failed=failed,pending_held=True))
  pending=[];finished_training=True
  for job in manifest['jobs']:
   run=Path(job['run']);jobstate=run.parent.parent/'jobs'/(job['id']+'.json')
   state=read(jobstate).get('status') if jobstate.exists() else None
   if not (run/'result.json').exists() and state not in ('failed','interrupted','cancelled'):finished_training=False
   for step,checkpoint in candidates(run):
    token=(job['id'],step)
    if step<job['floor_step'] or token in attempted:continue
    output=root/'runs'/job['id']/f'step_{step:010d}'
    pending.append(dict(job=job['id'],run=str(run),step=step,checkpoint=str(checkpoint),output=str(output),
         episodes=100 if checkpoint.name=='checkpoint-final.pt' else 40))
  if not failed:
   for lane in range(manifest['max_parallel']):
    if lane in live or not pending:continue
    item=pending.pop(0);attempted.add((item['job'],item['step']))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
       OPENBLAS_NUM_THREADS='1',XLA_PYTHON_CLIENT_PREALLOCATE='false',PYTHONUNBUFFERED='1',
       PYTHONDONTWRITEBYTECODE='1',PYTHONWARNINGS='ignore',MPLBACKEND='Agg',WANDB_MODE='disabled',
       USE_FLAX='0',USE_TORCH='1',D4RL_SUPPRESS_IMPORT_ERROR='1',MUJOCO_PY_FORCE_CPU='1',
       MUJOCO_PY_MUJOCO_PATH='/home/heechan/.mujoco/mujoco210',LD_LIBRARY_PATH='/home/heechan/.mujoco/mujoco210/bin')
    cmd=['/usr/bin/nice','-n','10','/home/heechan/.venv-ddiffpg-native/bin/python','-u',
         str(source/'antmaze_experiments/reevaluate_fixed_origin.py'), '--run',item['run'],
         '--checkpoint',item['checkpoint'],'--output',item['output'],'--episodes',str(item['episodes']),
         '--cpu-offset',str(4*lane)]
    log=(root/'logs'/f"{item['job']}-{item['step']}.log").open('x')
    proc=subprocess.Popen(cmd,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
    live[lane]=(proc,item,log)
  write(root/'status.json',dict(time=time.time(),controller_pid=os.getpid(),evaluation_source=manifest['evaluation_source'],
      running=[dict(**item,pid=proc.pid,lane=lane) for lane,(proc,item,_) in live.items()],pending=pending,
      completed=completed,failed=failed,pending_held=bool(failed),training_finished=finished_training))
  if failed and not live:return 1
  if finished_training and not live and not pending:
   write(root/'result.json',dict(completed=True,results=completed,evaluation_source=manifest['evaluation_source']));return 0
  time.sleep(10)


if __name__=='__main__':
 parser=argparse.ArgumentParser(allow_abbrev=False)
 parser.add_argument('--register',choices=list(HOST_JOBS))
 parser.add_argument('--root',type=Path)
 args=parser.parse_args();assert bool(args.register)!=bool(args.root)
 raise SystemExit(register(args.register) if args.register else watch(args.root))
