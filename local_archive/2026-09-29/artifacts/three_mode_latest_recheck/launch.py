import os, subprocess, sys, json
from pathlib import Path

base=Path('/home/heechan/optiq-experiments/three-mode-recheck-64x64-b32-20260921')
out=base/'results';out.mkdir(exist_ok=True)
jobs=[]
for seed in range(4):
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(seed),JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',
               OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MPLBACKEND='Agg')
    env.pop('LD_LIBRARY_PATH',None)
    command=['nice','-n','10',sys.executable,'-u',str(Path(__file__).with_name('run.py')),
             '--source',str(base/'frozen'),'--out',str(out),'--seed',str(seed),'--steps','20000']
    log=(out/f'seed{seed}.log').open('w')
    child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    jobs.append((seed,child,log))
    print(json.dumps(dict(seed=seed,pid=child.pid,command=command)),flush=True)
codes=[]
for seed,child,log in jobs:
    code=child.wait();log.close();codes.append(code)
    print(json.dumps(dict(seed=seed,exit_code=code)),flush=True)
raise SystemExit(max(codes))
