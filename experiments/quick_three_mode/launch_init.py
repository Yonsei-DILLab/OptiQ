import os,subprocess,sys
from pathlib import Path
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
processes=[]
for gpu,(variant,seed) in enumerate([('init1',0),('init1',1),('original',0),('original',1)]):
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MPLBACKEND='Agg',PYTHONPATH=os.getcwd());env.pop('LD_LIBRARY_PATH',None)
 f=(out/f'{variant}_seed{seed}.log').open('w')
 processes.append(subprocess.Popen([sys.executable,'experiments/quick_three_mode/audit_init.py','--seed',str(seed),'--variant',variant,'--out',str(out)],env=env,stdout=f,stderr=subprocess.STDOUT))
raise SystemExit(max(p.wait() for p in processes))
