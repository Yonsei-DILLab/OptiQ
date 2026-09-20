import os,subprocess,sys
from pathlib import Path
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
ps=[]
for gpu,(mode,seed) in enumerate([('fixed',0),('fixed',1),('fresh',0),('fresh',1)]):
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MPLBACKEND='Agg',PYTHONPATH=os.getcwd());env.pop('LD_LIBRARY_PATH',None)
 log=(out/f'{mode}_seed{seed}.log').open('w')
 ps.append(subprocess.Popen([sys.executable,'experiments/gmm40_fixed_fresh/run.py','--seed',str(seed),'--mode',mode,'--out',str(out)],env=env,stdout=log,stderr=subprocess.STDOUT))
raise SystemExit(max(p.wait() for p in ps))
