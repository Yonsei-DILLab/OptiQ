"""Pre-registered multidimensional Q tracking; no result-dependent selection."""
from pathlib import Path
import json
TAU=.25
DIMS=[1,2,4,8]
SIZES=[(256,1024),(512,512)]
EPSILONS=[1e-4,1e-3,1e-2,1e-1,1.,10.]
METHODS={'gmm_learned':('gmm',None,None),'exact_learned':('exact',None,None),
         'exact_fixed05':('exact',.5,None),'exact_fixed01':('exact',.1,None),
         'sinkhorn_fixed05':('sinkhorn',.5,.1),'sinkhorn_fixed01':('sinkhorn',.1,.1),
         'argmax_truncated':('legacy',None,.05)}
for eps in EPSILONS:METHODS['sinkhorn_e'+format(eps,'g')]=('sinkhorn',None,eps)
UPDATES=35000
PREFIX=20000
WARMUP=5000
GRID_N=8193
EDGES_N=513
SAMPLES=32768

def name(stage,family,dim,method,n,m,seed):return f'{family}_{stage}_D{dim}_{method}_N{n}_M{m}_s{seed}'
def tasks():
 out=[]
 for n,m in SIZES:
  for dim in DIMS:
   for method in METHODS:
    for seed in range(4):
     for family,stages in [('double',['prefix','mass']),('tri',['prefix','mass','split','replay','closed'])]:
      for stage in stages:
       parent=name('prefix',family,dim,method,n,m,seed) if stage in ['mass','split'] else None
       source=name('source','tri',dim,'sinkhorn_e0.1',16,64,seed) if stage=='replay' else None
       # First get small-N analytic results in every dimension, then larger matrices.
       priority=SIZES.index((n,m))*100 + (0 if family=='double' else 10) + {'prefix':0,'mass':1,'split':2,'replay':30,'closed':31}[stage]
       out.append(dict(name=name(stage,family,dim,method,n,m,seed),stage=stage,family=family,dim=dim,
        method=method,n=n,m=m,seed=seed,updates=PREFIX if stage=='prefix' else UPDATES,
        parent=parent,q_source=source,actor_batch=1,priority=priority))
 return sorted(out,key=lambda t:(t['priority'],DIMS.index(t['dim']),t['seed'],list(METHODS).index(t['method'])))

def raw_step(t):return t in {0,1,1000,10000,19980,20000,20001,20020,20200,21000,22000,24980,25000,25001,25020,25200,26000,27000,29980,30000,30001,30020,30200,35000}
def eval_step(t):return t%200==0 or t==1 or any(abs(t-p)<=100 and t%20==0 for p in (20000,22000,25000,27000,30000))
def write_json(path,obj):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
 tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)
