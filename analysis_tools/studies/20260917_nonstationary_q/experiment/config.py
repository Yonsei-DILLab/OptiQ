"""Frozen registration. Step t means the t-th actor update, starting at one."""
import json
from pathlib import Path

TAU = .25
SIZES = [(16,64),(256,16384),(1024,4096),(2048,2048)]
METHODS = {
    'gmm_learned': ('gmm', None),
    'exact_learned': ('exact', None),
    'sinkhorn_learned': ('sinkhorn', None),
    'exact_fixed05': ('exact', .5),
    'sinkhorn_fixed05': ('sinkhorn', .5),
    'exact_fixed01': ('exact', .1),
    'sinkhorn_fixed01': ('sinkhorn', .1),
}
UPDATES = 35000
PREFIX = 20000
WARMUP = 5000
GRID_N = 8193
EDGES_N = 513

def name(stage,method,n,m,seed):
    return f'{stage}_{method}_N{n}_M{m}_s{seed}'

def tasks():
    out=[]
    for stage in ['prefix','mass','split','source','replay','closed']:
        cells=[('sinkhorn_learned',16,64,s) for s in range(4)] if stage=='source' else [
            (method,n,m,s) for n,m in SIZES for method in METHODS for s in range(4)]
        for method,n,m,s in cells:
            out.append(dict(name=name(stage,method,n,m,s),stage=stage,method=method,n=n,m=m,seed=s,
                updates=PREFIX if stage=='prefix' else UPDATES,
                parent=name('prefix',method,n,m,s) if stage in ['mass','split'] else None,
                q_source=name('source','sinkhorn_learned',16,64,s) if stage=='replay' else None,
                actor_batch=256 if stage=='source' else 1))
    return out

def raw_step(t):
    return t in {1,1000,10000,19980,20000,20001,20020,21000,22000,24980,25000,25001,25020,26000,27000,29980,30000,30001,30020,35000}

def eval_step(t):
    return t%200==0 or t==1 or any(abs(t-p)<=100 and t%20==0 for p in (20000,22000,25000,27000,30000))

def write_json(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(p)
