"""Diagnose inference batching at the original 64-state extraction shape."""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from checkpoint_actor import forward, load_actor, locate


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoints',type=Path,required=True)
    p.add_argument('--core',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--commit',required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for seed in range(5):
        params,stats=load_actor(locate(a.checkpoints,'hopper',seed))
        def batched(obs,z):
            b,n=obs.shape[0],z.shape[0]
            oo=jnp.broadcast_to(obs[:,None],(b,n,obs.shape[1])).reshape(b*n,-1)
            zz=jnp.broadcast_to(z[None],(b,n,z.shape[1])).reshape(b*n,-1)
            m,l=forward(params,stats,oo,zz)
            return m.reshape(b,n,-1),l.reshape(b,n,-1)
        fn=jax.jit(batched)
        with np.load(a.core/'data'/f'hopper_s{seed}.npz') as f:
            for bs in (1,8,64):
                obs=f['states'][:64]
                out=[fn(jnp.asarray(obs[i:i+bs]),jnp.asarray(f['z'])) for i in range(0,len(obs),bs)]
                m=np.concatenate([np.asarray(x[0]) for x in out]);l=np.concatenate([np.asarray(x[1]) for x in out])
                row=dict(seed=seed,state_batch=bs,mu=float(abs(m-f['mu'][:64]).max()),
                         log_std=float(abs(l-f['log_std'][:64]).max()))
                rows.append(row);print(json.dumps(row),flush=True)
                (a.out/'probe.json').write_text(json.dumps(dict(source_commit=a.commit,jax=jax.__version__,rows=rows),indent=2)+'\n')


if __name__=='__main__':main()
