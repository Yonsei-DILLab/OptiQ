"""Compare mathematically equivalent BN arithmetic, not different policies."""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from checkpoint_actor import load_actor,locate


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoints',type=Path,required=True);p.add_argument('--core',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);rows=[]
    jax.config.update('jax_enable_x64',True)
    for seed in range(5):
        params,stats=load_actor(locate(a.checkpoints,'hopper',seed))
        with np.load(a.core/'data'/f'hopper_s{seed}.npz') as f:
            obs=f['states'][:64];z=f['z']
            oo=np.broadcast_to(obs[:,None],(64,len(z),obs.shape[1])).reshape(64*len(z),-1).copy()
            zz=np.broadcast_to(z[None],(64,len(z),z.shape[1])).reshape(64*len(z),-1).copy()
            for variant in ('left','combined_scale','division','affine','fp64'):
                dtype=jnp.float64 if variant=='fp64' else jnp.float32
                pp=jax.tree_util.tree_map(lambda v:jnp.asarray(v,dtype),params)
                ss=jax.tree_util.tree_map(lambda v:jnp.asarray(v,dtype),stats)
                def norm(x,name):
                    par,stat=pp[name],ss[name];eps=jnp.asarray(.001,dtype)
                    if variant=='combined_scale':return (x-stat['mean'])*(jax.lax.rsqrt(stat['var']+eps)*par['scale'])+par['bias']
                    if variant=='division':return (x-stat['mean'])/jnp.sqrt(stat['var']+eps)*par['scale']+par['bias']
                    if variant=='affine':
                        mul=jax.lax.rsqrt(stat['var']+eps)*par['scale']
                        return x*mul+(par['bias']-stat['mean']*mul)
                    return (x-stat['mean'])*jax.lax.rsqrt(stat['var']+eps)*par['scale']+par['bias']
                def fn(o,lat):
                    x=norm(jnp.concatenate([o,lat],-1),'BatchRenorm_0')
                    for i in range(2):
                        d=pp[f'Dense_{i}'];x=norm(jax.nn.gelu(x@d['kernel']+d['bias']),f'BatchRenorm_{i+1}')
                    return jnp.tanh(x@pp['mu']['kernel']+pp['mu']['bias']),jnp.clip(x@pp['log_std']['kernel']+pp['log_std']['bias'],-5, -1)
                m,l=jax.jit(fn)(jnp.asarray(oo,dtype),jnp.asarray(zz,dtype))
                dm=np.asarray(m).reshape(64,len(z),-1)-f['mu'][:64]
                dl=np.asarray(l).reshape(64,len(z),-1)-f['log_std'][:64]
                row=dict(seed=seed,variant=variant,mu_max=float(abs(dm).max()),ls_max=float(abs(dl).max()),
                         mu_rms=float(np.sqrt((dm*dm).mean())),ls_rms=float(np.sqrt((dl*dl).mean())))
                rows.append(row);print(json.dumps(row),flush=True)
                (a.out/'probe.json').write_text(json.dumps(dict(source_commit=a.commit,rows=rows),indent=2)+'\n')


if __name__=='__main__':main()
