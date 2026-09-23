"""Read-only GPU numeric diagnosis: no optimizer steps or production artifacts."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from .core import Experiment,dense_score


def main():
    cfg=json.loads(Path(__file__).with_name('config.json').read_text())
    cfg.update(n=8,m=11,batch=2,hidden_dims=[16,16],density_chunk=8)
    for precision in ('default','highest'):
        with jax.default_matmul_precision(precision):
            e=Experiment(cfg,'reverse',32,0);key=jax.random.split(jax.random.PRNGKey(701),4)[3]
            a=jnp.linspace(-9.7,9.7,31)[:,None]
            zs=jnp.concatenate([jax.random.normal(jax.random.fold_in(key,i),(8,1)) for i in range(4)])
            mu,ls=e.components(e.state.params,zs)
            def collect(_,i):
                z=jax.random.normal(jax.random.fold_in(key,i),(8,1))
                m,l=e.components(e.state.params,z)
                return None,(z,m,l)
            _,(zz,mm,ll)=jax.lax.scan(collect,None,jnp.arange(4))
            mm,ll=mm.reshape(32,1),ll.reshape(32,1)
            s,lp,_=e.density_score(e.state.params,a,key,32)
            ref,ref_lp=dense_score(a,mu,ls)
            same,same_lp=dense_score(a,mm,ll)
            def maxdiff(x,y):return float(np.max(np.abs(np.asarray(x-y))))
            report=dict(precision=precision,device=str(jax.devices()[0]),z_error=maxdiff(zs,zz.reshape(32,1)),mu_error=maxdiff(mu,mm),ls_error=maxdiff(ls,ll),score_error=maxdiff(s,ref),score_same_components_error=maxdiff(s,same),actions=np.asarray(a).ravel().tolist(),reference_score=np.asarray(ref).ravel().tolist(),stream_score=np.asarray(s).ravel().tolist(),log_density=np.asarray(ref_lp).tolist())
            print(json.dumps(report),flush=True)
        jax.clear_caches()


if __name__=='__main__':main()
