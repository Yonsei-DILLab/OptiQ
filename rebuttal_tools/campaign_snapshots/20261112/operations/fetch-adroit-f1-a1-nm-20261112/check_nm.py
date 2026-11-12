import os,sys,json,importlib.util
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp
from gymnasium import spaces
from train import compose_config
from ibolt.policy import IBOLTPolicy
from ibolt.algorithm import IBOLT
source=Path(os.environ['PYTHONPATH']); family='adroit' if 'adroit' in source.name else 'fetch'
oldpath=source.parent/('entropy-'+family+'-source/ibolt/algorithm.py')
spec=importlib.util.spec_from_file_location('ibolt.reference_nm_check',oldpath);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
cfg=compose_config(['wandb.activate=false']+(['+alg.critic.layer_norm=true'] if family=='adroit' else []))
if family=='fetch':cfg.alg.actor.log_std_max=1.
dim=30 if family=='adroit' else 4;obsdim=39 if family=='adroit' else 28
policy=IBOLTPolicy(spaces.Box(-np.inf,np.inf,(obsdim,),dtype=np.float32),spaces.Box(-1.,1.,(dim,),dtype=np.float32),cfg=cfg)
policy.build(jax.random.PRNGKey(5),lambda _:3e-4,3e-4)
obs=jax.random.normal(jax.random.PRNGKey(9),(8,obsdim));out=[]
for n,m in [(64,64),(64,256),(16,64)]:
 args=(policy.actor_state,policy.qf_state,obs,jax.random.PRNGKey(61),n,m//n,.1,1.)
 a,metrics,k=IBOLT.update_actor(*args);b,refmetrics,rk=old.IBOLT.update_actor(*args)
 leaves=jax.tree_util.tree_leaves(a);ref=jax.tree_util.tree_leaves(b)
 assert all(np.isfinite(np.asarray(x)).all() for x in leaves)
 diffs=[float(np.max(np.abs(np.asarray(x)-np.asarray(y)))) for x,y in zip(leaves,ref) if np.asarray(x).size]
 for x,y in zip(leaves,ref):np.testing.assert_allclose(x,y,rtol=1e-6,atol=1e-7)
 np.testing.assert_array_equal(k,rk)
 for name,value in refmetrics.items():np.testing.assert_allclose(metrics[name],value,rtol=1e-6,atol=1e-7)
 result={k:float(v) for k,v in metrics.items()}
 assert all(np.isfinite(list(result.values())))
 assert 1.-1e-5<=result['importance_ess_mean']<=m+1e-5
 out.append(dict(N=n,M=m,max_state_difference=max(diffs),metrics=result))
assert jax.default_backend()=='gpu'
print(json.dumps(dict(passed=True,family=family,backend=jax.default_backend(),comparisons=out)))
