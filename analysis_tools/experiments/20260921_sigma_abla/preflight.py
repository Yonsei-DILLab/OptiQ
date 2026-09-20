"""Check all resolved configs and actual initialized sigma before training."""
import copy
import json
import sys
import numpy as np
import jax
import jax.numpy as jnp
import train_sigma as abla
from omegaconf import OmegaConf
from optiq_dime.policy import SemiImplicitActor

rows=[]
for task in ('ant','walker2d','humanoid','hopper','halfcheetah'):
    reference=None
    for sigma in abla.SIGMAS:
        sys.argv=['preflight',str(sigma)]
        cfg=abla.compose_config([f'benchmark={task}'])
        algorithm=copy.deepcopy(OmegaConf.to_container(cfg.alg,resolve=True))
        algorithm['actor'].pop('initial_log_std')
        if reference is None:reference=algorithm
        assert algorithm==reference, 'Only initial scale may differ'
        rows.append(dict(task=task,sigma=sigma,log_sigma=cfg.alg.actor.initial_log_std))
for sigma in abla.SIGMAS:
    actor=SemiImplicitActor(action_dim=8,hidden_dims=(256,256),log_std_min=-5,
        log_std_max=-1,initial_log_std=float(np.log(sigma)))
    obs=jnp.ones((4,27));z=jax.random.normal(jax.random.PRNGKey(0),(4,8))
    params=actor.init(jax.random.PRNGKey(1),obs,z)
    mu,log_std=actor.apply(params,obs,z)
    np.testing.assert_allclose(np.exp(np.asarray(log_std)),sigma,rtol=1e-6)
assert '20260920_truncated_mll' in sys.modules['optiq_dime.policy'].__file__
print(json.dumps(dict(passed=True,configs=rows,implementation=sys.modules['optiq_dime.policy'].__file__)))
