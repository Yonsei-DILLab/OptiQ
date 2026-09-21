import copy
import json
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
from train_cap import compose_config
from optiq_dime.policy import SemiImplicitActor

for task in ('ant','humanoid'):
    reference=None
    for cap in (0.,-2.):
        cfg=compose_config(cap,[f'benchmark={task}'])
        alg=copy.deepcopy(OmegaConf.to_container(cfg.alg,resolve=True))
        alg['actor'].pop('log_std_max')
        if reference is None:reference=alg
        assert alg==reference
        print(json.dumps(dict(task=task,cap=cap,initial_sigma=.1,total_steps=cfg.total_steps)))
predictions=[]
for cap in (0.,-2.):
    actor=SemiImplicitActor(action_dim=8,hidden_dims=(256,256),
        log_std_min=-5.,log_std_max=cap,initial_log_std=float(np.log(.1)))
    obs=jnp.ones((4,27));z=jax.random.normal(jax.random.PRNGKey(0),(4,8))
    p=actor.init(jax.random.PRNGKey(1),obs,z)
    mu,logs=actor.apply(p,obs,z)
    np.testing.assert_allclose(np.exp(np.asarray(logs)),.1,rtol=1e-6)
    predictions.append(np.asarray(mu))
    p['params']['log_std']['bias']=jnp.full((8,),.5)
    _,logs=actor.apply(p,obs,z)
    np.testing.assert_allclose(logs,cap)
np.testing.assert_array_equal(*predictions)
print('PASS: only cap differs; matching initial means and sigma; both bounds effective')
