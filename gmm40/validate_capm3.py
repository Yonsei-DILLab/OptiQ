"""Validate the explicit cap=-3 profile while preserving default actor settings."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from .optiq_trg import OptiQTRG


def main():
    class Target:
        def jax_log_prob(self, x):
            return -.5*jnp.square(x).sum(-1)

    plan=json.loads(Path(__file__).with_name('campaign_capm3_plan.json').read_text())
    default=OptiQTRG(Target())
    profile=OptiQTRG(Target(),**plan['trg_actor'])
    assert (default.actor.log_std_min,default.actor.log_std_max,default.actor.initial_log_std)==(-5.,-1.,-1.)
    assert (profile.actor.log_std_min,profile.actor.log_std_max,profile.actor.initial_log_std)==(-5.,-3.,-3.)
    assert profile.actor.mean_output_init_scale==default.actor.mean_output_init_scale==1.
    assert profile.n==profile.m==64 and profile.batch==256 and profile.temperature==1.
    for path,value in jax.tree_util.tree_flatten_with_path(default.state.params)[0]:
        keys=[entry.key for entry in path];actual=profile.state.params
        for key in keys:actual=actual[key]
        if keys==['log_std','bias']:np.testing.assert_array_equal(actual,-3*jnp.ones_like(actual))
        else:np.testing.assert_array_equal(actual,value)
    z=jax.random.normal(jax.random.PRNGKey(42),(1024,2))
    _,ls=profile.actor.apply({'params':profile.state.params},jnp.zeros((1024,1)),z)
    np.testing.assert_array_equal(ls,-3*jnp.ones_like(ls))
    for bad in [dict(log_std_max=-5.),dict(log_std_max=-3.,initial_log_std=-1.),
                dict(log_std_max=float('nan')),dict(initial_log_std=-6.)]:
        try:OptiQTRG(Target(),**bad)
        except ValueError:pass
        else:raise AssertionError(bad)
    assert default.updates==profile.updates==0
    print(json.dumps(dict(status='passed',optimizer_updates=0,plan=plan,
                         default_bounds=[-5,-1],profile_bounds=[-5,-3],initial_log_std=-3)))


if __name__=='__main__':main()
