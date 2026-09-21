"""Validate the old numerical settings on TRG without optimizer updates."""
import json
import math
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from .optiq_trg import OptiQTRG, Proposal


def main():
    class Target:
        def jax_log_prob(self,x):return -.5*jnp.square(x).sum(-1)
    plan=json.loads(Path(__file__).with_name('campaign_oldinit_fresh_plan.json').read_text())
    default=OptiQTRG(Target())
    profile=OptiQTRG(Target(),**plan['trg_actor'])
    assert (default.actor.log_std_min,default.actor.log_std_max,default.actor.initial_log_std)==(-5.,-1.,-1.)
    assert default.teacher_std_floor==math.exp(-5)
    assert (profile.actor.log_std_min,profile.actor.log_std_max)==(-5.,1.)
    assert profile.actor.mean_output_init_scale==1. and profile.teacher_std_floor==.05
    z=jax.random.normal(jax.random.PRNGKey(42),(1024,2));obs=jnp.zeros((len(z),1))
    mu,ls=profile.actor.apply({'params':profile.state.params},obs,z)
    np.testing.assert_allclose(jnp.exp(ls),.5,rtol=1e-6)
    assert float(jax.grad(lambda x:jnp.clip(x,-5.,1.))(jnp.float32(math.log(.5))))==1.
    # Teacher-floor overrides must not change actor initialization or evaluation.
    actor_only=OptiQTRG(Target(),log_std_max=1.,initial_log_std=math.log(.5))
    for a,b in zip(jax.tree_util.tree_leaves(profile.state.params),jax.tree_util.tree_leaves(actor_only.state.params)):
        np.testing.assert_array_equal(a,b)
    for a,b in zip(profile._sample(profile.state.params,jax.random.PRNGKey(3),128),
                   actor_only._sample(actor_only.state.params,jax.random.PRNGKey(3),128)):
        np.testing.assert_array_equal(a,b)
    # Floor-active teacher: both sampling and log q must equal explicit sigma .05.
    means=jnp.array([[[0.,0.],[.3,-.4]]]);tiny=jnp.full_like(means,-5.)
    floored=Proposal(means,tiny,profile.teacher_std_floor)
    explicit=Proposal(means,jnp.full_like(means,math.log(.05)),math.exp(-5))
    key=jax.random.PRNGKey(9)
    x,_,_=floored.sample(key,32,'exact');y,_,_=explicit.sample(key,32,'exact')
    np.testing.assert_array_equal(x,y)
    np.testing.assert_array_equal(floored.log_prob(x),explicit.log_prob(x))
    assert np.isfinite(np.asarray(floored.log_prob(x))).all() and np.max(np.abs(x))<=1.
    for bad in (0.,-1.,float('nan'),float('inf')):
        try:OptiQTRG(Target(),teacher_std_floor=bad)
        except ValueError:pass
        else:raise AssertionError(bad)
    assert default.updates==profile.updates==actor_only.updates==0
    print(json.dumps(dict(status='passed',optimizer_updates=0,profile=plan['trg_actor'],
        initial_sigma=.5,initial_clip_derivative=1.,teacher_floor_sampling_density_consistent=True,
        teacher_floor_does_not_change_policy=True,default_preserved=True)))


if __name__=='__main__':main()
