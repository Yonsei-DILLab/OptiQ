"""Check the GMM40-only sigma/init changes without training a learner."""
import json
import math

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import kstest

from .optiq_trg import Actor, OptiQTRG, sample_box, SOURCE
import importlib

box = importlib.import_module(SOURCE + '.box_gaussian')


def main():
    class Target:
        def jax_log_prob(self, x):
            return -.5*jnp.square(x).sum(-1)

    agent = OptiQTRG(Target(), seed=0)
    assert agent.actor.log_std_min == -5
    assert agent.actor.log_std_max == 1
    assert agent.actor.initial_log_std == 1
    assert agent.actor.mean_output_init_scale == 1
    assert agent.n == agent.m == 64 and agent.batch == 256
    assert agent.temperature == 1 and agent.updates == 0
    _, key = jax.random.split(jax.random.PRNGKey(0))
    obs, z = jnp.zeros((1,1)), jnp.zeros((1,2))
    old = Actor(2,(256,256),-5.,-1.,-1.)
    before = old.init(key,obs,z)['params']
    after = agent.state.params
    for path, value in jax.tree_util.tree_flatten_with_path(before)[0]:
        keys = [entry.key for entry in path]
        actual = after
        for k in keys:
            actual = actual[k]
        if keys == ['mu','kernel']:
            np.testing.assert_allclose(actual,100*value,rtol=2e-6,atol=1e-7)
        elif keys == ['log_std','bias']:
            np.testing.assert_array_equal(actual,jnp.ones_like(actual))
        else:
            np.testing.assert_array_equal(actual,value)
    params = {**after, 'log_std': {**after['log_std'], 'bias': jnp.zeros(2)}}
    output = agent.actor.apply({'params':params},obs,z)[1]
    np.testing.assert_array_equal(output,jnp.zeros((1,2)))
    def scale_sum(bias):
        p = {**params,'log_std':{**params['log_std'],'bias':bias}}
        return agent.actor.apply({'params':p},obs,z)[1].sum()
    np.testing.assert_array_equal(jax.grad(scale_sum)(jnp.zeros(2)),jnp.ones(2))
    cases = []
    for center in [-1.,0.,1.]:
        for log_sigma in [-5.,-1.,0.,1.]:
            sigma = math.exp(log_sigma)
            lo,hi = (-1-center)/sigma,(1-center)/sigma
            mass = ndtr(hi)-ndtr(lo)
            points = np.linspace(-1,1,51,dtype=np.float32)
            mu = jnp.array([[[center]]]);ls = jnp.array([[[log_sigma]]])
            actual = box.component_log_prob(jnp.asarray(points)[None,:,None],mu,ls)[0,0]
            expected = -.5*((points.astype(float)-center)/sigma)**2-log_sigma-.5*math.log(2*math.pi)-math.log(mass)
            np.testing.assert_allclose(actual,expected,rtol=4e-6,atol=5e-5)
            normalization = quad(lambda x:math.exp(-.5*((x-center)/sigma)**2-log_sigma-.5*math.log(2*math.pi)-math.log(mass)),
                                 -1,1,epsabs=1e-10,points=[center])[0]
            assert abs(normalization-1)<1e-8
            def loss(log_std):
                return box.component_log_prob(jnp.array([[[center]]]),mu,log_std.reshape(1,1,1)).sum()
            assert np.isfinite(jax.grad(loss)(jnp.array(log_sigma)))
            samples = np.asarray(sample_box(jax.random.PRNGKey(103),jnp.full((8192,1),center),
                                           jnp.full((8192,1),log_sigma)))[:,0]
            assert np.isfinite(samples).all() and np.max(np.abs(samples))<=1
            pit = (ndtr((samples.astype(float)-center)/sigma)-ndtr(lo))/mass
            ks = float(kstest(pit,'uniform').statistic)
            assert ks<.02,(center,log_sigma,ks)
            cases.append(dict(center=center,log_sigma=log_sigma,normalization=normalization,ks=ks))
    print(json.dumps(dict(status='passed',optimizer_updates=agent.updates,
        initial_parameter_differences=['mean-head kernel multiplied by 100 (variance scale 1e-4 -> 1)',
            'log_std bias initialized at the new upper bound +1'],
        gradient_above_old_cap=[1.,1.],log_sigma_bounds=[-5.,1.],initial_log_sigma=1.,
        teacher_floor=math.exp(-5),cases=cases),indent=2))


if __name__ == '__main__':
    main()
