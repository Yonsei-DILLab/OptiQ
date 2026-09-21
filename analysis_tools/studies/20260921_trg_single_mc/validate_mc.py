"""Analytic tests for the exact executed single-critic / K-action path."""
from pathlib import Path
import json
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'repo'))
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from common.type_aliases import RLTrainState
from optiq_dime import OptiQDIME
from optiq_dime.policy import OptiQPolicy
from optiq_dime.single_mc import update


def actor_fn(v, s, z):
    return .3*z + v['params']['mu']['bias'], jnp.full_like(z, jnp.log(.4))


def q_fn(v, s, a, rngs=None, mutable=False, train=False):
    p = v['params']['q']
    q = (p[0] + p[1]*a[:, 0] + p[2]*a[:, 0]**2 + .2*s[:, 0])[None, :, None]
    return (q, {'batch_stats': {}}) if mutable else q


def main():
    actor = TrainState.create(apply_fn=actor_fn,
        params={'mu': {'bias': jnp.array([.1])}}, tx=optax.sgd(.01))
    critic = RLTrainState.create(apply_fn=q_fn, params={'q': jnp.array([1.,2.,3.])},
        target_params={'q': jnp.array([10.,4.,5.])}, batch_stats={}, target_batch_stats={},
        tx=optax.sgd(.01))
    obs = jnp.array([[0.],[1.],[2.]])
    nxt = obs + .1
    actions = jnp.array([[-.2],[.3],[.5]])
    rewards, dones = jnp.array([1.,2.,3.]), jnp.array([0.,1.,0.])
    key = jax.random.PRNGKey(13)
    records = []
    for k in [1, 64]:
        new, metrics, next_key = update(actor, critic, obs, actions, nxt, rewards, dones, .99, key, k)
        keys = jax.random.split(key, 6)
        repeated = jnp.repeat(nxt, k, axis=0)
        sampled = OptiQPolicy.sample_action(actor, repeated, keys[1], deterministic=False)
        all_q = q_fn({'params':critic.target_params}, repeated, sampled)[0,:,0].reshape(3,k)
        expected = rewards + .99*(1-dones)*all_q.mean(1)
        def oracle(p):
            current = q_fn({'params':p}, obs, actions)[0,:,0]
            return ((current-expected)**2).mean()
        expected_grad = jax.grad(oracle)(critic.params)
        np.testing.assert_allclose(metrics['critic_loss'], oracle(critic.params), rtol=2e-6)
        np.testing.assert_allclose(new.params['q'], critic.params['q']-.01*expected_grad['q'],rtol=2e-6)
        np.testing.assert_allclose(metrics['next_q_values'], expected.mean(),rtol=2e-6)
        np.testing.assert_array_equal(next_key, keys[0])
        assert float(expected[1])==2.0
        assert float(metrics['backup_entropy_term'])==0
        a_grad = jax.grad(lambda p: update(actor.replace(params=p),critic,obs,actions,nxt,
                             rewards,dones,.99,key,k)[1]['critic_loss'])(actor.params)
        t_grad = jax.grad(lambda p: update(actor,critic.replace(target_params=p),obs,actions,nxt,
                             rewards,dones,.99,key,k)[1]['critic_loss'])(critic.target_params)
        assert all(np.all(np.asarray(x)==0) for x in jax.tree_util.tree_leaves((a_grad,t_grad)))
        polyak = OptiQDIME.soft_update(.005,new)
        np.testing.assert_allclose(polyak.target_params['q'],.995*critic.target_params['q']+.005*new.params['q'])
        if k==64:
            assert float(all_q.std(1).min())>0
            mean_only = jnp.tanh(.3*jax.random.normal(jax.random.split(keys[1])[0],(192,1))+.1)
            assert not np.allclose(sampled,mean_only)
            routed, routed_metrics, _ = OptiQDIME.update_critic(False,False,.99,actor,critic,
                obs,actions,nxt,rewards,dones,1,jnp.array([0.]),-3600,3600,0.,0.,0.,key,
                True,0,.25,'td',64)
            np.testing.assert_array_equal(routed.params['q'],new.params['q'])
            np.testing.assert_array_equal(routed_metrics['critic_loss'],metrics['critic_loss'])
        records.append({'K':k,'loss':float(metrics['critic_loss']),'analytic_update':True,
                        'stopped_actor_and_target_gradient':True,'terminal_mask':True})
    result=dict(passed=True,time=time.time(),devices=[str(x) for x in jax.devices()],tests=records)
    (ROOT/'UNIT_VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
