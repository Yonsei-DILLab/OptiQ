"""Verify the actual 64x64 online path, density parity and stop gradients."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import SOURCE, compose_config, runner
sys.path.insert(0, str(SOURCE/'tests'))

import copy
import inspect
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from stable_baselines3.common.logger import configure
from optiq_dime.algorithm import OptiQDIME
from optiq_dime.distillation import direct_gmm_nll
from optiq_dime.semi_implicit import ConditionalGaussianProposal, conditional_mixture_log_prob
import optiq_dime.algorithm as algorithm
from test_semi_implicit import actor_state, critic_state


def test_density_and_direct_mll64_update_without_ot(monkeypatch):
    monkeypatch.setattr(algorithm, 'sinkhorn', lambda *a, **k: pytest.fail('OT must not be invoked'))
    actor, critic = actor_state(), critic_state()
    obs, key = jnp.ones((4, 3)), jax.random.PRNGKey(819)
    floor = np.exp(-5.)
    result = OptiQDIME.update_actor(actor,critic,obs,key,jnp.array([-3600.]),
        64,1,'exact',floor,.5,False,True,1.,False,16.,257,.25,.1,100,'mean','argmax',
        True,False,'direct_gmm_nll','conditional_mixture',0.,False,'mean')
    _, lk, pk, _ = jax.random.split(key,4)
    zk, _ = jax.random.split(lk)
    z = jax.random.normal(zk,(4,64,2))
    obsrep = jnp.repeat(obs,64,axis=0)
    mu, ls = actor.apply_fn({'params':actor.params},obsrep,z.reshape(-1,2))
    mu, ls = mu.reshape(4,64,2), ls.reshape(4,64,2)
    proposal = ConditionalGaussianProposal(mu,ls,floor)
    actions,u,indices = proposal.sample(pk,1,'exact')
    assert actions.shape == u.shape == (4,64,2) and indices.shape == (4,64)
    np.testing.assert_array_equal(proposal.effective_log_std(),ls)
    lp = conditional_mixture_log_prob(u,mu,ls)
    np.testing.assert_allclose(lp,proposal.log_prob(u),atol=1e-6)
    w = jax.nn.softmax(-lp,axis=-1)  # The fixture critic is constant.
    def loss(params):
        m,l = actor.apply_fn({'params':params},obsrep,z.reshape(-1,2))
        return direct_gmm_nll(m.reshape(4,64,2),l.reshape(4,64,2),u,w)[0]
    expected, grads = jax.value_and_grad(loss)(actor.params)
    expected_actor = actor.apply_gradients(grads=grads)
    np.testing.assert_allclose(result[1],expected,rtol=2e-5)
    for a,b in zip(jax.tree_util.tree_leaves(result[0].params),jax.tree_util.tree_leaves(expected_actor.params)):
        np.testing.assert_allclose(a,b,atol=2e-6)
    teacher_grads = jax.grad(lambda t,v: direct_gmm_nll(mu,ls,t,v)[0],argnums=(0,1))(u,w)
    assert all(np.count_nonzero(g) == 0 for g in teacher_grads)
    assert not any(k.startswith('ot_') for k in result[3])
    assert all(np.isfinite(v).all() for v in result[3].values())


@pytest.mark.parametrize('task',['ant','halfcheetah','hopper','walker2d','humanoid'])
def test_real_environment_mll64_both_heads_and_plain_td(task,tmp_path,monkeypatch):
    cfg = compose_config([f'benchmark={task}',f'output_root={tmp_path}',
        'alg.batch_size=4','alg.buffer_size=32','alg.learning_starts=2',
        'alg.actor.learning_starts=2','num_eval_episodes=1','eval_interval=4',
        'diagnostic_interval=4','checkpoint_interval=4'])
    calls=[]
    original=OptiQDIME.update_actor
    signature=inspect.signature(original)
    def record(*args,**kwargs):
        bound=signature.bind(*args,**kwargs).arguments
        calls.append((bound['num_policy_samples'],bound['proposals_per_policy_sample'],bound['distillation_loss']))
        return original(*args,**kwargs)
    OptiQDIME._train.clear_cache()
    monkeypatch.setattr(OptiQDIME,'update_actor',staticmethod(record))
    monkeypatch.setattr(algorithm,'sinkhorn',lambda *a,**k: pytest.fail('Online MLL must not call OT'))
    model,callbacks=runner.create_algorithm(cfg)
    model.set_logger(configure(str(tmp_path/'test_logs'),['csv']))
    cb=callbacks.callbacks[0]
    cb.eval_env.envs[0].env._max_episode_steps=2
    model.get_env().envs[0].env._max_episode_steps=2
    before=copy.deepcopy(model.policy.actor_state.params)
    try:
        model.learn(total_timesteps=6,callback=callbacks)
        assert calls and set(calls)=={(64,1,'direct_gmm_nll')}
        assert model.backup_mode=='td' and model.behavior_uniform_count==0
        assert model._n_updates==4
        for head in ('mu','log_std'):
            assert not np.array_equal(before[head]['kernel'],model.policy.actor_state.params[head]['kernel'])
        assert all(np.isfinite(p).all() for p in jax.tree_util.tree_leaves(model.policy.actor_state.params))
        assert set(cb.histories)=={'zero_z','stochastic_z'}
    finally:
        cb.eval_env.close();model.get_env().close();model.logger.close()
        OptiQDIME._train.clear_cache()
