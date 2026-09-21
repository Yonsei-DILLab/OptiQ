"""Independent likelihood/gradient and real v5 actor-update checks."""
import numpy as np
from scipy.special import logsumexp
import jax
import jax.numpy as jnp
from optiq_dime.distillation import direct_gmm_nll, conditional_ot_nll
from optiq_dime.algorithm import OptiQDIME
from optiq_dime.semi_implicit import ConditionalGaussianProposal
import optiq_dime.algorithm as module
from test_semi_implicit import actor_state, critic_state


def test_independent_joint_likelihood_gradients_and_detached_teacher():
    rng=np.random.default_rng(1717)
    mu=rng.normal(size=(2,3,2)).astype(np.float32)
    ls=rng.uniform(-1.5,.3,size=mu.shape).astype(np.float32)
    u=rng.normal(size=(2,5,2)).astype(np.float32)
    w=rng.dirichlet(np.ones(5),size=2).astype(np.float32)
    ell=(-.5*((u[:,None].astype(float)-mu[:,:,None])/np.exp(ls[:,:,None]))**2-ls[:,:,None]-.5*np.log(2*np.pi)).sum(-1)
    expected=-(w*(logsumexp(ell,axis=1)-np.log(3))).sum(-1).mean()
    fn=lambda m,l,t,v:direct_gmm_nll(m,l,t,v)[0]
    np.testing.assert_allclose(fn(mu,ls,u,w),expected,rtol=2e-6)
    gm,gl,gu,gw=jax.grad(fn,argnums=(0,1,2,3))(*map(jnp.asarray,(mu,ls,u,w)))
    A=w[:,None,:]*np.exp(ell-logsumexp(ell,axis=1,keepdims=True))
    delta=mu[:,:,None]-u[:,None]
    em=(A[...,None]*delta).sum(2)*np.exp(-2*ls)/2
    el=(A[...,None]*(1-delta**2*np.exp(-2*ls[:,:,None]))).sum(2)/2
    np.testing.assert_allclose(gm,em,atol=3e-6,rtol=2e-5)
    np.testing.assert_allclose(gl,el,atol=3e-6,rtol=2e-5)
    assert np.count_nonzero(gu)==np.count_nonzero(gw)==0
    singleton=direct_gmm_nll(mu[:,:1],ls[:,:1],u,w)[0]
    np.testing.assert_allclose(singleton,conditional_ot_nll(mu[:,:1],ls[:,:1],u,w[:,None]),rtol=2e-6)


def test_real_update_matches_frozen_teacher_and_never_calls_ot(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('GMM called Sinkhorn')
    monkeypatch.setattr(module,'sinkhorn',forbidden)
    actor=actor_state();critic=critic_state();obs=jnp.ones((4,3));key=jax.random.PRNGKey(817)
    result=OptiQDIME.update_actor(actor,critic,obs,key,jnp.array([-3600.]),
        16,4,'exact',.05,.5,False,True,1.,False,16.,257,.5,.1,100,'mean','argmax',
        True,False,'direct_gmm_nll','conditional_mixture',0.,False,'mean')
    nextkey,lk,pk,_=jax.random.split(key,4);zk,_=jax.random.split(lk)
    z=jax.random.normal(zk,(4,16,2));obsrep=jnp.repeat(obs,16,axis=0)
    mu,ls=actor.apply_fn({'params':actor.params},obsrep,z.reshape(-1,2))
    mu,ls=mu.reshape(4,16,2),ls.reshape(4,16,2)
    proposal=ConditionalGaussianProposal(mu,ls,.05)
    _,u,_=proposal.sample(pk,4,'exact');w=jax.nn.softmax(-proposal.log_prob(u),-1)
    def frozen(params):
        m,l=actor.apply_fn({'params':params},obsrep,z.reshape(-1,2))
        return direct_gmm_nll(m.reshape(4,16,2),l.reshape(4,16,2),u,w)[0]
    loss,g=jax.value_and_grad(frozen)(actor.params);expected=actor.apply_gradients(grads=g)
    np.testing.assert_allclose(result[1],loss,rtol=2e-5)
    for x,y in zip(jax.tree_util.tree_leaves(result[0].params),jax.tree_util.tree_leaves(expected.params)):
        np.testing.assert_allclose(x,y,atol=2e-6,rtol=2e-5)
    np.testing.assert_array_equal(result[2],nextkey)
    assert not any(k.startswith('ot_') for k in result[3])
