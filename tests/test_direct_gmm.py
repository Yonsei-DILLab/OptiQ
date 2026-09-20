"""Independent density/gradient identities and production Direct GMM routing."""
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf
from scipy.special import logsumexp
from scipy.stats import norm

import optiq_dime.algorithm as algorithm
from optiq_dime.algorithm import OptiQDIME
from optiq_dime.distillation import direct_gmm_nll
from optiq_dime.semi_implicit import ConditionalGaussianProposal, tanh_log_jacobian
from scripts.verify_v5 import verify as verify_v5
from scripts.verify_v5_direct_gmm import verify
from test_semi_implicit import actor_state, critic_state


def forbidden(*args, **kwargs):
    raise AssertionError("Direct GMM called Sinkhorn")


def update(actor, critic, obs, key, loss="direct_gmm_nll", epsilon=.1, iterations=100):
    return OptiQDIME.update_actor(actor, critic, obs, key, jnp.array([-3600.]),
        16, 4, 'exact', .05, .5, False, True, 1., False, 16., 257,
        .25, epsilon, iterations, 'mean', 'argmax', True, False,
        loss, 'conditional_mixture', 0., False, 'mean')


def test_profile_changes_only_loss_and_run_metadata():
    base=OmegaConf.to_container(verify_v5(['benchmark=ant']),resolve=True)
    direct=OmegaConf.to_container(verify(['benchmark=ant']),resolve=True)
    alias=OmegaConf.to_container(verify(['benchmark=ant'],'v5_direct_gmm/final'),resolve=True)
    assert direct==alias
    assert direct['alg']['actor']['distillation_loss']=='direct_gmm_nll'
    direct['alg']['actor']['distillation_loss']='conditional_ot_nll'
    for key in ('run_name','output_root','wandb'):
        direct[key]=base[key]
    assert direct==base


@pytest.mark.parametrize('override',[
    'alg.actor.density_correction=false','alg.actor.density_correction_beta=.5',
    'alg.actor.teacher_distribution=realized_kde','alg.actor.distillation_loss=conditional_ot_nll',
    'alg.actor.adaptive_density_beta=true','dual_mu_eval=false',
])
def test_reject_incompatible_profiles(override):
    with pytest.raises(ValueError):verify([override])


def test_marginal_nll_and_both_head_gradients_match_analytic_responsibilities():
    rng=np.random.default_rng(73)
    mu=rng.normal(size=(2,3,2)).astype('float32')
    ls=rng.uniform(-1,.4,size=mu.shape).astype('float32')
    u=rng.normal(size=(2,7,2)).astype('float32')
    w=rng.dirichlet(np.ones(7),size=2).astype('float32')
    ell=norm.logpdf(u[:,None],loc=mu[:,:,None],scale=np.exp(ls[:,:,None])).sum(-1)
    expected=-(w*(logsumexp(ell,axis=1)-np.log(3))).sum(-1).mean()
    responsibility=np.exp(ell-logsumexp(ell,axis=1,keepdims=True))
    coefficient=w[:,None,:,None]*responsibility[:,:,:,None]/2
    diff=mu[:,:,None]-u[:,None]
    grad_mu=(coefficient*diff*np.exp(-2*ls[:,:,None])).sum(2)
    grad_ls=(coefficient*(1-diff**2*np.exp(-2*ls[:,:,None]))).sum(2)
    value,grads=jax.value_and_grad(lambda a,b:direct_gmm_nll(a,b,jnp.array(u),jnp.array(w))[0],argnums=(0,1))(jnp.array(mu),jnp.array(ls))
    np.testing.assert_allclose(value,expected,rtol=2e-6)
    for actual,expected_grad in zip(grads,(grad_mu,grad_ls)):
        np.testing.assert_allclose(actual,expected_grad,rtol=3e-5,atol=2e-6)
    du,dw=jax.grad(lambda a,b:direct_gmm_nll(jnp.array(mu),jnp.array(ls),a,b)[0],argnums=(0,1))(jnp.array(u),jnp.array(w))
    assert not np.count_nonzero(du) and not np.count_nonzero(dw)
    # An action-space Jacobian changes the reported NLL by a teacher-only
    # constant, so both student-head gradients must be identical.
    action_grads=jax.grad(lambda a,b:direct_gmm_nll(a,b,jnp.array(u),jnp.array(w))[0]
        +(jnp.array(w)*tanh_log_jacobian(jnp.array(u))).sum(-1).mean(),argnums=(0,1))(jnp.array(mu),jnp.array(ls))
    for a,b in zip(grads,action_grads):np.testing.assert_array_equal(a,b)


def test_teacher_density_floor_joint_mixture_jacobian_and_sampling():
    mu=jnp.array([[[-1.,.7],[1.,-.9]],[[.3,.2],[-.4,1.]]])
    ls=jnp.log(jnp.array([[[.01,.8],[.7,.02]],[[.3,.02],[.01,.4]]]))
    teacher=ConditionalGaussianProposal(mu,ls,.05)
    key=jax.random.PRNGKey(97)
    action,u,index=teacher.sample(key,5,'exact')
    component_key,noise_key=jax.random.split(key)
    np.testing.assert_array_equal(index,jax.random.randint(component_key,(2,10),0,2))
    std=np.maximum(np.exp(ls),.05)
    expected_u=np.take_along_axis(np.asarray(mu),np.asarray(index)[:,:,None],axis=1)+np.take_along_axis(std,np.asarray(index)[:,:,None],axis=1)*np.asarray(jax.random.normal(noise_key,(2,10,2)))
    np.testing.assert_allclose(u,expected_u,atol=2e-7)
    np.testing.assert_array_equal(action,jnp.tanh(u))
    ell=norm.logpdf(np.asarray(u)[:,:,None],loc=np.asarray(mu)[:,None],scale=std[:,None]).sum(-1)
    jac=(2*(np.log(2)-np.asarray(u)-np.logaddexp(0,-2*np.asarray(u)))).sum(-1)
    expected=logsumexp(ell,axis=-1)-np.log(2)-jac
    np.testing.assert_allclose(teacher.log_prob(u),expected,atol=3e-5)
    # General nonconstant Q: temperature multiplies Q alone, not -log(q).
    q=.3*np.asarray(action)[...,0]-np.asarray(action)[...,1]**2
    logits=q/.25-expected
    weight=np.exp(logits-logsumexp(logits,axis=-1,keepdims=True))
    np.testing.assert_allclose(jax.nn.softmax(jnp.array(q)/.25-teacher.log_prob(u),axis=-1),weight,atol=2e-6)
    assert np.isfinite(teacher.log_prob(jnp.array([[[100.,-100.]],[[100.,-100.]]]))).all()


@pytest.mark.parametrize('seed',[0,1])
def test_production_teacher_rng_gradient_boundaries_and_no_ot(seed,monkeypatch):
    actor,critic=actor_state(),critic_state()
    obs,key=jnp.ones((4,3)),jax.random.PRNGKey(seed+17)
    baseline=update(actor,critic,obs,key,'conditional_ot_nll')
    OptiQDIME.update_actor.clear_cache()
    monkeypatch.setattr(algorithm,'sinkhorn',forbidden)
    try:
        direct=update(actor,critic,obs,key)
        ignored=update(actor,critic,obs,key,epsilon=.73,iterations=3)
        np.testing.assert_array_equal(baseline[2],direct[2])
        for metric in ('teacher_log_density_mean','source_ess_absolute','max_source_weight',
                       'full_weighted_q_gain','density_beta_mean','policy_spread_l2'):
            np.testing.assert_allclose(baseline[3][metric],direct[3][metric],atol=2e-6)
        for a,b in zip(jax.tree_util.tree_leaves(direct),jax.tree_util.tree_leaves(ignored)):
            np.testing.assert_array_equal(a,b)
        assert not any(k.startswith('ot_') or k in ('hard_projection_mass_tv','selected_delta_l2') for k in direct[3])
        assert direct[3]['density_beta_mean']==1
        for head in ('mu','log_std'):
            assert not np.array_equal(actor.params[head]['kernel'],direct[0].params[head]['kernel'])
        # Reconstruct the precise v5 teacher; constant twin Q cancels from w.
        _,latent_key,proposal_key,_=jax.random.split(key,4)
        zk,_=jax.random.split(latent_key)
        z=jax.random.normal(zk,(4,16,2))
        mu,ls=actor.apply_fn({'params':actor.params},jnp.repeat(obs,16,axis=0),z.reshape(-1,2))
        mu,ls=mu.reshape(4,16,2),ls.reshape(4,16,2)
        teacher=ConditionalGaussianProposal(mu,ls,.05)
        _,u,_=teacher.sample(proposal_key,4,'exact')
        w=jax.nn.softmax(-teacher.log_prob(u),axis=-1)
        np.testing.assert_allclose(direct[1],direct_gmm_nll(mu,ls,u,w)[0],atol=3e-6)
        qgrad=jax.grad(lambda p:update(actor,critic.replace(params=p),obs,key)[1])(critic.params)
        assert all(not np.count_nonzero(x) for x in jax.tree_util.tree_leaves(qgrad))
    finally:OptiQDIME.update_actor.clear_cache()


def test_real_ant_direct_path_keeps_existing_v5_training_and_evaluation(tmp_path,monkeypatch):
    # Reuse the established real-environment assertions: both heads train,
    # plain TD, no clipping/guard, Gaussian backups and paired mu-only eval.
    import test_v5
    OptiQDIME.update_actor.clear_cache()
    monkeypatch.setattr(algorithm,'sinkhorn',forbidden)
    monkeypatch.setattr(test_v5,'verify',verify)
    try:test_v5.test_real_ant_routes_mean_ot_and_keeps_td_and_paired_evaluation(tmp_path,monkeypatch)
    finally:OptiQDIME.update_actor.clear_cache()


class QuadraticEnergy:
    def jax_log_prob(self,x):return -.01*jnp.square(x).sum(-1)


def test_gmm40_shared_teacher_no_ot_and_checkpoint_continuation(tmp_path,monkeypatch):
    import gmm40.optiq as adapter
    kwargs=dict(seed=5,n=4,m=8,batch=3,hidden_dims=(16,16))
    ot=adapter.OptiQ(QuadraticEnergy(),**kwargs)
    original=ot.advance(1)
    monkeypatch.setattr(adapter,'sinkhorn',forbidden)
    direct=adapter.OptiQ(QuadraticEnergy(),**kwargs,distillation_loss='direct_gmm_nll')
    metrics=direct.advance(1)
    for name in ('teacher_ess','teacher_Q','weighted_teacher_Q','sampled_action_std','sigma_mean'):
        np.testing.assert_allclose(metrics[name],original[name],rtol=2e-6,atol=1e-6)
    assert not any(k.startswith('ot_') or k=='row_entropy' for k in metrics)
    path=tmp_path/'state.bin';direct.save(path)
    resumed=adapter.OptiQ(QuadraticEnergy(),**kwargs,distillation_loss='direct_gmm_nll')
    resumed.restore(path)
    a,b=direct.advance(2),resumed.advance(2)
    assert a==b and a['Q_evaluations']==3*3*8
    for a,b in zip(jax.tree_util.tree_leaves(direct.state),jax.tree_util.tree_leaves(resumed.state)):
        np.testing.assert_array_equal(a,b)
    with pytest.raises(ValueError):
        adapter.OptiQ(QuadraticEnergy(),**kwargs,distillation_loss='direct_gmm_nll',sigma_row_balance=True)
