import jax
import jax.numpy as jnp
import numpy as np
import pytest
from optiq_dime.semi_implicit import ConditionalGaussianProposal,conditional_mixture_log_prob


@pytest.mark.parametrize('mode',['exact','stratified'])
def test_sampling_and_density_use_identical_floored_conditional_scales(mode):
    mu=jnp.array([[[-1.,.3],[1.,-.3]]]);ls=jnp.log(jnp.array([[[.1,.2],[.5,1.2]]]))
    proposal=ConditionalGaussianProposal(mu,ls,.8)
    actions,u,indices=proposal.sample(jax.random.PRNGKey(1),2000,mode)
    for component in (0,1):
        selected=np.asarray(u)[0,np.asarray(indices)[0]==component]
        np.testing.assert_allclose(selected.mean(0),np.asarray(mu)[0,component],atol=.08)
        np.testing.assert_allclose(selected.std(0),np.maximum(np.exp(np.asarray(ls))[0,component],.8),atol=.08)
    np.testing.assert_allclose(actions,jnp.tanh(u),atol=1e-7)
    np.testing.assert_allclose(proposal.log_prob(u),
        conditional_mixture_log_prob(u,mu,jnp.maximum(ls,jnp.log(.8))),atol=1e-6)


def test_full_update_conditional_proposal_has_finite_gradients():
    from test_semi_implicit import actor_state,critic_state
    from optiq_dime import OptiQDIME
    actor,critic=actor_state(),critic_state()
    updated,loss,_,metrics=OptiQDIME.update_actor(actor,critic,jnp.ones((3,3)),jax.random.PRNGKey(5),
        jnp.array([-3600.]),16,4,'exact',.8,.5,False,True,1.,False,16.,257,.5,.25,100,
        'mean','argmax',True,False,'conditional_ot_nll','conditional_mixture')
    assert np.isfinite(loss) and all(np.isfinite(v).all() for v in metrics.values())
    assert int(updated.step)==int(actor.step)+1
