import jax
import jax.numpy as jnp
import numpy as np

from optiq_dime.distillation import conditional_ot_nll, hard_projection_mass_error


def test_ot_likelihood_retains_variance_and_has_correct_moment_optimum():
    teachers = jnp.array([[[-2.], [2.]]])
    probabilities = jnp.array([[[.5, .5], [.5, .5]]])
    mu, log_std = jnp.zeros((1,2,1)), jnp.full((1,2,1), jnp.log(2.))
    loss = conditional_ot_nll(mu, log_std, teachers, probabilities)
    np.testing.assert_allclose(loss, .5+jnp.log(2.)+.5*jnp.log(2*jnp.pi), rtol=1e-6)
    grads = jax.grad(conditional_ot_nll, argnums=(0,1))(mu, log_std, teachers, probabilities)
    for g in grads: np.testing.assert_allclose(g, 0., atol=1e-6)
    small_std_gradient = jax.grad(conditional_ot_nll, argnums=1)(mu, jnp.zeros_like(log_std), teachers, probabilities)
    assert bool(jnp.all(small_std_gradient < 0))  # Descent increases sigma.
    assert float(hard_projection_mass_error(probabilities, jnp.array([[.5,.5]]))) == .5


def test_teacher_and_transport_have_no_gradient_and_rows_keep_column_masses():
    mu = jnp.ones((1,2,2)); ls = jnp.zeros_like(mu)
    teachers = jnp.array([[[-1.,.2],[.5,-.7],[2.,1.]]])
    rows = jnp.array([[[.6,.4,0.],[0.,.2,.8]]])
    for g in jax.grad(conditional_ot_nll, argnums=(2,3))(mu,ls,teachers,rows):
        np.testing.assert_array_equal(g,jnp.zeros_like(g))
    direct = (.5*((teachers[:,None]-mu[:,:,None])**2)*jnp.exp(-2*ls[:,:,None])
              + ls[:,:,None]+.5*jnp.log(2*jnp.pi)).sum(-1)
    np.testing.assert_allclose(conditional_ot_nll(mu,ls,teachers,rows),
                               (direct*rows).sum(-1).mean(),rtol=1e-6)


def test_common_ot_actor_nll_updates_both_heads_without_critic_gradient():
    from test_semi_implicit import actor_state, critic_state
    from optiq_dime import OptiQDIME
    actor,critic = actor_state(),critic_state()
    def update(q):
        return OptiQDIME.update_actor(actor,q,jnp.ones((3,3)),jax.random.PRNGKey(5),
            jnp.array([-3600.]),16,4,'exact',.8,.5,False,True,1.,False,16.,257,
            .5,.25,100,'mean','argmax',True,False,'conditional_ot_nll')
    new,loss,_,metrics = update(critic)
    assert np.isfinite(loss) and all(np.isfinite(v).all() for v in metrics.values())
    for head in ('mu','log_std'):
        assert any(not np.array_equal(x,y) for x,y in zip(jax.tree_util.tree_leaves(actor.params[head]),
                                                        jax.tree_util.tree_leaves(new.params[head])))
    for g in jax.tree_util.tree_leaves(jax.grad(lambda p:update(critic.replace(params=p))[1])(critic.params)):
        np.testing.assert_array_equal(g,jnp.zeros_like(g))
