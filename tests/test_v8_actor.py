import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from test_v7_actor import small_actor, bias_actor, quadratic_q
from optiq_dime.conditional_sac import stratified_resample
from optiq_dime.conditional_sac_v8 import prepare_batch, actor_objective, update_actor


def settings():
    return dict(num_students=16,proposal_components=8,actor_samples=4,teacher_resample_count=4,
                min_iterations=5,max_iterations=200,relative_tolerance=1e-3,temperature=1.)


def test_teacher_corrected_once_and_no_actor_source_weight():
    actor=small_actor(); obs=jnp.zeros((2,3)); key=jax.random.PRNGKey(11)
    data=prepare_batch(actor,obs,key,quadratic_q,**settings())
    np.testing.assert_allclose(data['teacher_log_w'],jax.nn.log_softmax(data['teacher_q']-data['teacher_log_q']),atol=2e-6)
    resample_key=jax.random.split(key,7)[4]
    np.testing.assert_array_equal(data['teacher_indices'],stratified_resample(resample_key,jnp.exp(data['teacher_log_w']),4))
    np.testing.assert_array_equal(data['ot_weights'],jnp.ones((2,4))/4)
    np.testing.assert_array_equal(data['source_importance'],jnp.ones((2,4)))
    assert data['source_mu'].shape==(2,16,2)
    updated,loss,new_key,metrics=update_actor(actor,obs,key,quadratic_q,**settings())
    assert int(updated.step)==1 and np.isfinite(loss)
    assert float(metrics['actor_update_accepted'])==1
    assert float(metrics['actor_nll_used'])==0
    assert float(metrics['actor_source_importance_used'])==0
    assert float(metrics['teacher_valid'])==1
    assert not np.array_equal(key,new_key)


def test_source_uses_actual_sigma_and_old_parameters_are_frozen():
    actor=bias_actor(mu=.1,log_std=-5.);obs=jnp.zeros((1,1))
    data=prepare_batch(actor,obs,jax.random.PRNGKey(3),quadratic_q,**settings())
    np.testing.assert_allclose(jnp.exp(data['source_log_std']),np.exp(-5),rtol=1e-6)
    def objective(old_mu):
        altered=dict(data,source_mu=old_mu)
        return actor_objective(actor.params,actor,obs,altered,quadratic_q,temperature=1.)[0]
    assert np.count_nonzero(jax.grad(objective)(data['source_mu']))==0


def test_failed_solve_rejects_actor_adam_rng():
    actor=small_actor();obs=jnp.zeros((2,3));key=jax.random.PRNGKey(20)
    cfg=dict(settings(),min_iterations=1,max_iterations=1,relative_tolerance=1e-9)
    updated,_,new_key,metrics=update_actor(actor,obs,key,quadratic_q,**cfg)
    assert float(metrics['actor_update_accepted'])==0
    for a,b in zip(jax.tree.leaves(actor),jax.tree.leaves(updated)):
        np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(key,new_key)


def test_shared_source_cannot_silently_alias_distinct_states():
    actor=small_actor();obs=jnp.array([[0.,0.,0.],[1.,1.,1.]])
    _,_,_,metrics=update_actor(actor,obs,jax.random.PRNGKey(2),quadratic_q,shared_source=True,**settings())
    assert float(metrics['source_shared_valid'])==0
    assert float(metrics['actor_update_accepted'])==0


def test_invalid_teacher_rejects_finite_actor_update_and_preserves_adam_rng():
    base=bias_actor(mu=.1,log_std=-.7)
    actor=TrainState.create(apply_fn=base.apply_fn,params=base.params,tx=optax.adam(3e-4))
    # Preserve meaningful existing optimizer moments, not only an empty state.
    actor=actor.apply_gradients(grads=jax.tree.map(lambda x:jnp.full_like(x,.2),actor.params))
    obs=jnp.zeros((1,1));key=jax.random.PRNGKey(0)
    def q_fn(observations,actions):
        del observations
        return jnp.where(actions[...,0]>.6,jnp.nan,-actions[...,0]**2)
    run=jax.jit(lambda state,step_key:update_actor(
        state,obs,step_key,q_fn,num_students=8,proposal_components=16,
        actor_samples=2,teacher_resample_count=2,temperature=1.,
        min_iterations=3,max_iterations=20,return_batch=True))
    updated,loss,new_key,metrics,data=run(actor,key)
    assert not np.isfinite(np.asarray(data['teacher_log_w'])).all()
    # Previously this finite subset and fresh-action loss bypassed validation.
    assert np.isfinite(loss)
    assert float(metrics['actor_finite'])==1
    assert float(metrics['ot_converged_fraction'])==1
    assert float(metrics['teacher_valid'])==0
    assert float(metrics['actor_update_accepted'])==0
    for before,after in zip(jax.tree.leaves(actor),jax.tree.leaves(updated)):
        np.testing.assert_array_equal(before,after)
    np.testing.assert_array_equal(new_key,key)
