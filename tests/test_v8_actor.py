import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from test_v7_actor import small_actor, bias_actor, distinct_component_actor, quadratic_q
from optiq_dime.conditional_sac import stratified_resample
from optiq_dime.conditional_sac_v8 import assignment_log_probs, prepare_batch, actor_objective, update_actor
from optiq_dime.semi_implicit import ConditionalGaussianProposal


def settings():
    return dict(num_students=16,proposal_components=8,actor_samples=4,teacher_resample_count=4,
                min_iterations=5,max_iterations=1000,relative_tolerance=1e-3,
                temperature=1.,epsilon=.1)


def test_teacher_corrected_once_and_no_actor_source_weight():
    actor=small_actor(); obs=jnp.zeros((2,3)); key=jax.random.PRNGKey(11)
    data=prepare_batch(actor,obs,key,quadratic_q,**settings())
    np.testing.assert_allclose(data['teacher_log_w'],jax.nn.log_softmax(data['teacher_q']-data['teacher_log_q']),atol=2e-6)
    resample_key=jax.random.split(key,7)[4]
    np.testing.assert_array_equal(data['teacher_indices'],stratified_resample(resample_key,jnp.exp(data['teacher_log_w']),4))
    np.testing.assert_array_equal(data['ot_weights'],jnp.ones((2,4))/4)
    np.testing.assert_array_equal(data['source_importance'],jnp.ones((2,4)))
    assert 'source_mu' not in data and 'source_log_std' not in data
    expected_cost=jnp.sum((data['anchors'][:,:,None]-data['ot_u'][:,None])**2,axis=-1)
    np.testing.assert_allclose(data['cost'],expected_cost,atol=1e-7)
    updated,loss,new_key,metrics=update_actor(actor,obs,key,quadratic_q,**settings())
    assert int(updated.step)==1 and np.isfinite(loss)
    assert float(metrics['actor_update_accepted'])==1
    assert float(metrics['actor_nll_used'])==0
    assert float(metrics['actor_source_importance_used'])==0
    assert float(metrics['teacher_valid'])==1
    assert float(metrics['v8_latent_conditional_sac_used'])==1
    assert float(metrics['ot_source_gaussian_forward_used'])==0
    np.testing.assert_allclose(metrics['ot_epsilon'],.1)
    assert not np.array_equal(key,new_key)


def test_fixed_teacher_assignment_ignores_actor_sigma_and_temperature(monkeypatch):
    # Symmetric fixed teacher coordinates have equal weights under either
    # centered proposal. Changing g's sigma or alpha must not change the raw
    # geometric assignment when the weighted teacher itself is unchanged.
    fixed_u=jnp.array([[[-.5],[.5]]])
    def fixed_sample(self,key,repeats,mode):
        del self,key,repeats,mode
        return jnp.tanh(fixed_u),fixed_u,jnp.array([[0,1]])
    monkeypatch.setattr(ConditionalGaussianProposal,'sample',fixed_sample)
    obs=jnp.zeros((1,1));key=jax.random.PRNGKey(3)
    anchors=jnp.array([[[-1.],[-.2],[.3],[1.]]])
    cfg=dict(num_students=4,proposal_components=2,actor_samples=2,
             teacher_resample_count=2,student_latents=anchors,epsilon=.1,
             max_iterations=1000,relative_tolerance=1e-4)
    q_fn=lambda observations,actions:jnp.zeros(actions.shape[:-1])
    first=prepare_batch(bias_actor(mu=0.,log_std=-5.),obs,key,q_fn,temperature=.2,**cfg)
    second=prepare_batch(bias_actor(mu=0.,log_std=.5),obs,key,q_fn,temperature=1.,**cfg)
    assert np.all(first['ot']['converged']) and np.all(second['ot']['converged'])
    np.testing.assert_allclose(first['teacher_log_w'],second['teacher_log_w'],atol=1e-7)
    for field in ('teacher_indices','cost','source_indices'):
        np.testing.assert_array_equal(first[field],second[field])
    np.testing.assert_array_equal(first['ot']['plan'],second['ot']['plan'])
    assert not np.allclose(first['teacher_log_q'],second['teacher_log_q'])
    # The geometric plan assigns nearby sites instead of equal-density g rows.
    assert float(first['ot']['plan'][0,0,0])>100*float(first['ot']['plan'][0,0,1])


def test_only_proposal_and_selected_actor_forward_use_g_and_sigma_is_unfloored():
    calls=[];actor=distinct_component_actor(calls);obs=jnp.zeros((2,1))
    data=prepare_batch(actor,obs,jax.random.PRNGKey(3),quadratic_q,**settings())
    _,metrics=actor_objective(actor.params,actor,obs,data,quadratic_q,temperature=1.,epsilon=.1)
    assert calls==[(2*8,2),(2*4,2)]
    np.testing.assert_allclose(metrics['actor_std_min'],.015,rtol=1e-6)


def test_actor_assignment_and_q_gradients_live_but_map_frozen():
    actor=bias_actor(mu=.2);obs=jnp.zeros((1,1));temperature=.7;epsilon=.4
    anchors=jnp.array([[[-1.],[1.]]]);potential=jnp.array([[.2,-.1]])
    data=dict(anchors=anchors,actor_z=anchors[:,:1],source_indices=jnp.array([[0]]),
              actor_noise=jnp.zeros((1,1,1)),ot=dict(source_potential=potential))
    q_fn=lambda observations,actions:.3*actions[...,0]
    def objective(params,points,dual):
        frozen_data=dict(data,anchors=points,ot=dict(source_potential=dual))
        return actor_objective(params,actor,obs,frozen_data,q_fn,
                               temperature=temperature,epsilon=epsilon)[0]
    gradients,anchor_gradient,dual_gradient=jax.grad(objective,argnums=(0,1,2))(
        actor.params,anchors,potential)
    u=jnp.array([[[.2]]]);r=jnp.exp(assignment_log_probs(u,anchors,potential,epsilon))
    expected_z=jnp.sum(r*anchors.swapaxes(1,2),axis=-1).squeeze()
    assignment_derivative=2/epsilon*(-1.-expected_z)
    expected_mu=(2*temperature*jnp.tanh(.2)-.3*(1-jnp.tanh(.2)**2)
                 -temperature*assignment_derivative)
    np.testing.assert_allclose(gradients['mu']['bias'][0],expected_mu,rtol=1e-6)
    np.testing.assert_allclose(gradients['log_std']['bias'][0],-temperature,rtol=1e-6)
    np.testing.assert_array_equal(anchor_gradient,jnp.zeros_like(anchors))
    np.testing.assert_array_equal(dual_gradient,jnp.zeros_like(potential))


def test_failed_solve_rejects_actor_adam_rng():
    actor=small_actor();obs=jnp.zeros((2,3));key=jax.random.PRNGKey(20)
    cfg=dict(settings(),min_iterations=1,max_iterations=1,relative_tolerance=1e-9)
    updated,_,new_key,metrics=update_actor(actor,obs,key,quadratic_q,**cfg)
    assert float(metrics['actor_update_accepted'])==0
    for a,b in zip(jax.tree.leaves(actor),jax.tree.leaves(updated)):
        np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(key,new_key)


def test_shared_source_compatibility_flag_does_not_alias_states():
    actor=small_actor();obs=jnp.array([[0.,0.,0.],[1.,1.,1.]])
    key=jax.random.PRNGKey(2)
    shared=prepare_batch(actor,obs,key,quadratic_q,shared_source=True,**settings())
    separate=prepare_batch(actor,obs,key,quadratic_q,shared_source=False,**settings())
    for first,second in zip(jax.tree.leaves(shared),jax.tree.leaves(separate)):
        np.testing.assert_array_equal(first,second)
    assert not np.array_equal(shared['teacher_u'][0],shared['teacher_u'][1])


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
