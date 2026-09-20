"""OptiQ v8: raw latent-to-teacher OT and a separately learned actor."""
import math
from numbers import Real

import jax
import jax.numpy as jnp

from .conditional_sac import components, conditional_action_log_prob, stratified_resample
from .semi_implicit import ConditionalGaussianProposal
from .gaussian_transport import fresh_balanced_sinkhorn
from .latent_transport import latent_transport_cost


def assignment_log_probs(query_u, anchors, source_potential, epsilon=.1):
    """Live query against raw latent sites; potential stores dimensionless f/eps."""
    cost = latent_transport_cost(anchors, query_u).swapaxes(1, 2)
    return jax.nn.log_softmax(source_potential[:, None, :] - cost/epsilon, axis=-1)


def prepare_batch(actor_state, observations, key, q_fn, *, num_students=4096,
                  proposal_components=256, proposals_per_component=1,
                  teacher_sampling_mode='stratified', proposal_std=.05,
                  temperature=.25, epsilon=.1, max_iterations=2000, min_iterations=10,
                  relative_tolerance=1e-3, actor_samples=16, teacher_resample_count=16,
                  latent_seed=0, student_latents=None, action_scale=1.,
                  shared_source=False):
    """Fresh per-state raw-coordinate maps, then one latent draw per teacher.

    ``shared_source`` is retained for adapter compatibility only. There are no
    source-Gaussian forwards to share; every lane's teacher and OT stay separate.
    """
    if min(num_students, proposal_components, proposals_per_component, actor_samples, teacher_resample_count) < 1:
        raise ValueError('Positive sample counts required')
    if actor_samples != teacher_resample_count:
        raise ValueError('One actor query per resampled teacher occurrence required')
    for name, value in [('temperature', temperature), ('epsilon', epsilon),
                        ('proposal_std', proposal_std), ('action_scale', action_scale)]:
        if isinstance(value, Real) and not 0 < value < float('inf'):
            raise ValueError(f'{name} must be finite and positive')
    next_key, source_key, proposal_z_key, proposal_key, resample_key, index_key, noise_key = jax.random.split(key, 7)
    batch = observations.shape[0]
    dimension = actor_state.params['mu']['bias'].shape[0]
    if student_latents is None:
        if latent_seed is None:
            anchors = jax.random.normal(source_key, (batch, num_students, dimension))
        else:
            bank = jax.random.normal(jax.random.PRNGKey(latent_seed), (num_students, dimension))
            anchors = jnp.broadcast_to(bank, (batch, num_students, dimension))
    else:
        if student_latents.shape != (batch, num_students, dimension):
            raise ValueError('Invalid student latent shape')
        anchors = student_latents
    anchors = jax.lax.stop_gradient(anchors)
    frozen_params = jax.lax.stop_gradient(actor_state.params)
    proposal_z = jax.random.normal(proposal_z_key, (batch, proposal_components, dimension))
    old_mu, old_ls = components(actor_state, frozen_params, observations, proposal_z)
    proposal = ConditionalGaussianProposal(old_mu, old_ls, proposal_std)
    teacher_actions, teacher_u, component_indices = proposal.sample(proposal_key, proposals_per_component, teacher_sampling_mode)
    teacher_log_q = proposal.log_prob(teacher_u) - dimension*jnp.log(action_scale)
    teacher_q = q_fn(observations, teacher_actions)
    if teacher_q.shape != teacher_u.shape[:-1]:
        raise ValueError('q_fn must return [B,M]')
    log_w = jax.nn.log_softmax(teacher_q/temperature-teacher_log_q, axis=-1)
    # A NaN CDF can silently make stratified resampling choose column zero.
    # Validate the original teacher as well as the eventual actor loss/map:
    # a finite resampled subset is not evidence that importance weighting worked.
    teacher_finite = jnp.all(jnp.stack([
        jnp.all(jnp.isfinite(value)) for value in
        (teacher_u, teacher_actions, teacher_q, teacher_log_q, log_w,
         old_mu, old_ls, anchors)
    ]))
    teacher_weights = jnp.exp(log_w)
    teacher_valid = (teacher_finite & jnp.all(jnp.isfinite(teacher_weights))
                     & jnp.all(teacher_weights >= 0.)
                     & jnp.all(jnp.abs(teacher_weights.sum(-1)-1.) < 1e-5))
    indices = stratified_resample(resample_key, jnp.exp(log_w), teacher_resample_count)
    ot_u = jnp.take_along_axis(teacher_u, indices[..., None], axis=1)
    ot_weights = jnp.full(ot_u.shape[:-1], 1/teacher_resample_count)
    cost = latent_transport_cost(anchors, ot_u)
    log_kernel = -cost/epsilon
    ot = fresh_balanced_sinkhorn(log_kernel, ot_weights, max_iterations=max_iterations,
                                min_iterations=min_iterations, relative_tolerance=relative_tolerance)
    source_indices = jax.random.categorical(index_key, ot['log_assignment'].swapaxes(1, 2), axis=-1)
    actor_z = jnp.take_along_axis(anchors, source_indices[..., None], axis=1)
    source_log_mass = jnp.log(ot['source_mass'])
    data = dict(key=next_key, anchors=anchors, actor_z=actor_z, source_indices=source_indices,
        actor_noise=jax.random.normal(noise_key, actor_z.shape), source_log_mass=source_log_mass,
        source_importance=jnp.ones(source_indices.shape),
        source_log_importance=jnp.zeros(source_indices.shape), source_importance_correction=jnp.asarray(False),
        teacher_u=teacher_u, teacher_actions=teacher_actions, teacher_q=teacher_q,
        teacher_log_q=teacher_log_q, teacher_log_w=log_w, teacher_component_indices=component_indices,
        teacher_indices=indices, ot_u=ot_u, pair_u=ot_u, ot_weights=ot_weights,
        pair_teacher_indices=jnp.broadcast_to(jnp.arange(teacher_resample_count), source_indices.shape),
        ot=ot, cost=cost, shared_source_valid=jnp.asarray(True),
        teacher_valid=teacher_valid)
    return jax.lax.stop_gradient(data)


def actor_objective(params, actor_state, observations, data, q_fn, *, temperature,
                    epsilon=.1, action_scale=1.):
    mu, ls = components(actor_state, params, observations, data['actor_z'])
    u = mu + jnp.exp(ls)*data['actor_noise']
    actions = jnp.tanh(u)
    log_pi = conditional_action_log_prob(u, mu, ls, action_scale)
    # g is learned separately: raw z defines the map, not g(z)'s mu or sigma.
    # The new action query remains differentiable through its pre-tanh position.
    anchors, potential = jax.lax.stop_gradient((data['anchors'], data['ot']['source_potential']))
    log_assignment = assignment_log_probs(u, anchors, potential, epsilon)
    selected = jnp.take_along_axis(log_assignment, data['source_indices'][..., None], axis=-1)[..., 0]
    q = q_fn(observations, actions)
    entropy, allocation = temperature*log_pi, -temperature*selected
    loss = jnp.mean(entropy-q+allocation)
    return loss, dict(actor_loss=loss, actor_q_term=-q.mean(), actor_q_mean=q.mean(),
        actor_entropy_term=entropy.mean(), actor_assignment_term=allocation.mean(),
        actor_entropy_temperature=jnp.asarray(temperature), actor_conditional_entropy=-log_pi.mean(),
        actor_assigned_log_probability=selected.mean(), actor_std_mean=jnp.exp(ls).mean(),
        actor_std_min=jnp.exp(ls).min(), actor_std_max=jnp.exp(ls).max(),
        actor_log_std_mean=ls.mean(), actor_log_std_min=ls.min(), actor_log_std_max=ls.max(),
        student_action_saturation_fraction=(jnp.abs(actions)>.98).mean(),
        actor_training_pairs=jnp.asarray(float(u.shape[1])), actor_nll_used=jnp.asarray(0.),
        actor_source_importance_used=jnp.asarray(0.), actor_source_importance_mean=jnp.asarray(1.),
        actor_source_importance_max=jnp.asarray(1.), actor_source_importance_ess_fraction=jnp.asarray(1.),
        actor_source_log_importance_max=jnp.asarray(0.), v8_latent_conditional_sac_used=jnp.asarray(1.))


def update_actor(actor_state, observations, key, q_fn, *, return_batch=False, **settings):
    data = prepare_batch(actor_state, observations, key, q_fn, **settings)
    temperature = settings.get('temperature', .25)
    epsilon = settings.get('epsilon', .1)
    action_scale = settings.get('action_scale', 1.)
    (loss, metrics), gradients = jax.value_and_grad(actor_objective, has_aux=True)(
        actor_state.params, actor_state, observations, data, q_fn,
        temperature=temperature, epsilon=epsilon, action_scale=action_scale)
    norm = jnp.sqrt(sum(jnp.sum(g*g) for g in jax.tree.leaves(gradients)))
    proposal = actor_state.apply_gradients(grads=gradients)
    finite = jnp.isfinite(loss) & jnp.isfinite(norm)
    finite = finite & jnp.all(jnp.stack([jnp.all(jnp.isfinite(p)) for p in jax.tree.leaves(proposal)]))
    accepted = (jnp.all(data['ot']['converged']) & data['shared_source_valid']
                & data['teacher_valid'] & finite)
    updated = jax.lax.cond(accepted, lambda _: proposal, lambda _: actor_state, operand=None)
    next_key = jnp.where(accepted, data['key'], key)
    ot, weights = data['ot'], jnp.exp(data['teacher_log_w'])
    metrics.update(actor_gradient_norm=norm, actor_update_accepted=accepted.astype(jnp.float32),
        actor_finite=finite.astype(jnp.float32), source_shared_valid=data['shared_source_valid'].astype(jnp.float32),
        teacher_valid=data['teacher_valid'].astype(jnp.float32),
        ot_fresh_solve=jnp.asarray(1.), ot_persistent_dual=jnp.asarray(0.),
        ot_epsilon=jnp.asarray(epsilon), ot_source_gaussian_forward_used=jnp.asarray(0.),
        ot_converged_fraction=ot['converged'].astype(jnp.float32).mean(),
        ot_row_relative_error=ot['row_relative_error'].max(), ot_col_relative_error=ot['column_relative_error'].max(),
        ot_row_marginal_error=ot['row_error'].max(), ot_col_marginal_error=ot['column_error'].max(),
        ot_source_mass_tv=ot['source_tv'].mean(), ot_iterations_mean=ot['iterations'].astype(jnp.float32).mean(),
        ot_iterations_max=ot['iterations'].astype(jnp.float32).max(),
        ot_cost_mean=data['cost'].mean(), ot_source_count=jnp.asarray(float(data['anchors'].shape[1])),
        ot_teacher_count=jnp.asarray(float(data['ot_u'].shape[1])),
        teacher_candidate_count=jnp.asarray(float(weights.shape[1])),
        teacher_proposal_component_count=jnp.asarray(float(settings.get('proposal_components',256))),
        teacher_one_per_latent=jnp.asarray(float(settings.get('proposals_per_component',1)==1 and settings.get('teacher_sampling_mode','stratified')=='stratified')),
        teacher_log_density_mean=data['teacher_log_q'].mean(), teacher_beta_mean=jnp.asarray(1.),
        teacher_ess_mean=(1/jnp.sum(weights**2, axis=-1)).mean(),
        teacher_action_saturation_fraction=(jnp.abs(data['teacher_actions'])>.98).mean(),
        temperature=jnp.asarray(temperature), proposal_std_pretanh=jnp.asarray(settings.get('proposal_std',.05)))
    result = updated, loss, next_key, metrics
    return (*result, data) if return_batch else result
