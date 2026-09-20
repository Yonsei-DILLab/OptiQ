"""OptiQ v7: Boltzmann actor improvement conditional on a frozen latent OT map.

Shared by MuJoCo and the fixed-Q GMM validation. Q is a callback on normalized
actions [B,K,D]; its parameters must be frozen but its action input remains
differentiable. No policy-density division is applied to actor-generated draws.
The proposal density correction belongs to teacher construction only.
"""
import math
from numbers import Real

import jax
import jax.numpy as jnp

from .semi_implicit import ConditionalGaussianProposal, tanh_log_jacobian
from .latent_transport import (
    latent_transport_cost, sinkhorn_with_source_potential,
    latent_assignment_log_probs,
)


def components(actor_state, params, observations, latents):
    batch, count, dimension = latents.shape
    obs = jnp.broadcast_to(observations[:, None, :],
                           (batch, count, observations.shape[-1]))
    mu, log_std = actor_state.apply_fn(
        {'params': params}, obs.reshape(-1, observations.shape[-1]),
        latents.reshape(-1, dimension))
    return mu.reshape(latents.shape), log_std.reshape(latents.shape)


def conditional_action_log_prob(u, mu, log_std, action_scale=1.):
    """Actual generating conditional density, including tanh and scale Jacobian."""
    normal = (-.5*((u-mu)*jnp.exp(-log_std))**2-log_std
              -.5*math.log(2*math.pi)).sum(axis=-1)
    return normal-tanh_log_jacobian(u)-u.shape[-1]*math.log(action_scale)


def stratified_resample(key, weights, count):
    positions = (jnp.arange(count)[None, :]+jax.random.uniform(
        key, (weights.shape[0], count), dtype=weights.dtype))/count
    cdf = jnp.cumsum(weights, axis=-1).at[:, -1].set(1.)
    return jnp.minimum((positions[..., None] > cdf[:, None, :]).sum(-1),
                       weights.shape[-1]-1)


def prepare_batch(actor_state, observations, key, q_fn, *, num_students=4096,
                  proposal_components=256, proposals_per_component=1,
                  teacher_sampling_mode='stratified', proposal_std=.05,
                  temperature=.25, epsilon=.1, iterations=100,
                  actor_samples=16, teacher_resample_count=16,
                  student_latents=None, latent_seed=0, action_scale=1., source_potential=None,
                  source_importance_correction=True):
    """Build one fresh, detached teacher and OT map; retain new action noise.

    Each importance-resampled teacher occurrence selects one latent from its
    OT conditional. By default, detached (1/H)/sum_j(P_ij) corrects the resulting
    source sampling to the uniform quadrature prior; it is not teacher W applied
    twice. The explicit no-correction ablation uses unit actor weights, retaining
    raw log ratios for diagnostics. Teacher weighting and all OT draws are
    unchanged. This sampling correction does not repair an unconverged partition.
    """
    if num_students < 1 or proposals_per_component < 1 or iterations < 1:
        raise ValueError('Sample counts and OT iterations must be positive')
    if proposal_components is None:
        proposal_components = num_students
    if proposal_components < 1 or teacher_resample_count < 0:
        raise ValueError('Invalid teacher counts')
    if actor_samples is not None and actor_samples < 1:
        raise ValueError('actor_samples must be positive')
    if teacher_sampling_mode not in ('exact', 'stratified'):
        raise ValueError('Unknown teacher sampling mode')
    for name, value in [('action_scale', action_scale), ('proposal_std', proposal_std),
                        ('temperature', temperature), ('epsilon', epsilon)]:
        if isinstance(value, Real) and not 0 < value < float('inf'):
            raise ValueError(f'{name} must be positive and finite')
    count = num_students if actor_samples is None else actor_samples
    if teacher_resample_count and count != teacher_resample_count:
        raise ValueError('Use one actor pair per compressed teacher occurrence')
    keys = jax.random.split(key, 7)
    next_key, source_key, proposal_z_key, proposal_key, resample_key, index_key, noise_key = keys
    batch = observations.shape[0]
    dimension = actor_state.params['mu']['bias'].shape[0]
    if student_latents is None:
        if latent_seed is None:
            anchors = jax.random.normal(source_key, (batch, num_students, dimension),
                                        dtype=observations.dtype)
        else:
            bank = jax.random.normal(jax.random.PRNGKey(latent_seed),
                                     (num_students, dimension), dtype=observations.dtype)
            anchors = jnp.broadcast_to(bank, (batch, num_students, dimension))
    else:
        if student_latents.shape != (batch, num_students, dimension):
            raise ValueError('student_latents must have shape [B,num_students,action_dim]')
        anchors = student_latents
    anchors = jax.lax.stop_gradient(anchors)
    proposal_z = jax.random.normal(proposal_z_key, (batch, proposal_components, dimension),
                                   dtype=observations.dtype)
    frozen_params = jax.lax.stop_gradient(actor_state.params)
    old_mu, old_ls = components(actor_state, frozen_params, observations, proposal_z)
    proposal = ConditionalGaussianProposal(old_mu, old_ls, proposal_std)
    teacher_actions, teacher_u, teacher_component_indices = proposal.sample(
        proposal_key, proposals_per_component, teacher_sampling_mode)
    teacher_log_q = proposal.log_prob(teacher_u)-dimension*math.log(action_scale)
    teacher_q = q_fn(observations, teacher_actions)
    if teacher_q.shape != teacher_u.shape[:-1]:
        raise ValueError('q_fn must return [B,K] for actions [B,K,D]')
    teacher_log_w = jax.nn.log_softmax(teacher_q/temperature-teacher_log_q, axis=-1)
    teacher_u, teacher_actions, teacher_q, teacher_log_q, teacher_log_w = jax.lax.stop_gradient(
        (teacher_u, teacher_actions, teacher_q, teacher_log_q, teacher_log_w))
    if teacher_resample_count:
        indices = stratified_resample(resample_key, jnp.exp(teacher_log_w), teacher_resample_count)
        ot_u = jnp.take_along_axis(teacher_u, indices[..., None], axis=1)
        ot_weights = jnp.full(ot_u.shape[:-1], 1/teacher_resample_count, dtype=ot_u.dtype)
    else:
        indices = jnp.broadcast_to(jnp.arange(teacher_u.shape[1]), teacher_u.shape[:-1])
        ot_u, ot_weights = teacher_u, jnp.exp(teacher_log_w)
    cost = latent_transport_cost(anchors, ot_u)
    if source_potential is None:
        ot = jax.lax.stop_gradient(sinkhorn_with_source_potential(
            cost, ot_weights, epsilon, iterations))
    else:
        potential = jax.lax.stop_gradient(jnp.asarray(source_potential))
        if potential.shape not in ((1, num_students), (batch, num_students)):
            raise ValueError('source_potential must have shape [1,H] or [B,H]')
        potential = jnp.broadcast_to(potential, (batch, num_students))
        log_assignment = jax.nn.log_softmax((potential[:, :, None]-cost)/epsilon, axis=1)
        plan = ot_weights[:, None, :]*jnp.exp(log_assignment)
        source_mass, teacher_mass = plan.sum(-1), plan.sum(-2)
        ot = jax.lax.stop_gradient(dict(
            plan=plan, source_potential=potential,
            source_mass=source_mass, teacher_mass=teacher_mass,
            row_error=jnp.abs(source_mass-1/num_students).max(-1),
            column_error=jnp.abs(teacher_mass-ot_weights).max(-1),
            source_tv=.5*jnp.abs(source_mass-1/num_students).sum(-1)))
    log_r = latent_assignment_log_probs(ot_u, anchors, ot['source_potential'], epsilon)
    log_mass = jax.scipy.special.logsumexp(
        log_r + jnp.log(ot_weights)[..., None], axis=1)
    if teacher_resample_count:
        pair_teacher_indices = jnp.broadcast_to(jnp.arange(count), (batch, count))
        pair_log_r = log_r
    else:
        index_key, teacher_index_key = jax.random.split(index_key)
        pair_teacher_indices = jax.random.categorical(
            teacher_index_key, jnp.log(ot_weights)[:, None, :], shape=(batch, count))
        pair_log_r = jnp.take_along_axis(log_r, pair_teacher_indices[..., None], axis=1)
    source_indices = jax.random.categorical(index_key, pair_log_r, axis=-1)
    selected_log_mass = jnp.take_along_axis(log_mass, source_indices, axis=1)
    source_log_importance = jax.lax.stop_gradient(-math.log(num_students)-selected_log_mass)
    correction_used = jnp.asarray(source_importance_correction, dtype=jnp.bool_)
    source_importance = jnp.exp(jnp.where(correction_used, source_log_importance, 0.))
    actor_z = jnp.take_along_axis(anchors, source_indices[..., None], axis=1)
    noise = jax.random.normal(noise_key, actor_z.shape, dtype=actor_z.dtype)
    return dict(key=next_key, anchors=anchors, actor_z=actor_z,
                source_indices=source_indices, actor_noise=noise,
                pair_teacher_indices=pair_teacher_indices,
                pair_u=jnp.take_along_axis(ot_u, pair_teacher_indices[..., None], axis=1),
                source_log_mass=log_mass, source_log_importance=source_log_importance,
                source_importance=source_importance, source_importance_correction=correction_used,
                teacher_u=teacher_u, teacher_actions=teacher_actions,
                teacher_component_indices=teacher_component_indices,
                teacher_q=teacher_q, teacher_log_q=teacher_log_q,
                teacher_log_w=teacher_log_w, ot_u=ot_u, ot_weights=ot_weights,
                teacher_indices=indices, cost=cost, ot=ot)


def actor_objective(params, actor_state, observations, data, q_fn, *,
                    temperature, epsilon, action_scale=1.):
    """Reparameterized local KL; OT parameters frozen, action path live.

    The finite-bank target is t_i(a) proportional to exp(Q(a)/T)*r(i|a).
    This is not marginal-mixture entropy. At a balanced map, the conditional
    objective is the joint KL to the OT allocation of the Boltzmann target.
    """
    mu, ls = components(actor_state, params, observations, data['actor_z'])
    sigma = jnp.exp(ls)
    u = mu+sigma*data['actor_noise']
    actions = jnp.tanh(u)
    log_pi = conditional_action_log_prob(u, mu, ls, action_scale)
    log_r = latent_assignment_log_probs(u, data['anchors'],
                                        data['ot']['source_potential'], epsilon)
    selected_log_r = jnp.take_along_axis(log_r, data['source_indices'][..., None], axis=-1)[..., 0]
    actor_q = q_fn(observations, actions)
    entropy_term = temperature*log_pi
    assignment_term = -temperature*selected_log_r
    importance = data['source_importance']
    correction_used = jnp.asarray(data.get('source_importance_correction', True), dtype=jnp.float32)
    applied_log_importance = jnp.where(correction_used, data['source_log_importance'], 0.)
    loss = (importance*(entropy_term-actor_q+assignment_term)).mean()
    return loss, dict(
        actor_loss=loss, actor_q_mean=actor_q.mean(),
        actor_q_term=-(importance*actor_q).mean(),
        actor_entropy_term=(importance*entropy_term).mean(),
        actor_conditional_entropy=-log_pi.mean(),
        actor_entropy_temperature=jnp.asarray(temperature),
        actor_assignment_term=(importance*assignment_term).mean(),
        actor_assigned_log_probability=selected_log_r.mean(),
        actor_std_mean=sigma.mean(), actor_std_min=sigma.min(), actor_std_max=sigma.max(),
        actor_log_std_mean=ls.mean(), actor_log_std_min=ls.min(), actor_log_std_max=ls.max(),
        student_action_saturation_fraction=(jnp.abs(actions) > .98).astype(jnp.float32).mean(),
        v7_conditional_sac_used=jnp.asarray(1.),
        actor_nll_used=jnp.asarray(0.), actor_sampling_prior_uniform=jnp.asarray(0.),
        actor_sampling_ot_joint=jnp.asarray(1.), actor_source_importance_used=correction_used,
        actor_source_importance_mean=importance.mean(),
        actor_source_importance_max=importance.max(),
        actor_source_log_importance_max=applied_log_importance.max(),
        actor_source_raw_log_importance_max=data['source_log_importance'].max(),
        actor_source_importance_ess_fraction=jnp.mean(
            importance.sum(-1)**2/(importance.shape[-1]*(importance**2).sum(-1))),
        actor_training_pairs=jnp.asarray(float(u.shape[1])),
    )


def update_actor(actor_state, observations, key, q_fn, *, num_students=4096,
                 proposal_components=256, proposals_per_component=1,
                 teacher_sampling_mode='stratified', proposal_std=.05,
                 temperature=.25, epsilon=.1, iterations=100,
                 actor_samples=16, teacher_resample_count=16,
                 student_latents=None, latent_seed=0, action_scale=1.,
                 source_potential=None, return_batch=False, source_importance_correction=True):
    data = prepare_batch(
        actor_state, observations, key, q_fn, num_students=num_students,
        proposal_components=proposal_components, proposals_per_component=proposals_per_component,
        teacher_sampling_mode=teacher_sampling_mode, proposal_std=proposal_std,
        temperature=temperature, epsilon=epsilon, iterations=iterations,
        actor_samples=actor_samples, teacher_resample_count=teacher_resample_count,
        student_latents=student_latents, latent_seed=latent_seed, action_scale=action_scale,
        source_potential=source_potential, source_importance_correction=source_importance_correction)
    def loss_fn(params):
        return actor_objective(params, actor_state, observations, data, q_fn,
                               temperature=temperature, epsilon=epsilon, action_scale=action_scale)
    (loss, metrics), gradients = jax.value_and_grad(loss_fn, has_aux=True)(actor_state.params)
    new_state = actor_state.apply_gradients(grads=gradients)
    weights = jnp.exp(data['teacher_log_w'])
    ess = 1/jnp.sum(weights**2, axis=-1)
    ot = data['ot']
    metrics.update(
        source_ess_absolute=ess.mean(), source_ess_min=ess.min(),
        source_ess_fraction=(ess/weights.shape[-1]).mean(),
        density_beta_mean=jnp.asarray(1.), density_beta_min=jnp.asarray(1.),
        density_beta_max=jnp.asarray(1.), max_source_weight=weights.max(axis=-1).mean(),
        teacher_log_density_mean=data['teacher_log_q'].mean(),
        source_q_mean=data['teacher_q'].mean(), source_q_std=data['teacher_q'].std(axis=-1).mean(),
        teacher_action_saturation_fraction=(jnp.abs(data['teacher_actions']) > .98).astype(jnp.float32).mean(),
        temperature=jnp.asarray(temperature), proposal_std_pretanh=jnp.asarray(proposal_std),
        ot_cost_mean=data['cost'].mean(),
        ot_row_marginal_error=jnp.max(jnp.abs(ot['source_mass']-1/num_students)),
        ot_col_marginal_error=jnp.max(jnp.abs(ot['teacher_mass']-data['ot_weights'])),
        ot_source_mass_tv=jnp.mean(.5*jnp.abs(ot['source_mass']-1/num_students).sum(axis=-1)),
        ot_teacher_count=jnp.asarray(float(data['ot_u'].shape[1])),
        teacher_candidate_count=jnp.asarray(float(weights.shape[-1])),
        teacher_proposal_component_count=jnp.asarray(float(
            num_students if proposal_components is None else proposal_components)),
        teacher_one_per_latent=jnp.asarray(float(
            teacher_sampling_mode == 'stratified' and proposals_per_component == 1)),
        ot_source_count=jnp.asarray(float(num_students)),
        ot_source_log_mass_min=data['source_log_mass'].min(),
        ot_source_log_importance_second_moment=jax.scipy.special.logsumexp(
            -2*math.log(num_students)-data['source_log_mass'], axis=-1).mean(),
        ot_fixed_quadrature=jnp.asarray(float(student_latents is not None or latent_seed is not None)),
        teacher_resampling_used=jnp.asarray(float(bool(teacher_resample_count))),
        ot_fresh_solve=jnp.asarray(float(source_potential is None)),
        actor_gradient_norm=jnp.sqrt(sum(jnp.sum(g*g) for g in jax.tree.leaves(gradients))),
    )
    result = (new_state, loss, data['key'], metrics)
    return (*result, data) if return_batch else result
