"""Batched fixed-weight SMEM teacher and truncated-Gaussian trust region.

This transfers the validated GMM40 SMEM+TR update to a state-conditioned
shared actor.  Each replay state has its own stopped 64-component teacher;
the shared actor regresses all projected teachers on the same sampled latents.
"""
import itertools

import jax
import jax.numpy as jnp
import optax

from .box_gaussian import component_log_prob, log_normalizer, sample_box
from .critic_utils import critic_expectation
from .semi_implicit import ConditionalGaussianProposal


DEFAULTS = dict(
    em_steps=2,
    em_m_steps=4,
    em_step_size=.5,
    em_min_ess=2.,
    smem_every=20,
    smem_candidates=2,
    smem_partial_steps=2,
    smem_split_fraction=.5,
    tr_kl=.05,
    tr_bisections=28,
    projection_steps=10,
    projection_scale_weight=1.,
)


def box_moments(mu, log_std):
    """Mean and variance of a normalized diagonal Gaussian on [-1,1]."""
    std = jnp.exp(log_std)
    a, b = (-1. - mu) / std, (1. - mu) / std
    pa = jnp.exp(-.5 * a*a) / jnp.sqrt(2.*jnp.pi)
    pb = jnp.exp(-.5 * b*b) / jnp.sqrt(2.*jnp.pi)
    normalizer = jnp.exp(log_normalizer(mu, log_std))
    shift = (pa - pb) / normalizer
    mean = mu + std * shift
    variance = std**2 * (1. + (a*pa - b*pb) / normalizer - shift**2)
    return mean, jnp.maximum(variance, 0.)


def box_kl(mu, log_std, ref_mu, ref_log_std):
    """Exact per-component KL(new || reference), summed over action dims."""
    mean, variance = box_moments(mu, log_std)
    value = (
        ref_log_std - log_std
        + log_normalizer(ref_mu, ref_log_std) - log_normalizer(mu, log_std)
        + .5 * (
            (variance + (mean-ref_mu)**2) * jnp.exp(-2.*ref_log_std)
            - (variance + (mean-mu)**2) * jnp.exp(-2.*log_std)
        )
    )
    return jnp.maximum(value.sum(-1), 0.)


def natural_interpolate(old, candidate, fraction):
    old_mu, old_log_std = old
    mu, log_std = candidate
    old_precision, precision = jnp.exp(-2.*old_log_std), jnp.exp(-2.*log_std)
    mixed_precision = (1.-fraction)*old_precision + fraction*precision
    mixed_mu = (
        (1.-fraction)*old_precision*old_mu + fraction*precision*mu
    ) / mixed_precision
    return mixed_mu, -.5*jnp.log(mixed_precision)


def project(old, candidate, epsilon, iterations=28):
    """Project one state's joint component/action law into mean KL radius."""
    def divergence(params):
        return box_kl(*params, *old).mean()

    def solve(_):
        def bisect(_, interval):
            lo, hi = interval
            mid = .5*(lo+hi)
            feasible = divergence(natural_interpolate(old, candidate, mid)) <= epsilon
            return jnp.where(feasible, mid, lo), jnp.where(feasible, hi, mid)
        lo, _ = jax.lax.fori_loop(
            0, iterations, bisect, (jnp.array(0.), jnp.array(1.))
        )
        return natural_interpolate(old, candidate, lo), lo

    projected, fraction = jax.lax.cond(
        divergence(candidate) <= epsilon,
        lambda _: (candidate, jnp.array(1.)),
        solve,
        None,
    )
    projected = jax.tree.map(
        lambda old_value, value: jnp.where(epsilon == 0., old_value, value),
        old,
        projected,
    )
    return projected, jnp.where(epsilon == 0., 0., fraction)


def log_components(params, actions):
    return component_log_prob(actions[None], params[0][None], params[1][None])[0]


def weighted_nll(params, actions, weights):
    ell = log_components(params, actions)
    log_mix = jax.scipy.special.logsumexp(ell, axis=0) - jnp.log(ell.shape[0])
    return -jnp.sum(weights*log_mix)


def responsibilities(params, actions):
    return jax.nn.softmax(log_components(params, actions), axis=0)


def sufficient_statistics(weighted_responsibilities, actions):
    mass = weighted_responsibilities.sum(-1)
    normalized = weighted_responsibilities / jnp.maximum(mass[:, None], 1e-30)
    mean = normalized @ actions
    variance = jnp.sum(
        normalized[..., None] * (actions[None]-mean[:, None])**2, axis=1
    )
    ess = mass**2 / jnp.maximum((weighted_responsibilities**2).sum(-1), 1e-30)
    return mass, mean, variance, ess


def generalized_m_step(params, weighted_responsibilities, actions, active, cfg):
    mass, mean, variance, ess = sufficient_statistics(
        weighted_responsibilities, actions
    )
    active = active & (mass > 1e-8) & (ess >= cfg["em_min_ess"])

    def objective(mu, log_std):
        return (
            .5*((mu-mean)**2+variance)*jnp.exp(-2.*log_std)
            + log_std + log_normalizer(mu, log_std)
        ).sum(-1)

    def step(_, current):
        mu, log_std = current
        grad_mu, grad_log_std = jax.grad(
            lambda m, l: objective(m, l).sum(), argnums=(0, 1)
        )(mu, log_std)
        delta_mu = -jnp.exp(2.*log_std)*grad_mu
        delta_log_std = -.5*grad_log_std
        initial = objective(mu, log_std)

        def backtrack(i, carry):
            best_mu, best_log_std, accepted = carry
            rate = cfg["em_step_size"] * .5**i
            proposal_mu = jnp.clip(mu + rate*delta_mu, -1., 1.)
            proposal_log_std = jnp.clip(log_std + rate*delta_log_std, -5., -1.)
            value = objective(proposal_mu, proposal_log_std)
            take = active & ~accepted & jnp.isfinite(value) & (value <= initial)
            return (
                jnp.where(take[:, None], proposal_mu, best_mu),
                jnp.where(take[:, None], proposal_log_std, best_log_std),
                accepted | take,
            )

        mu, log_std, _ = jax.lax.fori_loop(
            0, 10, backtrack, (mu, log_std, ~active)
        )
        return mu, log_std

    return jax.lax.fori_loop(0, cfg["em_m_steps"], step, params)


def fit_em(params, actions, weights, cfg, active=None, group_mass=None, steps=None):
    if active is None:
        active = jnp.ones(params[0].shape[0], dtype=bool)

    def step(_, current):
        ell = log_components(current, actions)
        if group_mass is None:
            resp = jax.nn.softmax(ell, axis=0)
        else:
            resp = jax.nn.softmax(jnp.where(active[:, None], ell, -jnp.inf), axis=0)
            resp = resp*group_mass[None]
        return generalized_m_step(current, weights[None]*resp, actions, active, cfg)

    count = cfg["em_steps"] if steps is None else steps
    return jax.lax.fori_loop(0, count, step, params)


def split_merge_candidate(base, actions, weights, i, j, k, cfg):
    mu, log_std = base
    fraction = cfg["smem_split_fraction"]
    merged_mu = .5*(mu[i]+mu[j])
    merged_variance = (
        .5*(jnp.exp(2.*log_std[i])+jnp.exp(2.*log_std[j]))
        + .25*(mu[i]-mu[j])**2
    )
    axis = jnp.argmax(log_std[k])
    offset = jax.nn.one_hot(axis, mu.shape[-1])*fraction*jnp.exp(log_std[k])
    child_log_std = (
        log_std[k]
        + jax.nn.one_hot(axis, mu.shape[-1])*.5*jnp.log1p(-fraction**2)
    )
    ids = jnp.stack((i, j, k))
    candidate_mu = mu.at[ids].set(
        jnp.stack((merged_mu, mu[k]+offset, mu[k]-offset))
    )
    candidate_log_std = log_std.at[ids].set(
        jnp.stack((.5*jnp.log(merged_variance), child_log_std, child_log_std))
    )
    candidate = (
        candidate_mu.clip(-1., 1.), candidate_log_std.clip(-5., -1.)
    )
    active = jnp.zeros(mu.shape[0], dtype=bool).at[ids].set(True)
    group_mass = responsibilities(base, actions)[ids].sum(0)
    candidate = fit_em(
        candidate, actions, weights, cfg, active, group_mass,
        cfg["smem_partial_steps"],
    )
    permutations = jnp.asarray(list(itertools.permutations(range(3))))
    costs = jax.vmap(
        lambda permutation: box_kl(
            candidate[0][ids[permutation]], candidate[1][ids[permutation]],
            mu[ids], log_std[ids],
        ).sum()
    )(permutations)
    order = ids[permutations[jnp.argmin(costs)]]
    candidate = tuple(value.at[ids].set(value[order]) for value in candidate)
    return fit_em(candidate, actions, weights, cfg)


def fit_teacher(old, actions, weights, step, cfg):
    initial_nll = weighted_nll(old, actions, weights)
    base = fit_em(old, actions, weights, cfg)
    projected, fraction = project(old, base, cfg["tr_kl"], cfg["tr_bisections"])
    value = weighted_nll(projected, actions, weights)
    take = jnp.isfinite(value) & (value <= initial_nll)
    best = jax.tree.map(lambda new, previous: jnp.where(take, new, previous), projected, old)
    best_value = jnp.where(take, value, initial_nll)

    def structural_search(carry):
        ell = log_components(base, actions)
        resp = jax.nn.softmax(ell, axis=0)
        weighted_resp = resp*weights[None]
        mass, _, _, ess = sufficient_statistics(weighted_resp, actions)
        eligible = (mass > 1e-8) & (ess >= cfg["em_min_ess"])
        overlap = weighted_resp @ resp.T
        pairs = jnp.triu(jnp.ones_like(overlap, dtype=bool), 1)
        pairs &= eligible[:, None] & eligible[None, :]
        scores, indices = jax.lax.top_k(
            jnp.where(pairs, overlap, -jnp.inf).ravel(), cfg["smem_candidates"]
        )
        local_weights = weighted_resp/jnp.maximum(mass[:, None], 1e-30)
        split_score = jnp.sum(
            local_weights*(jnp.log(jnp.maximum(local_weights, 1e-30))-ell), axis=-1
        )

        def attempt(n, current):
            current_best, current_value, current_fraction, selected = current
            i, j = indices[n]//len(mass), indices[n]%len(mass)
            allowed = eligible.at[i].set(False).at[j].set(False)
            k = jnp.argmax(jnp.where(allowed, split_score, -jnp.inf))

            def evaluate(_):
                candidate = split_merge_candidate(
                    base, actions, weights, i, j, k, cfg
                )
                projected_candidate, candidate_fraction = project(
                    old, candidate, cfg["tr_kl"], cfg["tr_bisections"]
                )
                nll = weighted_nll(projected_candidate, actions, weights)
                accept = jnp.isfinite(nll) & (nll < current_value)
                return (
                    jax.tree.map(
                        lambda new, previous: jnp.where(accept, new, previous),
                        projected_candidate,
                        current_best,
                    ),
                    jnp.where(accept, nll, current_value),
                    jnp.where(accept, candidate_fraction, current_fraction),
                    selected | accept,
                )

            return jax.lax.cond(
                jnp.isfinite(scores[n]) & jnp.any(allowed),
                evaluate,
                lambda _: current,
                None,
            )

        return jax.lax.fori_loop(0, cfg["smem_candidates"], attempt, carry)

    searched = step % cfg["smem_every"] == 0
    best, value, fraction, selected = jax.lax.cond(
        searched,
        structural_search,
        lambda current: current,
        (best, best_value, jnp.where(take, fraction, 0.), jnp.array(False)),
    )
    component_ess = sufficient_statistics(
        weights[None]*responsibilities(old, actions), actions
    )[3]
    return jax.tree.map(jax.lax.stop_gradient, best), dict(
        old_nll=initial_nll,
        teacher_nll=value,
        teacher_nll_gain=initial_nll-value,
        teacher_kl_bound=box_kl(*best, *old).mean(),
        tr_fraction=fraction,
        smem_attempted=searched.astype(float),
        smem_selected=selected.astype(float),
        component_ess_min=component_ess.min(),
        component_ess_mean=component_ess.mean(),
    )


def batch_nll(params, actions, weights):
    return jax.vmap(weighted_nll, in_axes=((0, 0), 0, 0))(
        params, actions, weights
    )


def distill_actor(
    actor_state, observations, latents, old, fitted, actions, weights, cfg
):
    batch_size, component_count, action_dim = old[0].shape
    observation_dim = observations.shape[-1]
    repeated_observations = jnp.broadcast_to(
        observations[:, None, :], (batch_size, component_count, observation_dim)
    )
    flat_observations = repeated_observations.reshape(-1, observation_dim)
    flat_latents = latents.reshape(-1, action_dim)
    initial_per_state_nll = batch_nll(old, actions, weights)
    initial_nll = initial_per_state_nll.mean()

    def actor_outputs(params):
        output = actor_state.apply_fn(
            {"params": params}, flat_observations, flat_latents, return_raw=True
        )
        return tuple(value.reshape(batch_size, component_count, action_dim) for value in output)

    def regression(params):
        mu, _, raw_log_std = actor_outputs(params)
        return (
            .5*((mu-fitted[0])*jnp.exp(-old[1]))**2
            + cfg["projection_scale_weight"]*(raw_log_std-fitted[1])**2
        ).mean()

    def step(_, carry):
        trial, best, best_nll, accepted, grad_norm = carry
        _, grad = jax.value_and_grad(regression)(trial.params)
        trial = trial.apply_gradients(grads=grad)
        mu, log_std, _ = actor_outputs(trial.params)
        nll = batch_nll((mu, log_std), actions, weights).mean()
        kl = box_kl(mu, log_std, *old).mean()
        finite = jnp.all(jnp.stack([
            jnp.all(jnp.isfinite(value))
            for value in jax.tree.leaves((trial.params, trial.opt_state))
        ]))
        take = finite & jnp.isfinite(nll) & jnp.isfinite(kl)
        take &= (nll <= best_nll) & (kl <= cfg["tr_kl"])
        best = jax.tree.map(lambda new, previous: jnp.where(take, new, previous), trial, best)
        return (
            trial,
            best,
            jnp.where(take, nll, best_nll),
            accepted | take,
            optax.global_norm(grad),
        )

    _, best, _, accepted, grad_norm = jax.lax.fori_loop(
        0,
        cfg["projection_steps"],
        step,
        (actor_state, actor_state, initial_nll, jnp.array(False), jnp.array(0.)),
    )
    mu, log_std, raw_log_std = actor_outputs(best.params)
    final_per_state_nll = batch_nll((mu, log_std), actions, weights)
    final_nll = final_per_state_nll.mean()
    per_state_kl = box_kl(mu, log_std, *old).mean(axis=1)
    final_kl = per_state_kl.mean()
    valid = jnp.isfinite(final_nll) & jnp.isfinite(final_kl)
    valid &= (final_nll <= initial_nll) & (final_kl <= cfg["tr_kl"])
    best = jax.tree.map(lambda new, previous: jnp.where(valid, new, previous), best, actor_state)
    best = best.replace(step=actor_state.step+1)
    old_raw_log_std = actor_outputs(actor_state.params)[2]
    log_std = jnp.where(valid, log_std, old[1])
    raw_log_std = jnp.where(valid, raw_log_std, old_raw_log_std)
    final_per_state_nll = jnp.where(valid, final_per_state_nll, initial_per_state_nll)
    final_kl = jnp.where(valid, final_kl, 0.)
    per_state_kl = jnp.where(valid, per_state_kl, 0.)
    return best, dict(
        actor_loss=final_per_state_nll.mean(),
        actor_nll_gain=initial_nll-final_per_state_nll.mean(),
        actor_kl_bound=final_kl,
        actor_kl_state_max=per_state_kl.max(),
        actor_accepted=(accepted & valid).astype(float),
        final_actor_rejected=(accepted & ~valid).astype(float),
        projection_loss=regression(best.params),
        actor_grad_norm=grad_norm,
        actor_log_std_mean=log_std.mean(),
        actor_log_std_min=log_std.min(),
        actor_log_std_max=log_std.max(),
        actor_std_mean=jnp.exp(log_std).mean(),
        actor_std_min=jnp.exp(log_std).min(),
        actor_std_max=jnp.exp(log_std).max(),
        raw_scale_clipped_fraction=((raw_log_std < -5.) | (raw_log_std > -1.)).mean(),
    )


def update_actor(
    actor_state,
    qf_state,
    observations,
    key,
    z_atoms,
    num_policy_samples,
    proposals_per_policy_sample,
    proposal_sampling_mode,
    proposal_std,
    density_correction,
    density_beta,
    temperature,
    source_q_eval,
):
    """Run one state-conditioned SMEM+TR actor update."""
    if proposals_per_policy_sample != 1 or proposal_sampling_mode != "exact":
        raise ValueError("SMEM+TR comparison requires one exact draw per component")
    if not density_correction:
        raise ValueError("SMEM+TR comparison requires proposal-density correction")
    cfg = DEFAULTS
    key, latent_key, proposal_key, dropout_key = jax.random.split(key, 4)
    batch_size, observation_dim = observations.shape
    action_dim = actor_state.params["mu"]["bias"].shape[0]
    z_key, eps_key = jax.random.split(latent_key)
    latents = jax.random.normal(
        z_key, (batch_size, num_policy_samples, action_dim), dtype=observations.dtype
    )
    repeated_observations = jnp.broadcast_to(
        observations[:, None, :], (batch_size, num_policy_samples, observation_dim)
    )
    old_output = actor_state.apply_fn(
        {"params": actor_state.params},
        repeated_observations.reshape(-1, observation_dim),
        latents.reshape(-1, action_dim),
        return_raw=True,
    )
    old_mu, old_log_std, _ = (
        value.reshape(batch_size, num_policy_samples, action_dim)
        for value in old_output
    )
    old = jax.tree.map(jax.lax.stop_gradient, (old_mu, old_log_std))
    student_actions = sample_box(eps_key, old_mu, old_log_std)
    proposal = ConditionalGaussianProposal(old_mu, old_log_std, proposal_std)
    proposals, proposal_actions, _ = proposal.sample(
        proposal_key, proposals_per_policy_sample, proposal_sampling_mode
    )
    proposals = jax.lax.stop_gradient(
        proposals.reshape(batch_size, num_policy_samples, action_dim)
    )
    proposal_actions = jax.lax.stop_gradient(
        proposal_actions.reshape(batch_size, num_policy_samples, action_dim)
    )
    proposal_observations = jnp.broadcast_to(
        observations[:, None, :], (batch_size, num_policy_samples, observation_dim)
    )
    source_distributions = qf_state.apply_fn(
        {"params": qf_state.params, "batch_stats": qf_state.batch_stats},
        proposal_observations.reshape(-1, observation_dim),
        proposals.reshape(-1, action_dim),
        rngs={"dropout": dropout_key},
        train=False,
    ).reshape(2, batch_size, num_policy_samples, -1)
    source_qs = critic_expectation(source_distributions, z_atoms)
    if source_q_eval == "mean":
        source_q = source_qs.mean(axis=0)
    elif source_q_eval == "min":
        source_q = source_qs.min(axis=0)
    else:
        raise ValueError(f"Unknown source_q_eval: {source_q_eval}")
    source_q = jax.lax.stop_gradient(source_q)
    proposal_log_density = proposal.log_prob(proposal_actions)
    source_weights = jax.lax.stop_gradient(
        jax.nn.softmax(
            source_q/temperature - density_beta*proposal_log_density, axis=-1
        )
    )

    fitted, fit_metrics = jax.vmap(
        lambda old_params, x, w: fit_teacher(
            old_params, x, w, actor_state.step, cfg
        ),
        in_axes=((0, 0), 0, 0),
    )(old, proposal_actions, source_weights)
    actor_state, actor_metrics = distill_actor(
        actor_state,
        observations,
        latents,
        old,
        fitted,
        proposal_actions,
        source_weights,
        cfg,
    )
    ess = 1./jnp.square(source_weights).sum(-1)
    metrics = {
        **actor_metrics,
        **{name: value.mean() for name, value in fit_metrics.items()},
        "smem_selected_state_fraction": fit_metrics["smem_selected"].mean(),
        "source_ess_absolute": ess.mean(),
        "source_ess_fraction": (ess/num_policy_samples).mean(),
        "source_ess_min": ess.min(),
        "max_source_weight": source_weights.max(-1).mean(),
        "source_q_mean": source_q.mean(),
        "source_q_std": source_q.std(-1).mean(),
        "teacher_log_density_mean": proposal_log_density.mean(),
        "teacher_log_density_std": proposal_log_density.std(-1).mean(),
        "teacher_log_density_max": proposal_log_density.max(),
        "actor_mu_abs_max": jnp.abs(old_mu).max(),
        "teacher_action_saturation_fraction": jnp.mean(jnp.abs(proposals) > .99),
        "student_action_saturation_fraction": jnp.mean(jnp.abs(student_actions) > .99),
        "policy_spread_l2": jnp.linalg.norm(student_actions.std(1), axis=-1).mean(),
        "temperature": jnp.asarray(temperature),
    }
    return actor_state, metrics, key
