"""Fixed-weight split/merge generalized EM and box-Gaussian KL projection.

This is an adaptation, not the original variable-weight SMEM or a reproduction
of differentiable Gaussian trust-region layers. See SMEM_TR.md for equations.
Only stopped SNIS particles/weights enter fitting; no target labels or scores.
"""
import itertools

import jax
import jax.numpy as jnp
import optax

from upstream.box_gaussian import component_log_prob, log_normalizer


DEFAULTS = dict(em_steps=2, em_m_steps=4, em_step_size=.5, em_min_ess=2.,
                smem_every=20, smem_candidates=2, smem_partial_steps=2,
                smem_split_fraction=.5, tr_kl=.05, tr_bisections=28,
                projection_steps=10, projection_scale_weight=1.)


def add_arguments(parser):
    for name, value in DEFAULTS.items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(value), default=value)


def cli_arguments(args):
    return [item for name in DEFAULTS
            for item in ("--" + name.replace("_", "-"), str(getattr(args, name)))]


def validate_config(cfg):
    import math
    if cfg["components"] < 3:
        raise ValueError("Split/merge requires at least three components")
    for name in ("em_steps", "em_m_steps", "smem_every", "smem_candidates",
                 "smem_partial_steps", "tr_bisections", "projection_steps"):
        if not isinstance(cfg[name], int) or cfg[name] < 1:
            raise ValueError(name + " must be a positive integer")
    for name in ("em_step_size", "em_min_ess", "projection_scale_weight"):
        if not math.isfinite(cfg[name]) or cfg[name] <= 0:
            raise ValueError(name + " must be finite and positive")
    if not math.isfinite(cfg["tr_kl"]) or cfg["tr_kl"] < 0:
        raise ValueError("tr_kl must be finite and nonnegative")
    if not 0 < cfg["smem_split_fraction"] < 1:
        raise ValueError("smem_split_fraction must lie in (0, 1)")
    if cfg["smem_candidates"] > cfg["components"] * (cfg["components"] - 1) // 2:
        raise ValueError("Too many merge candidates")
    if (cfg["log_std_min"], cfg["log_std_max"]) != (-5., -1.):
        raise ValueError("This actor uses fixed log_std bounds [-5, -1]")


def box_moments(mu, ls):
    """Mean and variance of the normalized diagonal truncated distribution."""
    std = jnp.exp(ls)
    a, b = (-1. - mu) / std, (1. - mu) / std
    pa = jnp.exp(-.5 * a*a) / jnp.sqrt(2.*jnp.pi)
    pb = jnp.exp(-.5 * b*b) / jnp.sqrt(2.*jnp.pi)
    normalizer = jnp.exp(log_normalizer(mu, ls))
    shift = (pa - pb) / normalizer
    mean = mu + std * shift
    variance = std**2 * (1. + (a*pa - b*pb) / normalizer - shift**2)
    return mean, jnp.maximum(variance, 0.)


def box_kl(mu, ls, ref_mu, ref_ls):
    """Exact component KL(new || reference), up to floating-point error."""
    mean, variance = box_moments(mu, ls)
    value = (ref_ls - ls + log_normalizer(ref_mu, ref_ls) - log_normalizer(mu, ls)
             + .5 * ((variance + (mean-ref_mu)**2)*jnp.exp(-2.*ref_ls)
                     - (variance + (mean-mu)**2)*jnp.exp(-2.*ls)))
    return jnp.maximum(value.sum(-1), 0.)


def natural_interpolate(old, candidate, t):
    old_mu, old_ls = old
    mu, ls = candidate
    old_precision, precision = jnp.exp(-2.*old_ls), jnp.exp(-2.*ls)
    mixed_precision = (1.-t)*old_precision + t*precision
    mixed_mu = ((1.-t)*old_precision*old_mu + t*precision*mu) / mixed_precision
    return mixed_mu, -.5*jnp.log(mixed_precision)


def project(old, candidate, epsilon, iterations=28):
    """Minimize mean KL(projected_k || candidate_k), subject to a joint KL bound.

    The joint distribution includes a uniform component index. Its KL is an
    upper bound on marginal mixture KL. A common dual multiplier gives natural
    parameter interpolation for every truncated Gaussian, including its mean.
    This stopped teacher projection is not differentiated through.
    """
    def divergence(params):
        return box_kl(*params, *old).mean()

    def solve(_):
        def bisect(_, interval):
            lo, hi = interval
            mid = (lo+hi)*.5
            feasible = divergence(natural_interpolate(old, candidate, mid)) <= epsilon
            return jnp.where(feasible, mid, lo), jnp.where(feasible, hi, mid)
        lo, _ = jax.lax.fori_loop(0, iterations, bisect, (jnp.array(0.), jnp.array(1.)))
        return natural_interpolate(old, candidate, lo), lo

    projected, fraction = jax.lax.cond(
        divergence(candidate) <= epsilon, lambda _: (candidate, jnp.array(1.)), solve, None)
    # Return exactly the old parameters for a zero-radius region.
    projected = jax.tree.map(lambda a, b: jnp.where(epsilon == 0., a, b), old, projected)
    return projected, jnp.where(epsilon == 0., 0., fraction)


def log_components(params, x):
    return component_log_prob(x[None], params[0][None], params[1][None])[0]


def weighted_nll(params, x, w):
    ell = log_components(params, x)
    return -jnp.sum(w * (jax.scipy.special.logsumexp(ell, axis=0)-jnp.log(ell.shape[0])))


def responsibilities(params, x):
    return jax.nn.softmax(log_components(params, x), axis=0)


def sufficient_statistics(u, x):
    mass = u.sum(-1)
    normalized = u / jnp.maximum(mass[:, None], 1e-30)
    mean = normalized @ x
    # Center first: avoid cancellation when narrow components are far from zero.
    variance = jnp.sum(normalized[..., None]*(x[None]-mean[:, None])**2, axis=1)
    ess = mass**2 / jnp.maximum((u*u).sum(-1), 1e-30)
    return mass, mean, variance, ess


def generalized_m_step(params, u, x, active, cfg):
    mass, mean, variance, ess = sufficient_statistics(u, x)
    active = active & (mass > 1e-8) & (ess >= cfg["em_min_ess"])

    def objective(mu, ls):
        return (.5*((mu-mean)**2+variance)*jnp.exp(-2.*ls)
                + ls + log_normalizer(mu, ls)).sum(-1)

    def step(_, params):
        mu, ls = params
        gm, gl = jax.grad(lambda m, l: objective(m, l).sum(), argnums=(0, 1))(mu, ls)
        # Positive diagonal preconditioning; exact Q controls acceptance.
        dm, dl = -jnp.exp(2.*ls)*gm, -.5*gl
        initial = objective(mu, ls)
        def backtrack(i, carry):
            best_mu, best_ls, accepted = carry
            rate = cfg["em_step_size"] * .5**i
            m = jnp.clip(mu+rate*dm, -1., 1.)
            l = jnp.clip(ls+rate*dl, -5., -1.)
            value = objective(m, l)
            take = active & ~accepted & jnp.isfinite(value) & (value <= initial)
            return (jnp.where(take[:, None], m, best_mu),
                    jnp.where(take[:, None], l, best_ls), accepted | take)
        m, l, _ = jax.lax.fori_loop(0, 10, backtrack, (mu, ls, ~active))
        return m, l
    return jax.lax.fori_loop(0, cfg["em_m_steps"], step, params)


def fit_em(params, x, w, cfg, active=None, group_mass=None, steps=None):
    if active is None:
        active = jnp.ones(params[0].shape[0], dtype=bool)
    def step(_, params):
        ell = log_components(params, x)
        if group_mass is None:
            r = jax.nn.softmax(ell, axis=0)
        else:
            # Partial EM preserves each particle's old responsibility mass of
            # the three affected components, as in SMEM's partial E-step.
            r = jax.nn.softmax(jnp.where(active[:, None], ell, -jnp.inf), axis=0)
            r = r * group_mass[None]
        return generalized_m_step(params, w[None]*r, x, active, cfg)
    return jax.lax.fori_loop(0, cfg["em_steps"] if steps is None else steps, step, params)


def split_merge_candidate(base, x, w, i, j, k, cfg):
    mu, ls = base
    fraction = cfg["smem_split_fraction"]
    # Moment-preserving initializers for the underlying untruncated Gaussians;
    # the exact truncated likelihood is used for all subsequent fitting.
    merged_mu = .5*(mu[i]+mu[j])
    merged_var = .5*(jnp.exp(2.*ls[i])+jnp.exp(2.*ls[j])) + .25*(mu[i]-mu[j])**2
    axis = jnp.argmax(ls[k])
    offset = jax.nn.one_hot(axis, mu.shape[-1])*fraction*jnp.exp(ls[k])
    child_ls = ls[k] + jax.nn.one_hot(axis, mu.shape[-1])*.5*jnp.log1p(-fraction**2)
    ids = jnp.stack((i, j, k))
    m = mu.at[ids].set(jnp.stack((merged_mu, mu[k]+offset, mu[k]-offset)))
    l = ls.at[ids].set(jnp.stack((.5*jnp.log(merged_var), child_ls, child_ls)))
    candidate = (m.clip(-1., 1.), l.clip(-5., -1.))
    active = jnp.zeros(mu.shape[0], dtype=bool).at[ids].set(True)
    group_mass = responsibilities(base, x)[ids].sum(0)
    candidate = fit_em(candidate, x, w, cfg, active, group_mass, cfg["smem_partial_steps"])
    # Component labels are arbitrary; choose the least costly permutation of
    # the three new components before constraining their movement.
    permutations = jnp.array(list(itertools.permutations(range(3))))
    costs = jax.vmap(lambda p: box_kl(candidate[0][ids[p]], candidate[1][ids[p]],
                                     mu[ids], ls[ids]).sum())(permutations)
    order = ids[permutations[jnp.argmin(costs)]]
    candidate = tuple(a.at[ids].set(a[order]) for a in candidate)
    return fit_em(candidate, x, w, cfg)


def fit_teacher(old, x, w, step, cfg):
    initial_nll = weighted_nll(old, x, w)
    base = fit_em(old, x, w, cfg)
    projected, fraction = project(old, base, cfg["tr_kl"], cfg["tr_bisections"])
    value = weighted_nll(projected, x, w)
    take = jnp.isfinite(value) & (value <= initial_nll)
    best = jax.tree.map(lambda a, b: jnp.where(take, a, b), projected, old)
    best_value = jnp.where(take, value, initial_nll)

    def structural_search(carry):
        ell = log_components(base, x)
        r = jax.nn.softmax(ell, axis=0)
        u = r*w[None]
        mass, _, _, ess = sufficient_statistics(u, x)
        eligible = (mass > 1e-8) & (ess >= cfg["em_min_ess"])
        overlap = (r*w[None]) @ r.T
        pairs = jnp.triu(jnp.ones_like(overlap, dtype=bool), 1)
        pairs &= eligible[:, None] & eligible[None, :]
        scores, indices = jax.lax.top_k(jnp.where(pairs, overlap, -jnp.inf).ravel(),
                                      cfg["smem_candidates"])
        local_w = u/jnp.maximum(mass[:, None], 1e-30)
        # Finite-sample local log-density mismatch ranking (a SMEM heuristic,
        # not the continuous KL of an empirical measure against a Gaussian).
        split_score = jnp.sum(local_w*(jnp.log(jnp.maximum(local_w, 1e-30))-ell), axis=-1)
        def attempt(n, carry):
            best, best_value, best_fraction, selected = carry
            i, j = indices[n]//len(mass), indices[n]%len(mass)
            allowed = eligible.at[i].set(False).at[j].set(False)
            k = jnp.argmax(jnp.where(allowed, split_score, -jnp.inf))
            def evaluate(_):
                candidate = split_merge_candidate(base, x, w, i, j, k, cfg)
                new, f = project(old, candidate, cfg["tr_kl"], cfg["tr_bisections"])
                nll = weighted_nll(new, x, w)
                accept = jnp.isfinite(nll) & (nll < best_value)
                return (jax.tree.map(lambda a, b: jnp.where(accept, a, b), new, best),
                        jnp.where(accept, nll, best_value),
                        jnp.where(accept, f, best_fraction), selected | accept)
            return jax.lax.cond(jnp.isfinite(scores[n]) & jnp.any(allowed), evaluate,
                                lambda _: carry, None)
        return jax.lax.fori_loop(0, cfg["smem_candidates"], attempt, carry)

    searched = step % cfg["smem_every"] == 0
    best, value, fraction, selected = jax.lax.cond(
        searched, structural_search, lambda c: c,
        (best, best_value, jnp.where(take, fraction, 0.), jnp.array(False)))
    ess = sufficient_statistics(w[None]*responsibilities(old, x), x)[3]
    return jax.tree.map(jax.lax.stop_gradient, best), dict(
        old_nll=initial_nll, teacher_nll=value, teacher_nll_gain=initial_nll-value,
        teacher_kl_bound=box_kl(*best, *old).mean(), tr_fraction=fraction,
        smem_attempted=searched.astype(float), smem_selected=selected.astype(float),
        component_ess_min=ess.min(), component_ess_mean=ess.mean())


def distill_actor(state, z, old, fitted, x, w, cfg):
    initial_nll = weighted_nll(old, x, w)
    def regression(params):
        mu, _, raw_ls = state.apply_fn({"params": params}, z, return_raw=True)
        return (.5*((mu-fitted[0])*jnp.exp(-old[1]))**2
                + cfg["projection_scale_weight"]*(raw_ls-fitted[1])**2).mean()

    def step(_, carry):
        trial, best, best_nll, accepted, grad_norm = carry
        _, grad = jax.value_and_grad(regression)(trial.params)
        trial = trial.apply_gradients(grads=grad)
        params = trial.apply_fn({"params": trial.params}, z)
        nll = weighted_nll(params, x, w)
        kl = box_kl(*params, *old).mean()
        finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(v))
                         for v in jax.tree.leaves((trial.params, trial.opt_state))]))
        take = finite & jnp.isfinite(nll) & jnp.isfinite(kl)
        take &= (nll <= best_nll) & (kl <= cfg["tr_kl"])
        best = jax.tree.map(lambda a, b: jnp.where(take, a, b), trial, best)
        return trial, best, jnp.where(take, nll, best_nll), accepted | take, optax.global_norm(grad)

    _, best, nll, accepted, grad_norm = jax.lax.fori_loop(
        0, cfg["projection_steps"], step, (state, state, initial_nll, jnp.array(False), jnp.array(0.)))
    # Outer step advances even on rejection; Adam moments/counter come from the
    # selected iterate (or the unchanged state). Rejection must not freeze the
    # structural-search schedule.
    mu, ls, raw_ls = best.apply_fn({"params": best.params}, z, return_raw=True)
    # Validate the final evaluation path as well as the inner optimization
    # path. On GPU the two compiled GEMM contexts can differ numerically;
    # narrow components magnify output differences in the KL calculation.
    final_nll = weighted_nll((mu, ls), x, w)
    final_kl = box_kl(mu, ls, *old).mean()
    valid = jnp.isfinite(final_nll) & jnp.isfinite(final_kl)
    valid &= (final_nll <= initial_nll) & (final_kl <= cfg["tr_kl"])
    best = jax.tree.map(lambda a, b: jnp.where(valid, a, b), best, state)
    best = best.replace(step=state.step+1)
    old_raw_ls = state.apply_fn({"params": state.params}, z, return_raw=True)[2]
    ls = jnp.where(valid, ls, old[1])
    raw_ls = jnp.where(valid, raw_ls, old_raw_ls)
    nll = jnp.where(valid, final_nll, initial_nll)
    return best, dict(loss=nll, actor_nll_gain=initial_nll-nll,
        actor_kl_bound=jnp.where(valid, final_kl, 0.), actor_accepted=(accepted & valid).astype(float),
        final_actor_rejected=(accepted & ~valid).astype(float),
        projection_loss=regression(best.params), grad_norm=grad_norm,
        mean_log_std=ls.mean(), raw_scale_clipped_fraction=((raw_ls < -5.) | (raw_ls > -1.)).mean())


def make_smem_update(target, cfg):
    from core import teacher
    validate_config(cfg)
    @jax.jit
    def update(state, z, key):
        key, tkey = jax.random.split(key)
        old = jax.tree.map(jax.lax.stop_gradient, state.apply_fn({"params": state.params}, z))
        x, w, sampler_info = teacher(tkey, *old, target, cfg)
        x = jax.lax.stop_gradient(x.reshape(-1, old[0].shape[-1]))
        w = jax.lax.stop_gradient(w.reshape(-1)/cfg["populations"])
        fitted, fit_info = fit_teacher(old, x, w, state.step, cfg)
        new, actor_info = distill_actor(state, z, old, fitted, x, w, cfg)
        return new, key, dict(sampler_info, **fit_info, **actor_info)
    return update
