"""Direct box-GMM NLL with SNIS or population annealing / MALA teachers."""
from functools import partial
import math

import flax.linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax

from upstream.box_gaussian import mixture_log_prob, sample_box
from upstream.distillation import direct_gmm_nll


class Actor(nn.Module):
    """Upstream SemiImplicitActor, observation-free, unchanged default heads."""
    @nn.compact
    def __call__(self, z, return_raw=False):
        init = lambda scale: nn.initializers.variance_scaling(scale, "fan_avg", "uniform")
        x = z
        for _ in range(2):
            x = nn.gelu(nn.Dense(256, kernel_init=init(1.0))(x))
        mu = nn.Dense(2, kernel_init=init(1e-4), name="mu")(x)
        logstd = nn.Dense(2, kernel_init=init(0.0),
                         bias_init=nn.initializers.constant(-1.), name="log_std")(x)
        outputs = (jnp.tanh(mu), jnp.clip(logstd, -5., -1.))
        return (*outputs, logstd) if return_raw else outputs


def initialize(seed, components=64, lr=3e-4):
    actor = Actor()
    key, ikey, zkey = jax.random.split(jax.random.PRNGKey(seed), 3)
    # Finite prior is an explicit toy adaptation: exact evaluation of q, no KDE.
    z = jax.random.normal(zkey, (components, 2))
    params = actor.init(ikey, z)["params"]
    state = TrainState.create(apply_fn=actor.apply, params=params, tx=optax.adam(lr))
    return state, z, key


def target_parameters():
    """Exact seed/layout/scales used in the supplied GMM40 benchmark, x=50a."""
    import torch
    with torch.random.fork_rng():
        torch.manual_seed(0)
        centers = ((torch.rand(40, 2) - .5) * 80.).numpy() / 50.
        scales = torch.nn.functional.softplus(torch.ones(40, 2)).numpy() / 50.
    return jnp.asarray(centers), jnp.log(jnp.asarray(scales))


def target_logp(x, target):
    # Each component is conditioned on the box. Omitted original tail < 1e-14.
    mu, ls = target
    shape = x.shape[:-1]
    return mixture_log_prob(x.reshape(1, -1, 2), mu[None], ls[None]).reshape(shape)


def mixture_sample(key, mu, ls, shape):
    ikey, nkey = jax.random.split(key)
    ids = jax.random.randint(ikey, shape, 0, mu.shape[0])
    return sample_box(nkey, mu[ids], ls[ids])


def reference_sample(key, mu, ls, shape, defensive):
    qkey, ukey, bkey = jax.random.split(key, 3)
    q = mixture_sample(qkey, mu, ls, shape)
    uniform = jax.random.uniform(ukey, shape + (2,), minval=-1.+1e-6, maxval=1.-1e-6)
    choose = jax.random.uniform(bkey, shape) < defensive
    return jnp.where(choose[..., None], uniform, q)


def reference_logp(x, mu, ls, defensive):
    shape = x.shape[:-1]
    lq = mixture_log_prob(x.reshape(1, -1, 2), mu[None], ls[None]).reshape(shape)
    if defensive == 0.:
        return lq
    return jnp.logaddexp(jnp.log1p(-defensive) + lq,
                        jnp.log(defensive) - 2.*jnp.log(2.))


def log_jacobian(y):
    return jnp.sum(2.*(jnp.log(2.) - y - jax.nn.softplus(-2.*y)), axis=-1)


def bridge_value_grad(y, t, mu, ls, target, defensive):
    def total(y):
        a = jnp.tanh(y)
        lp = (1.-t)*reference_logp(a, mu, ls, defensive) + t*target_logp(a, target)
        lp = lp + log_jacobian(y)
        return jnp.sum(lp), lp
    (_, value), grad = jax.value_and_grad(total, has_aux=True)(y)
    return value, grad


def mala_step(key, y, t, mu, ls, target, defensive, step_size):
    """Full asymmetric MH correction in unconstrained y coordinates."""
    nkey, ukey = jax.random.split(key)
    value, grad = bridge_value_grad(y, t, mu, ls, target, defensive)
    forward_mean = y + .5*step_size**2*grad
    proposal = forward_mean + step_size*jax.random.normal(nkey, y.shape)
    new_value, new_grad = bridge_value_grad(proposal, t, mu, ls, target, defensive)
    reverse_mean = proposal + .5*step_size**2*new_grad
    log_forward = -.5*jnp.sum(((proposal-forward_mean)/step_size)**2, axis=-1)
    log_reverse = -.5*jnp.sum(((y-reverse_mean)/step_size)**2, axis=-1)
    log_accept = new_value - value + log_reverse - log_forward
    accept = jnp.isfinite(log_accept) & (jnp.log(jax.random.uniform(ukey, value.shape)) < log_accept)
    result = jnp.where(accept[..., None], proposal, y)
    jump = jnp.sum((jnp.tanh(result)-jnp.tanh(y))**2, axis=-1).mean()
    return result, accept.mean(), jump


def systematic_indices(key, weights):
    count = weights.shape[-1]
    positions = (jax.random.uniform(key, ()) + jnp.arange(count))/count
    cdf = jnp.cumsum(weights).at[-1].set(1.)
    return jnp.searchsorted(cdf, positions, side="right").clip(0, count-1)


def teacher(key, mu, ls, target, cfg):
    batch, count = cfg["populations"], cfg["particles"]
    smc = cfg["method"] == "smc_mala"
    defensive = cfg["defensive"] if smc else 0.
    key, initkey = jax.random.split(key)
    x = reference_sample(initkey, mu, ls, (batch, count), defensive)
    initial_logratio = target_logp(x, target)-reference_logp(x, mu, ls, defensive)
    initial_w = jax.nn.softmax(initial_logratio, axis=-1)
    initial_ess = (1./jnp.sum(initial_w**2, axis=-1)).mean()
    if not smc:
        return x, initial_w, dict(teacher_ess=initial_ess, initial_is_ess=initial_ess,
            max_weight=initial_w.max(-1).mean(), mala_acceptance=jnp.array(0.),
            mala_jump_sq=jnp.array(0.), resampling_events=jnp.array(0.))

    schedule = jnp.linspace(0., 1., cfg["stages"]+1)**2
    y = jnp.arctanh(x.clip(-1.+1e-6, 1.-1e-6))
    logw = jnp.full((batch, count), -jnp.log(float(count)))

    def stage(carry, ts):
        y, logw, key = carry
        t0, t1 = ts
        a = jnp.tanh(y)
        ratio = target_logp(a, target)-reference_logp(a, mu, ls, defensive)
        logw += (t1-t0)*ratio
        logw -= jax.scipy.special.logsumexp(logw, axis=-1, keepdims=True)
        w = jnp.exp(logw)
        ess = 1./jnp.sum(w*w, axis=-1)
        do_resample = ess < cfg["resample_ess"]*count
        key, rkey = jax.random.split(key)
        ids = jax.vmap(systematic_indices)(jax.random.split(rkey, batch), w)
        resampled = jnp.take_along_axis(y, ids[..., None], axis=1)
        y = jnp.where(do_resample[:, None, None], resampled, y)
        logw = jnp.where(do_resample[:, None], -jnp.log(float(count)), logw)
        step = cfg["mala_step"] / jnp.sqrt(1.+15.*t1)

        def move(carry, _):
            y, key = carry
            key, mkey = jax.random.split(key)
            y, accept, jump = mala_step(mkey, y, t1, mu, ls, target, defensive, step)
            return (y, key), (accept, jump)
        (y, key), (accept, jump) = jax.lax.scan(move, (y, key), None, length=cfg["moves"])
        return (y, logw, key), (accept.mean(), jump.mean(), do_resample.mean(), ess.mean())

    (y, logw, key), stats = jax.lax.scan(stage, (y, logw, key), (schedule[:-1], schedule[1:]))
    w = jax.nn.softmax(logw, axis=-1)
    return jnp.tanh(y), w, dict(teacher_ess=(1./jnp.sum(w*w, -1)).mean(),
        initial_is_ess=initial_ess, max_weight=w.max(-1).mean(),
        mala_acceptance=stats[0].mean(), mala_jump_sq=stats[1].mean(),
        resampling_events=stats[2].sum(), intermediate_ess=stats[3].mean())


def make_update(target, cfg):
    if cfg["method"] == "snis_smem_tr_aux":
        from smem_repair import make_smem_repair_update
        return make_smem_repair_update(target, cfg)
    if cfg["method"] == "snis_smem_tr":
        from smem_tr import make_smem_update
        return make_smem_update(target, cfg)
    if cfg["method"] not in ("snis", "smc_mala"):
        raise ValueError("Unknown method: " + cfg["method"])
    @jax.jit
    def update(state, z, key):
        key, tkey = jax.random.split(key)
        mu, ls = state.apply_fn({"params": state.params}, z)
        x, w, info = teacher(tkey, jax.lax.stop_gradient(mu), jax.lax.stop_gradient(ls), target, cfg)
        x, w = jax.lax.stop_gradient(x), jax.lax.stop_gradient(w)
        def loss(params):
            mu, ls = state.apply_fn({"params": params}, z)
            return direct_gmm_nll(mu[None], ls[None], x.reshape(1, -1, 2),
                                  w.reshape(1, -1)/cfg["populations"])[0]
        value, grad = jax.value_and_grad(loss)(state.params)
        state = state.apply_gradients(grads=grad)
        return state, key, dict(info, loss=value, grad_norm=optax.global_norm(grad),
                               mean_log_std=ls.mean())
    return update


def defaults(method="snis", seed=0):
    cfg = dict(method=method, seed=seed, components=64, particles=64, populations=8,
        lr=3e-4, steps=4000, eval_every=250, eval_samples=16384,
        stages=16, moves=2, mala_step=.2, resample_ess=.5, defensive=.1,
        coordinate_scale=50., density_beta=1., temperature=1.,
        hidden_dims=[256, 256], log_std_min=-5., log_std_max=-1.,
        initial_log_std=-1., latent_prior="fixed_gaussian_codebook",
        target="GMM40-seed0-box-normalized", phase="pilot")
    if method in ("snis_smem_tr", "snis_smem_tr_aux"):
        from smem_tr import DEFAULTS
        cfg.update(DEFAULTS)
    if method == "snis_smem_tr_aux":
        cfg.update(projection_steps=1, actor_backtracks=10,
                   projection_objective="em_auxiliary")
    return cfg
