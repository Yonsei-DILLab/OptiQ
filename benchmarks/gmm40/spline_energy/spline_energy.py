"""One bounded energy circuit: Q, V, density and exact one-shot sampling.

F(a)=sum_r exp(h_r) prod_j linear_exp_knots(l[r,j,:],a_j)
Q=log F, V=log integral F, pi=F/integral F. Temperature is one.
This state-free bandit model is a sum-product network with trainable leaves.
"""
from functools import partial
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from flax.training.train_state import TrainState
import optax
import numpy as np
from core import target_logp


def log_integrals(leaves):
    dx = 2. / (leaves.shape[-1] - 1)
    return logsumexp(jnp.logaddexp(leaves[..., :-1], leaves[..., 1:]), -1) + jnp.log(dx/2.)


def circuit_partition(params):
    leaf_i = log_integrals(params["leaves"])
    root_mass = params["height"] + leaf_i.sum(-1)
    return logsumexp(root_mass), root_mass, leaf_i


def circuit_q(params, actions):
    shape = actions.shape[:-1]
    actions = actions.reshape(-1, 2)
    leaves = params["leaves"]
    bins = leaves.shape[-1] - 1
    u = ((actions + 1.) * bins / 2.).clip(0., float(bins))
    idx = jnp.minimum(jnp.floor(u).astype(jnp.int32), bins-1)
    fraction = u - idx
    left = jnp.take_along_axis(leaves[None], idx[:, None, :, None], axis=3)[..., 0]
    right = jnp.take_along_axis(leaves[None], (idx+1)[:, None, :, None], axis=3)[..., 0]
    leaf_value = jnp.logaddexp(left + jnp.log1p(-fraction[:, None]),
                               right + jnp.log(fraction[:, None]))
    result = logsumexp(params["height"][None] + leaf_value.sum(-1), axis=1)
    inside = jnp.all((actions >= -1.) & (actions <= 1.), -1)
    return jnp.where(inside, result, -jnp.inf).reshape(shape)


def circuit_logp(params, actions):
    return circuit_q(params, actions) - circuit_partition(params)[0]


def circuit_sample(params, key, count, parallel_root=False):
    """Root category, leaf interval, analytic quadratic inverse: no recurrence."""
    rkey, bkey, ukey = jax.random.split(key, 3)
    logz, mass, _ = circuit_partition(params)
    cdf = jnp.cumsum(jnp.exp(mass-logz)).at[-1].set(1.)
    root_uniform = jax.random.uniform(rkey, (count,))
    if parallel_root:
        # Equivalent to searchsorted(side='right'), without a binary-search loop.
        roots = jnp.sum(cdf[None] <= root_uniform[:, None], axis=-1)
    else:
        roots = jnp.searchsorted(cdf, root_uniform, side="right")
    roots = roots.clip(0, len(mass)-1)
    leaves = params["leaves"]
    scaled = jnp.exp(leaves - leaves.max(-1, keepdims=True))
    area = .5 * (scaled[..., :-1] + scaled[..., 1:])
    leaf_cdf = jnp.cumsum(area, -1) / area.sum(-1, keepdims=True)
    leaf_cdf = leaf_cdf.at[..., -1].set(1.)
    uniform = jax.random.uniform(bkey, (count, 2))
    bins = jnp.sum(leaf_cdf[roots] <= uniform[..., None], -1).clip(0, area.shape[-1]-1)
    selected = scaled[roots]
    y0 = jnp.take_along_axis(selected, bins[..., None], -1)[..., 0]
    y1 = jnp.take_along_axis(selected, (bins+1)[..., None], -1)[..., 0]
    within = jax.random.uniform(ukey, (count, 2))
    mass_within = within * .5 * (y0+y1)
    # Rationalized quadratic root also has the correct limit y1 == y0.
    discriminant = jnp.maximum(y0*y0 + 2.*(y1-y0)*mass_within, 0.)
    t = 2.*mass_within / jnp.maximum(y0+jnp.sqrt(discriminant), 1e-30)
    return (-1. + (bins+t.clip(0., 1.))*(2./area.shape[-1])).clip(-1., 1.)


def initialize_circuit(seed, rank=64, knots=129, lr=3e-4):
    if int(np.sqrt(rank))**2 != rank:
        raise ValueError("This documented 2D grid initialization requires square rank")
    side = int(np.sqrt(rank))
    points = -1. + (np.arange(side)+.5)*2./side
    centers = np.stack(np.meshgrid(points, points, indexing="ij"), -1).reshape(rank, 2)
    rng = np.random.default_rng(seed)
    centers += rng.normal(0., .01, centers.shape)
    grid = np.linspace(-1., 1., knots)
    # Target-independent, overlapping local leaves; scale is INITIALIZATION only.
    leaves = -.5*((grid[None, None]-centers[..., None])/.25)**2
    leaves += rng.normal(0., .02, leaves.shape)
    leaves = jnp.asarray(leaves, dtype=jnp.float32)
    height = -jnp.log(float(rank))-log_integrals(leaves).sum(-1)
    params = dict(leaves=leaves, height=height)
    return TrainState.create(apply_fn=circuit_q, params=params, tx=optax.adam(lr)), jax.random.PRNGKey(seed)


def make_circuit_update(target, count=512, defensive=.5):
    if not 0. < defensive < 1.:
        raise ValueError("Defensive proposal must have full box support")
    @jax.jit
    def update(state, key):
        key, skey, ukey, mkey = jax.random.split(key, 4)
        sampled = circuit_sample(state.params, skey, count)
        uniform = jax.random.uniform(ukey, (count, 2), minval=-1., maxval=1.)
        choose = jax.random.uniform(mkey, (count,)) < defensive
        actions = jax.lax.stop_gradient(jnp.where(choose[:, None], uniform, sampled))
        log_proposal = jnp.logaddexp(jnp.log1p(-defensive)+circuit_logp(state.params, actions),
                                    jnp.log(defensive)-jnp.log(4.))
        target_q = target_logp(actions, target)
        weights = jax.lax.stop_gradient(jnp.exp(target_q-log_proposal))
        def loss(params):
            logz = circuit_partition(params)[0]
            q = circuit_q(params, actions)
            # Generalized KL on unnormalized energy, not self-normalized weights.
            return jnp.exp(logz)-jnp.mean(weights*q)
        loss_value, grads = jax.value_and_grad(loss)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, key, dict(loss=loss_value, grad_norm=optax.global_norm(grads),
            estimated_target_partition=weights.mean(),
            source_ess=weights.sum()**2/jnp.maximum(jnp.square(weights).sum(), 1e-20))
    return update


def coarse_cell_mass(params, cells=8):
    leaves = params["leaves"]
    intervals = leaves.shape[-1]-1
    if intervals % cells:
        raise ValueError("Cell boundaries must align with knots")
    logz, masses, integrals = circuit_partition(params)
    factors = jnp.exp(leaves-integrals[..., None])
    area = (factors[..., :-1]+factors[..., 1:]) / intervals
    prob = area.reshape(leaves.shape[0], 2, cells, intervals//cells).sum(-1)
    return jnp.einsum("r,ri,rj->ij", jnp.exp(masses-logz), prob[:, 0], prob[:, 1])
