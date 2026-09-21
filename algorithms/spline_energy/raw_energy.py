"""State-conditioned extension of the successful GMM40 raw energy circuit.

There is no independently learned V head. Q=alpha*log(F), logZ=log integral F,
and pi=F/Z all derive from the same unnormalized positive spline factors.
The existing analytic inverse-CDF sampler consumes the normalized view.
"""
import math

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpy as np

from .model import log_integrals, mixture_log_value


def reference_initial_factors(seed, rank, action_dim, knots):
    """Exact GMM40 initializer in 2D; target-independent box centers in D>2."""
    rng = np.random.default_rng(seed)
    side = math.isqrt(rank)
    if action_dim == 2 and side * side == rank:
        points = -1. + (np.arange(side) + .5) * 2. / side
        centers = np.stack(np.meshgrid(points, points, indexing="ij"), -1).reshape(rank, 2)
        centers += rng.normal(0., .01, centers.shape)
    else:
        # A Cartesian 2D grid has no dimension-independent counterpart. This
        # explicit extension retains the reference span and leaf width .25.
        centers = rng.uniform(-.875, .875, (rank, action_dim))
    grid = np.linspace(-1., 1., knots)
    leaves = -.5 * ((grid[None, None] - centers[..., None]) / .25) ** 2
    leaves += rng.normal(0., .02, leaves.shape)
    leaves = jnp.asarray(leaves, jnp.float32)
    roots = -jnp.log(float(rank)) - log_integrals(leaves).sum(-1)
    return roots, leaves


def output_from_factors(roots, leaves, temperature):
    leaf_integrals = log_integrals(leaves)
    root_mass = roots + leaf_integrals.sum(-1)
    logz = logsumexp(root_mass, axis=-1)
    return {"value": temperature * logz, "log_partition": logz,
            "log_weights": root_mass - logz[..., None],
            "log_leaves": leaves - leaf_integrals[..., None],
            "raw_log_roots": roots, "raw_log_leaves": leaves}


class ConditionalRawEnergyCircuit(nn.Module):
    action_dim: int
    rank: int = 64
    knots: int = 129
    temperature: float = .25
    hidden_dims: tuple = (256, 256)
    initialization_seed: int = 0

    @nn.compact
    def __call__(self, observations):
        if self.rank <= 0 or self.knots < 2 or self.temperature <= 0:
            raise ValueError("Need positive rank/temperature and at least two knots")
        x = observations
        for index, width in enumerate(self.hidden_dims):
            x = nn.Dense(width, name=f"encoder_{index}")(x)
            x = nn.LayerNorm(name=f"encoder_ln_{index}")(x)
            x = nn.gelu(x)
        roots0, leaves0 = reference_initial_factors(
            self.initialization_seed, self.rank, self.action_dim, self.knots)

        def fixed_bias(values):
            return lambda _key, shape, dtype: values.astype(dtype).reshape(shape)

        # Zero output kernels make the initial conditional circuit exactly the
        # state-free reference at every state. State dependence is then learned.
        roots = nn.Dense(self.rank, kernel_init=nn.initializers.zeros,
                         bias_init=fixed_bias(roots0), name="roots")(x)
        leaves = nn.Dense(self.rank * self.action_dim * self.knots,
                          kernel_init=nn.initializers.zeros,
                          bias_init=fixed_bias(leaves0), name="leaves")(x)
        leaves = leaves.reshape(observations.shape[:-1] +
                                (self.rank, self.action_dim, self.knots))
        return output_from_factors(roots, leaves, self.temperature)


class StateFreeRawEnergyCircuit(nn.Module):
    """The GMM40 case of the same raw-energy equations, without unused state weights."""
    action_dim: int = 2
    rank: int = 64
    knots: int = 129
    temperature: float = 1.
    initialization_seed: int = 0

    @nn.compact
    def __call__(self):
        roots0, leaves0 = reference_initial_factors(
            self.initialization_seed, self.rank, self.action_dim, self.knots)
        roots = self.param("roots", lambda _key, shape: roots0.reshape(shape), (self.rank,))
        leaves = self.param("leaves", lambda _key, shape: leaves0.reshape(shape),
                            (self.rank, self.action_dim, self.knots))
        return output_from_factors(roots[None], leaves[None], self.temperature)


def fixed_q_forward_energy_loss(output, actions, target_log_energy, proposal_log_prob):
    """The original GMM40 generalized-KL estimator, in dimensionless energy.

    Only for a fixed queryable oracle with known proposal density (e.g. GMM40).
    This is NOT exponentiated TD regression and is not the MuJoCo update.
    Inputs can be B repeated copies of one state with B oracle actions.
    Keep the oracle in a numerically reasonable common energy gauge, as the
    normalized GMM40 target does. No self-normalization or weight clipping.
    """
    actions = jax.lax.stop_gradient(actions)
    weights = jax.lax.stop_gradient(jnp.exp(target_log_energy - proposal_log_prob))
    energy = mixture_log_value(output["raw_log_roots"], output["raw_log_leaves"], actions)
    return jnp.mean(jnp.exp(output["log_partition"]) - weights * energy)


def sample_many_from_single_output(output, key, count):
    """Draw ``count`` actions from one state's exact normalized circuit."""
    if count <= 0:
        raise ValueError("count must be positive")
    weights = output["log_weights"]
    leaves = output["log_leaves"]
    if weights.shape[0] != 1 or leaves.shape[0] != 1:
        raise ValueError("Expected exactly one state")
    weights, leaves = jnp.exp(weights[0]), leaves[0]
    root_key, bin_key, within_key = jax.random.split(key, 3)
    root_cdf = jnp.cumsum(weights).at[-1].set(1.)
    root_u = jax.random.uniform(root_key, (count,))
    roots = jnp.sum(root_cdf[None] <= root_u[:, None], axis=-1).clip(0, weights.shape[0] - 1)
    selected = leaves[roots]
    scaled = jnp.exp(selected - selected.max(axis=-1, keepdims=True))
    areas = .5 * (scaled[..., :-1] + scaled[..., 1:])
    cdf = jnp.cumsum(areas, axis=-1) / areas.sum(axis=-1, keepdims=True)
    cdf = cdf.at[..., -1].set(1.)
    uniform = jax.random.uniform(bin_key, selected.shape[:-1])
    bins = jnp.sum(cdf <= uniform[..., None], axis=-1).clip(0, areas.shape[-1] - 1)
    y0 = jnp.take_along_axis(scaled, bins[..., None], axis=-1)[..., 0]
    y1 = jnp.take_along_axis(scaled, (bins + 1)[..., None], axis=-1)[..., 0]
    within = jax.random.uniform(within_key, selected.shape[:-1])
    mass = within * .5 * (y0 + y1)
    discriminant = jnp.maximum(y0 * y0 + 2. * (y1 - y0) * mass, 0.)
    fraction = 2. * mass / jnp.maximum(y0 + jnp.sqrt(discriminant), 1e-30)
    return (-1. + (bins + fraction.clip(0., 1.)) * (2. / areas.shape[-1])).clip(-1., 1.)
