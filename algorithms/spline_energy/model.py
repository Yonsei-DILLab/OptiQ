"""State-conditioned positive spline circuit for joint Q, V and policy."""
from functools import partial
import math

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpy as np


def log_integrals(log_heights):
    """Exact integral of positive piecewise-linear leaves in log space."""
    dx = 2.0 / (log_heights.shape[-1] - 1)
    return logsumexp(jnp.logaddexp(log_heights[..., :-1],
                                  log_heights[..., 1:]), axis=-1) + math.log(dx / 2.0)


def initial_leaf_bias(rank, action_dim, knots):
    """Fixed target-independent broad modes; identical across training seeds."""
    rng = np.random.default_rng(1729 + action_dim)
    centers = rng.uniform(-0.5, 0.5, (rank, action_dim))
    centers[0] = 0.0
    grid = np.linspace(-1.0, 1.0, knots)
    values = -0.5 * ((grid[None, None] - centers[..., None]) / 0.5) ** 2
    values += rng.normal(0.0, 0.01, values.shape)
    return values.astype(np.float32).reshape(-1)


class ConditionalSplineCircuit(nn.Module):
    """One network whose normalized density defines both policy and soft Q."""

    action_dim: int
    rank: int = 16
    knots: int = 33
    hidden_dims: tuple = (256, 256)

    @nn.compact
    def __call__(self, observations):
        x = observations
        for index, width in enumerate(self.hidden_dims):
            x = nn.Dense(width, name=f"encoder_{index}")(x)
            x = nn.LayerNorm(name=f"encoder_ln_{index}")(x)
            x = nn.gelu(x)
        value = nn.Dense(1, kernel_init=nn.initializers.zeros,
                         bias_init=nn.initializers.zeros, name="value")(x)[..., 0]
        roots = nn.Dense(self.rank, kernel_init=nn.initializers.normal(0.01),
                         bias_init=nn.initializers.zeros, name="roots")(x)
        bias = jnp.asarray(initial_leaf_bias(self.rank, self.action_dim, self.knots))

        def bias_init(_key, shape, dtype=jnp.float32):
            assert int(np.prod(shape)) == bias.size
            return bias.astype(dtype).reshape(shape)

        leaves = nn.Dense(self.rank * self.action_dim * self.knots,
                          kernel_init=nn.initializers.normal(0.003),
                          bias_init=bias_init, name="leaves")(x)
        leaves = leaves.reshape(observations.shape[:-1] +
                                (self.rank, self.action_dim, self.knots))
        leaf_integrals = log_integrals(leaves)
        return {
            "value": value,
            "log_weights": jax.nn.log_softmax(roots, axis=-1),
            "log_leaves": leaves - leaf_integrals[..., None],
        }


def mixture_log_value(log_weights, leaves, actions):
    """Evaluate a positive sum-product circuit, normalized or unnormalized."""
    bins = leaves.shape[-1] - 1
    u = ((actions + 1.0) * bins / 2.0).clip(0.0, float(bins))
    index = jnp.minimum(jnp.floor(u).astype(jnp.int32), bins - 1)
    fraction = u - index
    gather = index[:, None, :, None]
    left = jnp.take_along_axis(leaves, gather, axis=-1)[..., 0]
    right = jnp.take_along_axis(leaves, gather + 1, axis=-1)[..., 0]
    log_fraction = jnp.log(jnp.maximum(fraction, 1e-30))[:, None, :]
    log_one_minus = jnp.log(jnp.maximum(1.0 - fraction, 1e-30))[:, None, :]
    log_leaf = jnp.logaddexp(left + log_one_minus, right + log_fraction)
    result = logsumexp(log_weights + log_leaf.sum(axis=-1), axis=-1)
    inside = jnp.all((actions >= -1.0) & (actions <= 1.0), axis=-1)
    return jnp.where(inside, result, -jnp.inf)


def log_prob_from_output(output, actions):
    """Evaluate the exact normalized mixture density on ``[-1, 1]^D``."""
    return mixture_log_value(output["log_weights"], output["log_leaves"], actions)


def q_from_output(output, actions, temperature):
    """Q=alpha log F for raw energies; legacy V+alpha log pi otherwise."""
    if "raw_log_roots" in output:
        return temperature * mixture_log_value(
            output["raw_log_roots"], output["raw_log_leaves"], actions)
    return output["value"] + temperature * log_prob_from_output(output, actions)


def sample_from_output(output, key):
    """One sample per state via a root CDF and analytic spline inverse CDF."""
    root_key, bin_key, within_key = jax.random.split(key, 3)
    weights = jnp.exp(output["log_weights"])
    root_cdf = jnp.cumsum(weights, axis=-1).at[:, -1].set(1.0)
    root_u = jax.random.uniform(root_key, (weights.shape[0],))
    roots = jnp.sum(root_cdf <= root_u[:, None], axis=-1).clip(0, weights.shape[-1] - 1)
    selected = output["log_leaves"][jnp.arange(weights.shape[0]), roots]
    scaled = jnp.exp(selected - selected.max(axis=-1, keepdims=True))
    areas = 0.5 * (scaled[..., :-1] + scaled[..., 1:])
    cdf = jnp.cumsum(areas, axis=-1) / areas.sum(axis=-1, keepdims=True)
    cdf = cdf.at[..., -1].set(1.0)
    uniform = jax.random.uniform(bin_key, selected.shape[:-1])
    bins = jnp.sum(cdf <= uniform[..., None], axis=-1).clip(0, areas.shape[-1] - 1)
    y0 = jnp.take_along_axis(scaled, bins[..., None], axis=-1)[..., 0]
    y1 = jnp.take_along_axis(scaled, (bins + 1)[..., None], axis=-1)[..., 0]
    within = jax.random.uniform(within_key, selected.shape[:-1])
    mass = within * 0.5 * (y0 + y1)
    discriminant = jnp.maximum(y0 * y0 + 2.0 * (y1 - y0) * mass, 0.0)
    t = 2.0 * mass / jnp.maximum(y0 + jnp.sqrt(discriminant), 1e-30)
    return (-1.0 + (bins + t.clip(0.0, 1.0)) * (2.0 / areas.shape[-1])).clip(-1.0, 1.0)


@partial(jax.jit, static_argnames=("model",))
def sample_action(params, observations, key, model):
    """Sample with one state-network forward pass and no iterative action search."""
    return sample_from_output(model.apply({"params": params}, observations), key)


def coarse_leaf_probabilities(output, cells):
    """Analytic cell probabilities, used by normalization validation."""
    leaves = output["log_leaves"]
    intervals = leaves.shape[-1] - 1
    if intervals % cells:
        raise ValueError("Cells must align with spline knots")
    dx = 2.0 / intervals
    areas = 0.5 * dx * (jnp.exp(leaves[..., :-1]) + jnp.exp(leaves[..., 1:]))
    return areas.reshape(areas.shape[:-1] + (cells, intervals // cells)).sum(axis=-1)
