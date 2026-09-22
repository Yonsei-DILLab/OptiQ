"""Verbatim legacy ImplicitActor, isolated from RL trainer imports."""
from collections.abc import Sequence
import jax
import jax.numpy as jnp
import flax.linen as nn

def kernel_init(scale: float = 1.0):
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")

class ImplicitActor(nn.Module):
    action_dim: int
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations: jax.Array, latents: jax.Array) -> jax.Array:
        x = jnp.concatenate((observations, latents), axis=-1)
        for width in self.hidden_dims:
            x = nn.Dense(width, kernel_init=kernel_init())(x)
            x = nn.gelu(x)
        return nn.Dense(
            self.action_dim,
            kernel_init=kernel_init(1.0e-2),
        )(x)
