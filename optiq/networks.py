from collections.abc import Sequence

import flax.linen as nn
import jax.numpy as jnp


def kernel_init(scale: float = 1.0):
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    output_dim: int
    final_kernel_scale: float = 1.0

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = inputs
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim, kernel_init=kernel_init())(x)
            x = nn.gelu(x)
        return nn.Dense(
            self.output_dim,
            kernel_init=kernel_init(self.final_kernel_scale),
        )(x)


class Actor(nn.Module):
    """One-step implicit actor a = mu(s, z)."""

    action_dim: int
    hidden_dims: Sequence[int] = (256, 256, 256)

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        latents: jnp.ndarray,
    ) -> jnp.ndarray:
        inputs = jnp.concatenate((observations, latents), axis=-1)
        return MLP(
            self.hidden_dims,
            self.action_dim,
            final_kernel_scale=1.0e-2,
        )(inputs)


class CriticHead(nn.Module):
    hidden_dims: Sequence[int] = (256, 256, 256)

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> jnp.ndarray:
        inputs = jnp.concatenate((observations, actions), axis=-1)
        return MLP(self.hidden_dims, 1)(inputs).squeeze(-1)


class TwinCritic(nn.Module):
    """Two independently parameterized scalar critics."""

    hidden_dims: Sequence[int] = (256, 256, 256)

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> jnp.ndarray:
        q1 = CriticHead(self.hidden_dims, name="q1")(observations, actions)
        q2 = CriticHead(self.hidden_dims, name="q2")(observations, actions)
        return jnp.stack((q1, q2), axis=0)
