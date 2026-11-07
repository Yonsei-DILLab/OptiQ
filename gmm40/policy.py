"""GMM40 actor initialization, isolated from the shared policy defaults."""
import flax.linen as nn
import jax.numpy as jnp

from ibolt.policy import SemiImplicitActor, kernel_init


class GMM40Actor(SemiImplicitActor):
    """Same actor and output heads, with orthogonal hidden kernels (gain sqrt(2))."""

    @nn.compact
    def __call__(self, observations, latents):
        x = jnp.concatenate((observations, latents), axis=-1)
        for width in self.hidden_dims:
            x = nn.gelu(nn.Dense(
                width, kernel_init=nn.initializers.orthogonal(scale=2.0**0.5),
            )(x))
        mu = nn.Dense(
            self.action_dim, kernel_init=kernel_init(self.mean_output_init_scale), name="mu",
        )(x)
        raw_log_std = nn.Dense(
            self.action_dim, kernel_init=kernel_init(self.log_std_output_init_scale),
            bias_init=nn.initializers.constant(self.initial_log_std), name="log_std",
        )(x)
        return jnp.tanh(mu), jnp.clip(raw_log_std, self.log_std_min, self.log_std_max)
