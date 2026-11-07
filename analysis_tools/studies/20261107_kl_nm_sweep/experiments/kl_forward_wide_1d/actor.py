"""Action-range adaptation of actor definitions from direct-gmm-trg@36aab085bcfe8219997e1bc21e3ad8fe55f266b0."""
from collections.abc import Sequence
import flax.linen as nn
import jax.numpy as jnp

def kernel_init(scale: float = 1.0):
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")

class SemiImplicitActor(nn.Module):
    """Bounded Gaussian center and log scale for box-truncated conditionals."""

    action_dim: int
    hidden_dims: Sequence[int]
    log_std_min: float
    log_std_max: float
    initial_log_std: float
    mean_output_init_scale: float = 1.0e-4
    log_std_output_init_scale: float = 0.0
    mean_latent_skip_scale: float = 0.0

    @nn.compact
    def __call__(self, observations, latents):
        x = jnp.concatenate((observations, latents), axis=-1)
        for width in self.hidden_dims:
            x = nn.gelu(nn.Dense(width, kernel_init=kernel_init())(x))
        mu = nn.Dense(self.action_dim, kernel_init=kernel_init(self.mean_output_init_scale), name="mu")(x)
        if self.mean_latent_skip_scale != 0.0:
            mu = mu + self.mean_latent_skip_scale*latents
        raw_log_std = nn.Dense(
            self.action_dim, kernel_init=kernel_init(self.log_std_output_init_scale),
            bias_init=nn.initializers.constant(self.initial_log_std), name="log_std",
        )(x)
        return 10.0 * jnp.tanh(mu), jnp.clip(raw_log_std, self.log_std_min, self.log_std_max)
