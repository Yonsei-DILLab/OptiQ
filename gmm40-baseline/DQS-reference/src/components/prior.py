import distrax 
import jax
import jax.numpy as jnp

class Prior:
    def __init__(self, dim, scale):
        self.dim = dim
        self.scale = scale
        self.dist = distrax.MultivariateNormalDiag(
            jnp.zeros((dim)),
            jnp.ones((dim)) * (scale**2),
        )
        self.key = jax.random.PRNGKey(0)

    def log_prob(self, x):
        return self.dist.log_prob(x)

    def sample(self, n_samples):
        key, self.key = jax.random.split(self.key)
        return self.dist.sample(seed=key, sample_shape=(n_samples,))
