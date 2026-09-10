"""Tanh-Gaussian policy density and a distinct pre-tanh teacher KDE.

All densities are joint action densities in normalized action coordinates.
Pre-tanh values are retained; atanh(tanh(u)) is never used.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp


def tanh_log_jacobian(u):
    return jnp.sum(2.0 * (jnp.log(2.0) - u - jax.nn.softplus(-2.0 * u)), axis=-1)


def conditional_mixture_log_prob(u, mu, log_std):
    """u: [B,K,D], conditional parameters: [B,M,D]; result: [B,K]."""
    standardized = (u[:, :, None, :] - mu[:, None, :, :]) * jnp.exp(-log_std[:, None, :, :])
    # Sum action coordinates FIRST, then mix entire Gaussian components.
    component = jnp.sum(
        -0.5 * standardized**2 - log_std[:, None, :, :] - 0.5 * jnp.log(2.0 * jnp.pi),
        axis=-1,
    )
    return (jsp.special.logsumexp(component, axis=-1) - jnp.log(mu.shape[1])
            - tanh_log_jacobian(u))


def actor_components(actor_state, observations, key, count):
    """Independent conditional components per replay state."""
    batch_size, obs_dim = observations.shape
    action_dim = actor_state.params["mu"]["bias"].shape[0]
    z = jax.random.normal(key, (batch_size, count, action_dim), dtype=observations.dtype)
    obs = jnp.broadcast_to(observations[:, None, :], (batch_size, count, obs_dim))
    mu, log_std = actor_state.apply_fn(
        {"params": actor_state.params}, obs.reshape(batch_size * count, obs_dim),
        z.reshape(batch_size * count, action_dim),
    )
    return mu.reshape(batch_size, count, action_dim), log_std.reshape(batch_size, count, action_dim)


def idac_action_and_log_density(actor_state, observations, key, count):
    """One action per state; its generating component is included among count."""
    latent_key, noise_key = jax.random.split(key)
    mu, log_std = actor_components(actor_state, observations, latent_key, count)
    eps = jax.random.normal(noise_key, mu[:, 0].shape, dtype=mu.dtype)
    u = mu[:, 0] + jnp.exp(log_std[:, 0]) * eps
    log_g = conditional_mixture_log_prob(u[:, None], mu, log_std)[:, 0]
    return jnp.tanh(u), log_g


class PretanhTeacherKDE(NamedTuple):
    """Teacher q_F; centers are realized student u, NOT actor means."""

    centers: jax.Array
    bandwidth: float

    def sample(self, key, repeats, mode):
        batch_size, n_centers, action_dim = self.centers.shape
        count = n_centers * repeats
        component_key, noise_key = jax.random.split(key)
        if mode == "exact":
            indices = jax.random.randint(component_key, (batch_size, count), 0, n_centers)
        elif mode == "stratified":
            indices = jnp.broadcast_to(jnp.repeat(jnp.arange(n_centers), repeats), (batch_size, count))
        else:
            raise ValueError("Teacher sampling must be exact or stratified")
        selected = jnp.take_along_axis(self.centers, indices[:, :, None], axis=1)
        noise = jax.random.normal(noise_key, (batch_size, count, action_dim), dtype=self.centers.dtype)
        v = selected + self.bandwidth * noise
        return jnp.tanh(v), v, indices

    def log_prob(self, v):
        log_std = jnp.full_like(self.centers, jnp.log(self.bandwidth))
        return conditional_mixture_log_prob(v, self.centers, log_std)


class ConditionalGaussianProposal(NamedTuple):
    """Explicit mixture of actor conditionals, with a teacher-only std floor.

    This is a distinct proposal option, not a KDE on realized student actions.
    The density uses exactly the same scales as sampling, including the floor.
    """
    means: jax.Array
    log_std: jax.Array
    minimum_std: float

    def effective_log_std(self):
        return jnp.maximum(self.log_std, jnp.log(self.minimum_std))

    def sample(self, key, repeats, mode):
        batch,components,dim=self.means.shape
        count=components*repeats
        component_key,noise_key=jax.random.split(key)
        if mode=='exact':
            indices=jax.random.randint(component_key,(batch,count),0,components)
        elif mode=='stratified':
            indices=jnp.broadcast_to(jnp.repeat(jnp.arange(components),repeats),(batch,count))
        else:
            raise ValueError('Teacher sampling must be exact or stratified')
        mu=jnp.take_along_axis(self.means,indices[:,:,None],axis=1)
        ls=jnp.take_along_axis(self.effective_log_std(),indices[:,:,None],axis=1)
        v=mu+jnp.exp(ls)*jax.random.normal(noise_key,(batch,count,dim),dtype=mu.dtype)
        return jnp.tanh(v),v,indices

    def log_prob(self,v):
        return conditional_mixture_log_prob(v,self.means,self.effective_log_std())
