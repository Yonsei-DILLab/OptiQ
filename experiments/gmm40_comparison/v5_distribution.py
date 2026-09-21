"""Unmodified numerical definitions extracted from v5-gmm40."""
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

def conditional_ot_nll(mu, log_std, teacher_u, row_probabilities):
    """Expected Gaussian NLL, averaged over states and latent components.

    Shapes: mu/log_std [B,N,D], teacher_u [B,K,D], probabilities [B,N,K].
    Unlike fitting one argmax point, this objective retains the row's variance.
    """
    targets = jax.lax.stop_gradient(teacher_u)
    probabilities = jax.lax.stop_gradient(row_probabilities)
    # Sufficient moments avoid materializing [B,N,K,D].
    mean = jnp.einsum("bnk,bkd->bnd", probabilities, targets)
    variance = jnp.maximum(
        jnp.einsum("bnk,bkd->bnd", probabilities, targets**2) - mean**2, 0.0
    )
    squared_error = (mu - mean)**2 + variance
    nll = 0.5 * squared_error * jnp.exp(-2.0 * log_std) + log_std + 0.5 * jnp.log(2*jnp.pi)
    return nll.sum(axis=-1).mean()
