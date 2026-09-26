"""Value-weighted marginal likelihood of a box-truncated Gaussian mixture."""
import jax
import jax.numpy as jnp
from .box_gaussian import component_log_prob

def direct_gmm_nll(mu, log_std, actions, weights):
    actions, weights = jax.lax.stop_gradient(actions), jax.lax.stop_gradient(weights)
    component = component_log_prob(actions, mu, log_std)
    marginal = jax.scipy.special.logsumexp(component, axis=1) - jnp.log(mu.shape[1])
    return -(weights * marginal).sum(-1).mean(), component
