"""Marginal student-mixture NLL; physical actions with box truncation."""

import jax
import jax.numpy as jnp
from .box_gaussian import component_log_prob


def direct_gmm_nll(mu, log_std, teacher_u, weights):
    """Marginal finite-mixture NLL with detached value-weighted teacher.

    B,N,D conditional parameters; B,M,D candidates; B,M weights.
    teacher_u is an actual bounded action in this implementation. Includes
    the differentiable truncation normalization, no transformation Jacobian.
    No responsibility is detached inside the loss.
    """
    u = jax.lax.stop_gradient(teacher_u)
    w = jax.lax.stop_gradient(weights)
    ell = component_log_prob(u, mu, log_std)
    log_mix = jax.scipy.special.logsumexp(ell, axis=1) - jnp.log(mu.shape[1])
    return -(w * log_mix).sum(-1).mean(), ell


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


def hard_projection_mass_error(row_probabilities, column_weights):
    """TV error caused by replacing each uniform-source row by its argmax."""
    count = row_probabilities.shape[-1]
    hard_mass = jax.nn.one_hot(jnp.argmax(row_probabilities, axis=-1), count).mean(axis=-2)
    return (0.5 * jnp.abs(hard_mass - column_weights).sum(axis=-1)).mean()
