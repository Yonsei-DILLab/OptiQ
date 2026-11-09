"""Diagonal Gaussian conditioned on [-1,1]^D, with bounded centers.

No tanh Jacobian or clipping of unconditioned Gaussian actions. Because centers
are inside the box and sigma<=exp(-1), each coordinate normalization is >=~.5.
Inverse CDF avoids exponentially inefficient whole-box rejection.
"""
import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr, ndtri, logsumexp


def log_normalizer(mu, log_std):
    inv = jnp.exp(-log_std)
    return jnp.log(ndtr((1-mu)*inv)-ndtr((-1-mu)*inv))


def sample_box(key, mu, log_std):
    std=jnp.exp(log_std)
    lo=ndtr((-1-mu)/std); hi=ndtr((1-mu)/std)
    uniform=jax.random.uniform(key,mu.shape,dtype=mu.dtype)
    # Floating point endpoint protection only; not action clipping.
    p=lo+(hi-lo)*uniform
    p=jnp.clip(p,jnp.finfo(p.dtype).eps,1-jnp.finfo(p.dtype).eps)
    # ndtri/ndtr float32 roundoff can overshoot the support by a few ULPs
    # (observed -1.00000012). Project numerical residue only, after exact
    # inverse-CDF truncation, so valid generated samples never get logp=-inf.
    return jnp.clip(mu+std*ndtri(p),-1.,1.)


def component_log_prob(actions, mu, log_std):
    """[B,M,D], [B,N,D] -> [B,N,M], exact normalized component density."""
    delta=(actions[:,None]-mu[:,:,None])*jnp.exp(-log_std[:,:,None])
    ell=(-.5*delta**2-log_std[:,:,None]-.5*jnp.log(2*jnp.pi)
         -log_normalizer(mu,log_std)[:,:,None]).sum(-1)
    inside=jnp.all((actions>=-1)&(actions<=1),axis=-1)
    return jnp.where(inside[:,None],ell,-jnp.inf)


def mixture_log_prob(actions,mu,log_std):
    return logsumexp(component_log_prob(actions,mu,log_std),axis=1)-jnp.log(mu.shape[1])
