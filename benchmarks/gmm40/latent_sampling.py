"""Randomized stratification of the existing two-dimensional Gaussian prior."""
import math
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp


@partial(jax.jit, static_argnames=['count','dtype'])
def gaussian_grid(key, count, dtype=jnp.float32):
    """One jittered draw per equal-probability 2D cell, in random row order.

    This uses no target information. Each row is marginally standard Gaussian
    after the random permutation; rows are dependent because of stratification.
    Endpoint guards only avoid nonfinite inverse-CDF values in float arithmetic.
    """
    if count <= 0:
        raise ValueError('Latent count must be positive')
    height = math.isqrt(count)
    while count % height:
        height -= 1
    width = count // height
    jitter_key, permutation_key, swap_key = jax.random.split(key,3)
    cells = jnp.stack((jnp.arange(count)//width,jnp.arange(count)%width),axis=-1)
    jitter = jax.random.uniform(jitter_key,(count,2),dtype=dtype,minval=jnp.finfo(dtype).eps)
    quantiles = (cells.astype(dtype)+jitter)/jnp.asarray([height,width],dtype=dtype)
    quantiles = jnp.minimum(quantiles,jnp.nextafter(jnp.asarray(1.,dtype),jnp.asarray(0.,dtype)))
    z = jsp.special.ndtri(quantiles)
    z = jnp.where(jax.random.bernoulli(swap_key),z[:,::-1],z)
    return jax.random.permutation(permutation_key,z,axis=0)
