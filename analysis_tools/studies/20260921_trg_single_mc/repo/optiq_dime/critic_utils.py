"""Interpret scalar or categorical critic outputs without changing Q's scale."""

import jax.numpy as jnp


def critic_expectation(prediction, support):
    """A single output is raw Q, not a categorical mass at support[0]."""
    if prediction.shape[-1] == 1:
        return prediction[..., 0]
    return jnp.sum(prediction * support, axis=-1)
