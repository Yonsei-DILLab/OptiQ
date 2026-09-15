"""Exact best-of-k law of an existing finite, weighted teacher.

This transforms the empirical Boltzmann target, not its sampling proposal.
No new actions, fitted Gaussians, density estimates, or resampling are needed.
"""

import jax.numpy as jnp


def best_of_k_mass(weights, scores, k):
    """Return the winner law of k IID draws from categorical ``weights``.

    Inputs have shape [..., candidates], with normalized nonnegative weights
    and finite scores. Equal-score draws are resolved by the first occurrence
    in the draw sequence; their winning mass is proportional to their weights.
    For a tie group with lower/upper CDF L/U, each member gets
        w_i * (U**k - L**k) / (U - L).
    The polynomial identity below avoids cancellation and division by zero.
    It also preserves tiny top-score masses when float32 rounds L and U to 1.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("best-of-k requires a positive integer")
    if weights.shape != scores.shape or weights.ndim < 1 or weights.shape[-1] < 1:
        raise ValueError("weights and scores must have matching nonempty candidate axes")
    if k == 1:
        return weights
    weights = weights / weights.sum(axis=-1, keepdims=True)
    other_scores = scores[..., None, :]
    own_scores = scores[..., :, None]
    lower = jnp.sum(jnp.where(other_scores < own_scores, weights[..., None, :], 0), axis=-1)
    upper = jnp.sum(jnp.where(other_scores <= own_scores, weights[..., None, :], 0), axis=-1)
    lower, upper = jnp.clip(lower, 0, 1), jnp.clip(upper, 0, 1)
    multiplier = jnp.zeros_like(weights)
    for power in range(k):
        multiplier += upper ** (k - 1 - power) * lower ** power
    winner = weights * multiplier
    # Roundoff normalization; the categorical winner law already sums to one.
    return winner / winner.sum(axis=-1, keepdims=True)
