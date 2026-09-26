"""Actor/critic Adam with optional pre-Adam global L2 gradient clipping."""
import math

import optax


def adam_with_grad_clip(learning_rate, b1, b2, max_grad_norm=None):
    adam = optax.adam(learning_rate=learning_rate, b1=b1, b2=b2)
    if max_grad_norm is None:
        return adam
    limit = float(max_grad_norm)
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError("ac_grad_norm must be null or positive and finite")
    # For the critic, the tree includes both Q networks (one joint norm).
    return optax.chain(optax.clip_by_global_norm(limit), adam)
