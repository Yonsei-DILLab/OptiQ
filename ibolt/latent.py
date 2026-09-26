"""Optional finite latent prior with a reproducible, fixed codebook."""
from flax import struct
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp


class FiniteMixtureTrainState(TrainState):
    # Static distribution metadata comes from the saved run config on restore.
    latent_components: int = struct.field(pytree_node=False, default=16)
    latent_codebook_seed: int = struct.field(pytree_node=False, default=20260911)


def finite_latent_codes(actor_state, action_dim, dtype=jnp.float32):
    """Uniform finite support, centered and scaled once per action coordinate."""
    codes = jax.random.normal(jax.random.PRNGKey(actor_state.latent_codebook_seed),
                              (actor_state.latent_components, action_dim), dtype=dtype)
    centered = codes-codes.mean(axis=0, keepdims=True)
    return centered/jnp.maximum(centered.std(axis=0, keepdims=True), 1.e-6)


def sample_latents(actor_state, key, shape, dtype=jnp.float32):
    if isinstance(actor_state, FiniteMixtureTrainState):
        codes = finite_latent_codes(actor_state, shape[-1], dtype)
        indices = jax.random.randint(key, shape[:-1], 0, actor_state.latent_components)
        return codes[indices]
    return jax.random.normal(key, shape, dtype=dtype)


def stratified_finite_latents(actor_state, batch_size, count, action_dim, dtype):
    if count != actor_state.latent_components:
        raise ValueError("Finite mixture OT must use one student per actual component")
    return jnp.broadcast_to(finite_latent_codes(actor_state, action_dim, dtype),
                            (batch_size, count, action_dim))
