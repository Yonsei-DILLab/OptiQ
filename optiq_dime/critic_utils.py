"""Scalar and categorical Q helpers; scalar TD follows mujoco-setting."""

import jax
import jax.numpy as jnp


def validate_critic_config(critic):
    kind = critic.get("type", "distributional")
    if kind not in {"distributional", "scalar"}:
        raise ValueError("critic.type must be distributional or scalar")
    if isinstance(critic.n_atoms, bool) or not isinstance(critic.n_atoms, int):
        raise ValueError("critic.n_atoms must be an integer")
    if kind == "scalar":
        if critic.n_atoms != 1 or critic.entr_coeff != 0:
            raise ValueError("Scalar critic requires n_atoms=1 and entr_coeff=0")
        if critic.v_min is not None or critic.v_max is not None:
            raise ValueError("Scalar critic has no atom support: set v_min/v_max=null")
    else:
        if critic.n_atoms < 2:
            raise ValueError("Distributional critic requires n_atoms >= 2; select critic.type=scalar for scalar TD")
        if critic.v_min is None or critic.v_max is None or critic.v_min >= critic.v_max:
            raise ValueError("critic.v_min must be below critic.v_max")


def critic_support(num_atoms, v_min, v_max):
    # A scalar head has no categorical support. Keep the common JIT signature.
    return jnp.ones((1,)) if num_atoms == 1 else jnp.linspace(v_min, v_max, num_atoms)


def critic_expectation(outputs, support):
    if outputs.shape[-1] == 1:
        return outputs[..., 0]
    return jnp.sum(outputs * support, axis=-1)


def scalar_td_loss(current_q, next_q, rewards, dones, gamma):
    """Twin-min bootstrap and summed critic MSE from mujoco-setting (7e2da67).

    No atom clipping or distribution entropy applies.
    `dones` excludes time-limit truncations, as provided by the shared buffer.
    """
    targets = jax.lax.stop_gradient(
        rewards + (1.0 - dones) * gamma * next_q.min(axis=0)
    )
    return jnp.square(current_q - targets[None, :]).mean(axis=1).sum(), targets
