"""Distill the live actor's best-of-k winner law with uniform empirical OT.

Each teacher atom comes from its own independent group of k full policy draws.
There is no Boltzmann tilt, proposal-density correction, or teacher sigma floor.
"""

import jax
import jax.numpy as jnp

from .distillation import conditional_ot_nll
from .transport import sinkhorn


def select_winner_atoms(candidate_u, twin_q):
    """Select within [batch, winner, candidate], retaining original pre-tanh u."""
    scores = twin_q.min(axis=0)
    indices = scores.argmax(axis=-1)
    winners_u = jnp.take_along_axis(candidate_u, indices[..., None, None], axis=2).squeeze(2)
    selected_q = jnp.take_along_axis(scores, indices[..., None], axis=-1).squeeze(-1)
    metrics = {
        "teacher_winner_q": selected_q.mean(),
        "teacher_candidate_q": scores.mean(),
        "teacher_winner_q_gain": (selected_q - scores.mean(axis=-1)).mean(),
    }
    return jax.lax.stop_gradient(winners_u), jax.tree.map(jax.lax.stop_gradient, metrics)


def sample_winner_teacher(actor_state, qf_state, observations, key, num_winners, k):
    """Fresh continuous z AND Gaussian epsilon per candidate, live twin-min Q."""
    batch, obs_dim = observations.shape
    action_dim = actor_state.params["mu"]["bias"].shape[0]
    z_key, eps_key, dropout_key = jax.random.split(key, 3)
    shape = (batch * num_winners * k, action_dim)
    obs = jnp.broadcast_to(observations[:, None, None, :],
                           (batch, num_winners, k, obs_dim)).reshape(-1, obs_dim)
    z = jax.random.normal(z_key, shape, dtype=observations.dtype)
    params = jax.tree.map(jax.lax.stop_gradient, actor_state.params)
    mu, log_std = actor_state.apply_fn({"params": params}, obs, z)
    u = mu + jnp.exp(log_std) * jax.random.normal(eps_key, shape, dtype=mu.dtype)
    u = jax.lax.stop_gradient(u)
    twin_q = qf_state.apply_fn(
        {"params": qf_state.params, "batch_stats": qf_state.batch_stats},
        obs, jnp.tanh(u), rngs={"dropout": dropout_key}, train=False,
    )
    if twin_q.shape != (2, shape[0], 1):
        raise ValueError("Winner teacher requires two scalar critics")
    return select_winner_atoms(u.reshape(batch, num_winners, k, action_dim),
                              twin_q.reshape(2, batch, num_winners, k))


def winner_ot_targets(mu, winner_u, epsilon, iterations):
    """Uniform columns (winners), uniform rows (student latent means)."""
    actions = jnp.tanh(jax.lax.stop_gradient(winner_u))
    costs = ((jnp.tanh(mu)[:, :, None, :] - actions[:, None, :, :])**2).sum(-1)
    weights = jnp.full(actions.shape[:2], 1.0 / actions.shape[1], dtype=mu.dtype)
    plan = jax.lax.stop_gradient(sinkhorn(costs, weights, epsilon, iterations))
    rows = plan / jnp.maximum(plan.sum(axis=-1, keepdims=True), 1.e-20)
    return rows, {
        "ot_cost_mean": costs.mean(),
        "ot_row_marginal_error": jnp.abs(plan.sum(-1) - 1.0 / mu.shape[1]).mean(),
        "ot_col_marginal_error": jnp.abs(plan.sum(-2) - weights).mean(),
        "teacher_uniform_mass": jnp.asarray(1.0),
    }


def update_winner_actor(actor_state, qf_state, observations, key,
                        num_policy_samples, proposals_per_policy_sample,
                        k, epsilon, iterations):
    key, latent_key, teacher_key, _ = jax.random.split(key, 4)
    batch, obs_dim = observations.shape
    action_dim = actor_state.params["mu"]["bias"].shape[0]
    # Preserve the v5 student latent draw; teacher uses independent continuous draws.
    z_key, _ = jax.random.split(latent_key)
    z = jax.random.normal(z_key, (batch, num_policy_samples, action_dim), observations.dtype)
    obs = jnp.broadcast_to(observations[:, None, :],
                           (batch, num_policy_samples, obs_dim)).reshape(-1, obs_dim)
    num_winners = num_policy_samples * proposals_per_policy_sample
    winner_u, teacher_metrics = sample_winner_teacher(
        actor_state, qf_state, observations, teacher_key, num_winners, k)

    def loss_fn(params):
        mu, log_std = actor_state.apply_fn({"params": params}, obs, z.reshape(-1, action_dim))
        mu, log_std = [x.reshape(batch, num_policy_samples, action_dim) for x in (mu, log_std)]
        rows, metrics = winner_ot_targets(mu, winner_u, epsilon, iterations)
        loss = conditional_ot_nll(mu, log_std, winner_u, rows)
        metrics.update(
            actor_loss=loss, actor_std_mean=jnp.exp(log_std).mean(),
            actor_std_min=jnp.exp(log_std).min(), actor_std_max=jnp.exp(log_std).max(),
            actor_log_std_mean=log_std.mean(),
            student_action_saturation_fraction=(jnp.abs(jnp.tanh(mu)) > .99).mean(),
            teacher_action_saturation_fraction=(jnp.abs(jnp.tanh(winner_u)) > .99).mean(),
            teacher_best_of_k=jnp.asarray(float(k)),
            teacher_winner_count=jnp.asarray(float(num_winners)),
            teacher_candidate_count=jnp.asarray(float(num_winners * k)),
            **teacher_metrics,
        )
        return loss, metrics

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(actor_state.params)
    return actor_state.apply_gradients(grads=grads), loss, key, metrics
