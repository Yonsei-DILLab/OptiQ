import functools
from dataclasses import dataclass, replace
from typing import Any

import flax
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax

from .networks import Actor, TwinCritic
from .transport import (
    clip_action,
    sample_truncated_gaussian,
    sinkhorn,
    truncated_mixture_log_density,
)


nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)


@dataclass(frozen=True)
class OptiQConfig:
    actor_hidden_dims: tuple[int, ...] = (256, 256, 256)
    critic_hidden_dims: tuple[int, ...] = (256, 256, 256)
    actor_lr: float = 3.0e-4
    critic_lr: float = 3.0e-4
    discount: float = 0.99
    target_tau: float = 0.005
    actor_update_frequency: int = 1

    num_particles: int = 16
    proposals_per_particle: int = 5
    proposal_std: float = 0.2
    proposal_clip: float = 0.5
    include_anchor: bool = True
    temperature: float = 0.25
    sinkhorn_epsilon: float = 0.05
    sinkhorn_iterations: int = 30

    td_noise_std: float = 0.2
    td_noise_clip: float = 0.5

    action_low: tuple[float, ...] = ()
    action_high: tuple[float, ...] = ()

    def validate(self) -> None:
        if not self.actor_hidden_dims or not self.critic_hidden_dims:
            raise ValueError("Actor and critic must each have at least one hidden layer.")
        if self.actor_lr <= 0.0 or self.critic_lr <= 0.0:
            raise ValueError("Learning rates must be positive.")
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("discount must be in [0, 1].")
        if not 0.0 < self.target_tau <= 1.0:
            raise ValueError("target_tau must be in (0, 1].")
        if self.actor_update_frequency < 1:
            raise ValueError("actor_update_frequency must be at least one.")
        if self.num_particles < 1 or self.proposals_per_particle < 1:
            raise ValueError("num_particles and proposals_per_particle must be positive.")
        if self.include_anchor and self.proposals_per_particle < 2:
            raise ValueError("An anchor requires at least two proposals per particle.")
        if self.proposal_std <= 0.0 or self.td_noise_std <= 0.0:
            raise ValueError("Proposal and TD noise standard deviations must be positive.")
        if self.proposal_clip <= 0.0 or self.td_noise_clip <= 0.0:
            raise ValueError("Proposal and TD noise clips must be positive.")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        if self.sinkhorn_epsilon <= 0.0 or self.sinkhorn_iterations < 1:
            raise ValueError("Sinkhorn settings must be positive.")


class OptiQ(flax.struct.PyTreeNode):
    rng: Any
    actor: TrainState
    critic: TrainState
    target_critic_params: Any
    config: OptiQConfig = nonpytree_field()

    @classmethod
    def create(
        cls,
        seed: int,
        observation_dim: int,
        action_low,
        action_high,
        config: OptiQConfig | None = None,
    ) -> "OptiQ":
        config = OptiQConfig() if config is None else config
        action_low = tuple(float(value) for value in action_low)
        action_high = tuple(float(value) for value in action_high)
        if len(action_low) != len(action_high) or not action_low:
            raise ValueError("Action bounds must be non-empty and have equal length.")
        if any(low >= high for low, high in zip(action_low, action_high, strict=True)):
            raise ValueError("Every action lower bound must be below its upper bound.")
        config = replace(config, action_low=action_low, action_high=action_high)
        config.validate()

        action_dim = len(action_low)
        actor_def = Actor(action_dim, config.actor_hidden_dims)
        critic_def = TwinCritic(config.critic_hidden_dims)
        rng, actor_rng, critic_rng = jax.random.split(jax.random.PRNGKey(seed), 3)
        dummy_observations = jnp.zeros((1, observation_dim), dtype=jnp.float32)
        dummy_actions = jnp.zeros((1, action_dim), dtype=jnp.float32)
        dummy_latents = jnp.zeros_like(dummy_actions)

        actor_params = actor_def.init(actor_rng, dummy_observations, dummy_latents)["params"]
        critic_params = critic_def.init(
            critic_rng, dummy_observations, dummy_actions
        )["params"]
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(config.actor_lr),
        )
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.adam(config.critic_lr),
        )
        return cls(rng, actor, critic, critic_params, config)

    @functools.partial(jax.jit, static_argnames=("deterministic",))
    def sample_actions(
        self,
        observations: jax.Array,
        rng: jax.Array,
        deterministic: bool = False,
    ) -> jax.Array:
        action_dim = len(self.config.action_low)
        latents = (
            jnp.zeros((observations.shape[0], action_dim), observations.dtype)
            if deterministic
            else jax.random.normal(
                rng, (observations.shape[0], action_dim), dtype=observations.dtype
            )
        )
        raw_actions = self.actor.apply_fn(
            {"params": self.actor.params}, observations, latents
        )
        return clip_action(
            raw_actions, self.config.action_low, self.config.action_high
        )

    def _critic_loss(self, critic_params, batch, rng):
        batch_size = batch["observations"].shape[0]
        action_dim = len(self.config.action_low)
        latent_rng, noise_rng = jax.random.split(rng)
        next_latents = jax.random.normal(
            latent_rng, (batch_size, action_dim), dtype=batch["observations"].dtype
        )
        next_actions = self.actor.apply_fn(
            {"params": self.actor.params}, batch["next_observations"], next_latents
        )
        next_actions = clip_action(
            next_actions, self.config.action_low, self.config.action_high
        )
        next_actions = sample_truncated_gaussian(
            noise_rng,
            next_actions,
            repeats=1,
            std=self.config.td_noise_std,
            perturb_clip=self.config.td_noise_clip,
            action_low=self.config.action_low,
            action_high=self.config.action_high,
        )[:, 0]

        target_qs = self.critic.apply_fn(
            {"params": self.target_critic_params},
            batch["next_observations"],
            next_actions,
        )
        target_value = jnp.min(target_qs, axis=0)
        targets = jax.lax.stop_gradient(
            batch["rewards"]
            + self.config.discount * batch["masks"] * target_value
        )
        qs = self.critic.apply_fn(
            {"params": critic_params}, batch["observations"], batch["actions"]
        )
        loss = jnp.mean(jnp.square(qs - targets[None]))
        metrics = {
            "critic/loss": loss,
            "critic/q_mean": qs.mean(),
            "critic/target_mean": targets.mean(),
        }
        return loss, metrics

    def _actor_loss(self, actor_params, critic_params, batch, rng):
        observations = batch["observations"]
        batch_size = observations.shape[0]
        action_dim = len(self.config.action_low)
        num_particles = self.config.num_particles
        repeats = self.config.proposals_per_particle
        latent_rng, proposal_rng, assignment_rng = jax.random.split(rng, 3)

        latents = jax.random.normal(
            latent_rng,
            (batch_size, num_particles, action_dim),
            dtype=observations.dtype,
        )
        repeated_observations = jnp.broadcast_to(
            observations[:, None, :],
            (batch_size, num_particles, observations.shape[-1]),
        )
        raw_actions = self.actor.apply_fn(
            {"params": actor_params},
            repeated_observations.reshape(batch_size * num_particles, -1),
            latents.reshape(batch_size * num_particles, action_dim),
        ).reshape(batch_size, num_particles, action_dim)
        particles = clip_action(
            raw_actions, self.config.action_low, self.config.action_high
        )

        proposals = sample_truncated_gaussian(
            proposal_rng,
            particles,
            repeats,
            self.config.proposal_std,
            self.config.proposal_clip,
            self.config.action_low,
            self.config.action_high,
            include_anchor=self.config.include_anchor,
        ).reshape(batch_size, num_particles * repeats, action_dim)
        num_proposals = proposals.shape[1]
        proposal_observations = jnp.broadcast_to(
            observations[:, None, :],
            (batch_size, num_proposals, observations.shape[-1]),
        )
        source_qs = self.critic.apply_fn(
            {"params": critic_params},
            proposal_observations.reshape(batch_size * num_proposals, -1),
            proposals.reshape(batch_size * num_proposals, action_dim),
        ).reshape(2, batch_size, num_proposals)
        source_q = jax.lax.stop_gradient(source_qs.mean(axis=0))

        proposal_log_density = truncated_mixture_log_density(
            jax.lax.stop_gradient(proposals),
            jax.lax.stop_gradient(particles),
            self.config.proposal_std,
            self.config.proposal_clip,
            self.config.action_low,
            self.config.action_high,
        )
        logits = source_q / self.config.temperature - proposal_log_density
        source_weights = jax.lax.stop_gradient(jax.nn.softmax(logits, axis=-1))

        squared_costs = jnp.sum(
            jnp.square(particles[:, :, None, :] - proposals[:, None, :, :]),
            axis=-1,
        )
        costs = squared_costs / (
            squared_costs.mean(axis=(-2, -1), keepdims=True) + 1.0e-8
        )
        transport = jax.lax.stop_gradient(
            sinkhorn(
                costs,
                source_weights,
                self.config.sinkhorn_epsilon,
                self.config.sinkhorn_iterations,
            )
        )
        row_distribution = transport / jnp.maximum(
            transport.sum(axis=-1, keepdims=True), 1.0e-20
        )
        selected_indices = jax.random.categorical(
            assignment_rng,
            jnp.log(jnp.maximum(row_distribution, 1.0e-20)),
            axis=-1,
        )
        selected_actions = jax.vmap(
            lambda actions, indices: actions[indices]
        )(proposals, selected_indices)
        selected_actions = jax.lax.stop_gradient(selected_actions)
        loss = jnp.mean(jnp.sum(jnp.square(raw_actions - selected_actions), axis=-1))

        source_ess = 1.0 / jnp.sum(jnp.square(source_weights), axis=-1)
        metrics = {
            "actor/loss": loss,
            "actor/source_ess_fraction": (source_ess / num_proposals).mean(),
            "actor/selected_delta_l2": jnp.linalg.norm(
                particles - selected_actions, axis=-1
            ).mean(),
            "actor/policy_spread_l2": jnp.linalg.norm(
                particles.std(axis=1), axis=-1
            ).mean(),
        }
        return loss, metrics

    @jax.jit
    def update(self, batch: dict[str, jax.Array]):
        critic_rng, actor_rng, next_rng = jax.random.split(self.rng, 3)

        (critic_loss, critic_metrics), critic_grads = jax.value_and_grad(
            self._critic_loss, has_aux=True
        )(self.critic.params, batch, critic_rng)
        del critic_loss
        new_critic = self.critic.apply_gradients(grads=critic_grads)
        new_target_params = optax.incremental_update(
            new_critic.params,
            self.target_critic_params,
            self.config.target_tau,
        )

        (actor_loss, actor_metrics), actor_grads = jax.value_and_grad(
            self._actor_loss, has_aux=True
        )(self.actor.params, new_critic.params, batch, actor_rng)
        del actor_loss
        candidate_actor = self.actor.apply_gradients(grads=actor_grads)
        update_actor = (
            new_critic.step % self.config.actor_update_frequency == 0
        )
        new_actor = jax.tree_util.tree_map(
            lambda candidate, current: jnp.where(
                update_actor, candidate, current
            ),
            candidate_actor,
            self.actor,
        )
        return (
            self.replace(
                rng=next_rng,
                actor=new_actor,
                critic=new_critic,
                target_critic_params=new_target_params,
            ),
            {**critic_metrics, **actor_metrics},
        )
