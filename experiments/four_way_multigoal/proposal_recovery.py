"""Compare proposal families on the symmetric four-way multi-goal task.

This ports the experiment from the repository's ``skewed-proposal`` branch to
the proposal functions used by the current critic-DIME implementation.  All
methods share initialization, replay data logic, optimization, and evaluation;
only proposal geometry/allocation differ.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

import flax
import flax.linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax

from optiq_dime.policy import ImplicitActor
from optiq_dime.proposals import (
    proposal_log_density,
    sample_proposals,
    stabilize_proposal_log_density,
)
from optiq_dime.transport import clip_action, sample_truncated_gaussian, sinkhorn

from .environment import MultiGoalEnv


METHODS = {
    "stratified_truncated_normal": ("isotropic_truncated", "stratified"),
    "exact_truncated_normal": ("isotropic_truncated", "exact"),
    "qgradcov_exact_truncated_normal": ("qgradcov_truncated", "exact"),
    "qgradcov_stratified_truncated_normal": (
        "qgradcov_truncated",
        "stratified",
    ),
    "qgradcov_exact_normal": ("qgradcov_gaussian", "exact"),
    "qgradcov_stratified_normal": ("qgradcov_gaussian", "stratified"),
    "gamma_qgrad_exact_normal": ("gamma_qgradcov_gaussian", "exact"),
    "gamma_qgrad_stratified_normal": (
        "gamma_qgradcov_gaussian",
        "stratified",
    ),
}


def kernel_init(scale: float = 1.0):
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


class CriticHead(nn.Module):
    hidden_dims: tuple[int, ...]

    @nn.compact
    def __call__(self, observations: jax.Array, actions: jax.Array) -> jax.Array:
        x = jnp.concatenate((observations, actions), axis=-1)
        for width in self.hidden_dims:
            x = nn.Dense(width, kernel_init=kernel_init())(x)
            x = nn.gelu(x)
        return nn.Dense(1, kernel_init=kernel_init())(x).squeeze(-1)


class TwinCritic(nn.Module):
    hidden_dims: tuple[int, ...]

    @nn.compact
    def __call__(self, observations: jax.Array, actions: jax.Array) -> jax.Array:
        q1 = CriticHead(self.hidden_dims, name="q1")(observations, actions)
        q2 = CriticHead(self.hidden_dims, name="q2")(observations, actions)
        return jnp.stack((q1, q2))


@dataclass(frozen=True)
class ExperimentConfig:
    proposal_family: str
    proposal_sampling_mode: str
    actor_hidden_dims: tuple[int, ...] = (256, 256, 256)
    critic_hidden_dims: tuple[int, ...] = (256, 256, 256)
    actor_lr: float = 3.0e-4
    critic_lr: float = 3.0e-4
    discount: float = 0.99
    target_tau: float = 0.005
    num_particles: int = 16
    proposals_per_particle: int = 5
    proposal_std: float = 0.2
    proposal_clip: float = 0.5
    perpendicular_std_ratio: float = 0.5
    gamma_shape: float = 2.0
    gamma_scale: float = 0.1
    gamma_perpendicular_std: float = 0.1
    include_anchor: bool = True
    clip_untruncated: bool = True
    temperature: float = 0.25
    sinkhorn_epsilon: float = 0.05
    sinkhorn_iterations: int = 30
    td_noise_std: float = 0.2
    td_noise_clip: float = 0.5


class FourWayAgent(flax.struct.PyTreeNode):
    rng: Any
    actor: TrainState
    critic: TrainState
    target_critic_params: Any
    config: ExperimentConfig = flax.struct.field(pytree_node=False)

    @classmethod
    def create(cls, seed: int, config: ExperimentConfig) -> "FourWayAgent":
        actor_model = ImplicitActor(2, config.actor_hidden_dims)
        critic_model = TwinCritic(config.critic_hidden_dims)
        rng, actor_rng, critic_rng = jax.random.split(jax.random.PRNGKey(seed), 3)
        observations = jnp.zeros((1, 2), dtype=jnp.float32)
        actions = jnp.zeros((1, 2), dtype=jnp.float32)
        actor_params = actor_model.init(actor_rng, observations, actions)["params"]
        critic_params = critic_model.init(critic_rng, observations, actions)["params"]
        actor = TrainState.create(
            apply_fn=actor_model.apply,
            params=actor_params,
            tx=optax.adam(config.actor_lr),
        )
        critic = TrainState.create(
            apply_fn=critic_model.apply,
            params=critic_params,
            tx=optax.adam(config.critic_lr),
        )
        return cls(rng, actor, critic, critic_params, config)

    @jax.jit
    def sample_actions(
        self,
        observations: jax.Array,
        rng: jax.Array,
    ) -> jax.Array:
        latents = jax.random.normal(rng, observations.shape, observations.dtype)
        actions = self.actor.apply_fn(
            {"params": self.actor.params}, observations, latents
        )
        return clip_action(actions)

    def critic_loss(self, critic_params, batch, rng):
        latent_key, noise_key = jax.random.split(rng)
        next_latents = jax.random.normal(
            latent_key, batch["next_observations"].shape
        )
        next_actions = self.actor.apply_fn(
            {"params": self.actor.params},
            batch["next_observations"],
            next_latents,
        )
        next_actions = clip_action(next_actions)
        next_actions = sample_truncated_gaussian(
            noise_key,
            next_actions,
            repeats=1,
            std=self.config.td_noise_std,
            perturb_clip=self.config.td_noise_clip,
        )[:, 0]
        target_q = self.critic.apply_fn(
            {"params": self.target_critic_params},
            batch["next_observations"],
            next_actions,
        ).min(axis=0)
        targets = jax.lax.stop_gradient(
            batch["rewards"]
            + self.config.discount * batch["masks"] * target_q
        )
        q = self.critic.apply_fn(
            {"params": critic_params}, batch["observations"], batch["actions"]
        )
        loss = jnp.mean(jnp.square(q - targets[None]))
        return loss, {
            "critic_loss": loss,
            "critic_q_mean": q.mean(),
            "critic_target_mean": targets.mean(),
        }

    def actor_loss(self, actor_params, critic_params, observations, rng):
        config = self.config
        batch_size = observations.shape[0]
        latent_key, proposal_key, assignment_key = jax.random.split(rng, 3)
        latents = jax.random.normal(
            latent_key,
            (batch_size, config.num_particles, 2),
            dtype=observations.dtype,
        )
        repeated_observations = jnp.broadcast_to(
            observations[:, None, :],
            (batch_size, config.num_particles, 2),
        )
        raw_actions = self.actor.apply_fn(
            {"params": actor_params},
            repeated_observations.reshape(-1, 2),
            latents.reshape(-1, 2),
        ).reshape(batch_size, config.num_particles, 2)
        particles = clip_action(raw_actions)
        detached_particles = jax.lax.stop_gradient(particles)

        q_gradients = None
        if config.proposal_family != "isotropic_truncated":
            flat_observations = repeated_observations.reshape(-1, 2)

            def summed_q(flat_actions):
                twin_q = self.critic.apply_fn(
                    {"params": critic_params}, flat_observations, flat_actions
                )
                return twin_q.mean(axis=0).sum()

            q_gradients = jax.grad(summed_q)(
                detached_particles.reshape(-1, 2)
            ).reshape(detached_particles.shape)
            q_gradients = jax.lax.stop_gradient(q_gradients)

        proposals, _, out_of_bounds_fraction = sample_proposals(
            proposal_key,
            detached_particles,
            q_gradients,
            config.proposals_per_particle,
            config.proposal_sampling_mode,
            config.proposal_family,
            config.proposal_std,
            config.proposal_clip,
            config.include_anchor,
            config.perpendicular_std_ratio,
            config.gamma_shape,
            config.gamma_scale,
            config.gamma_perpendicular_std,
            config.clip_untruncated,
        )
        proposals = proposals.reshape(batch_size, -1, 2)
        num_proposals = proposals.shape[1]
        proposal_observations = jnp.broadcast_to(
            observations[:, None, :], (batch_size, num_proposals, 2)
        )
        source_qs = self.critic.apply_fn(
            {"params": critic_params},
            proposal_observations.reshape(-1, 2),
            proposals.reshape(-1, 2),
        ).reshape(2, batch_size, num_proposals)
        source_q = jax.lax.stop_gradient(source_qs.mean(axis=0))
        raw_log_density = proposal_log_density(
            jax.lax.stop_gradient(proposals),
            detached_particles,
            q_gradients,
            config.proposal_family,
            config.proposal_std,
            config.proposal_clip,
            config.perpendicular_std_ratio,
            config.gamma_shape,
            config.gamma_scale,
            config.gamma_perpendicular_std,
        )
        unsupported_fraction = jnp.mean(~jnp.isfinite(raw_log_density))
        log_density = stabilize_proposal_log_density(raw_log_density)
        logits = source_q / config.temperature - log_density
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
                config.sinkhorn_epsilon,
                config.sinkhorn_iterations,
            )
        )
        row_distribution = transport / jnp.maximum(
            transport.sum(axis=-1, keepdims=True), 1.0e-20
        )
        selected_indices = jax.random.categorical(
            assignment_key,
            jnp.log(jnp.maximum(row_distribution, 1.0e-20)),
            axis=-1,
        )
        selected_actions = jax.vmap(lambda values, indices: values[indices])(
            proposals, selected_indices
        )
        selected_actions = jax.lax.stop_gradient(selected_actions)
        loss = jnp.mean(jnp.sum(jnp.square(raw_actions - selected_actions), axis=-1))
        ess = 1.0 / jnp.sum(jnp.square(source_weights), axis=-1)
        gradient_norm = (
            jnp.asarray(0.0)
            if q_gradients is None
            else jnp.linalg.norm(q_gradients, axis=-1).mean()
        )
        return loss, {
            "actor_loss": loss,
            "source_ess_fraction": (ess / num_proposals).mean(),
            "proposal_gradient_norm": gradient_norm,
            "proposal_out_of_bounds_fraction": out_of_bounds_fraction,
            "proposal_unsupported_fraction": unsupported_fraction,
            "policy_spread_l2": jnp.linalg.norm(
                particles.std(axis=1), axis=-1
            ).mean(),
        }

    @jax.jit
    def update(self, batch: dict[str, jax.Array]):
        critic_key, actor_key, next_key = jax.random.split(self.rng, 3)
        (_, critic_metrics), critic_gradients = jax.value_and_grad(
            self.critic_loss, has_aux=True
        )(self.critic.params, batch, critic_key)
        critic = self.critic.apply_gradients(grads=critic_gradients)
        target_params = optax.incremental_update(
            critic.params, self.target_critic_params, self.config.target_tau
        )
        (_, actor_metrics), actor_gradients = jax.value_and_grad(
            self.actor_loss, has_aux=True
        )(
            self.actor.params,
            critic.params,
            batch["observations"],
            actor_key,
        )
        actor = self.actor.apply_gradients(grads=actor_gradients)
        return (
            self.replace(
                rng=next_key,
                actor=actor,
                critic=critic,
                target_critic_params=target_params,
            ),
            {**critic_metrics, **actor_metrics},
        )


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int):
        self.capacity = capacity
        self.observations = np.empty((capacity, 2), np.float32)
        self.actions = np.empty((capacity, 2), np.float32)
        self.rewards = np.empty(capacity, np.float32)
        self.next_observations = np.empty((capacity, 2), np.float32)
        self.masks = np.empty(capacity, np.float32)
        self.index = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, observation, action, reward, next_observation, terminated):
        self.observations[self.index] = observation
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_observations[self.index] = next_observation
        self.masks[self.index] = 0.0 if terminated else 1.0
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, jax.Array]:
        indices = self.rng.integers(self.size, size=batch_size)
        return {
            "observations": jnp.asarray(self.observations[indices]),
            "actions": jnp.asarray(self.actions[indices]),
            "rewards": jnp.asarray(self.rewards[indices]),
            "next_observations": jnp.asarray(self.next_observations[indices]),
            "masks": jnp.asarray(self.masks[indices]),
        }


def cardinal_targets(latents: jax.Array, magnitude: float) -> jax.Array:
    horizontal = jnp.abs(latents[:, 0]) >= jnp.abs(latents[:, 1])
    signs = jnp.where(latents >= 0.0, 1.0, -1.0)
    x_actions = jnp.stack(
        (signs[:, 0] * magnitude, jnp.zeros_like(signs[:, 0])), axis=-1
    )
    y_actions = jnp.stack(
        (jnp.zeros_like(signs[:, 1]), signs[:, 1] * magnitude), axis=-1
    )
    return jnp.where(horizontal[:, None], x_actions, y_actions)


def initialize_cardinal_policy(
    agent: FourWayAgent,
    seed: int,
    steps: int,
    magnitude: float = 0.25,
) -> FourWayAgent:
    optimizer = optax.adam(1.0e-3)
    optimizer_state = optimizer.init(agent.actor.params)

    @jax.jit
    def update(params, state, rng):
        latents = jax.random.normal(rng, (512, 2))
        observations = jnp.zeros((512, 2), dtype=jnp.float32)
        targets = cardinal_targets(latents, magnitude)

        def loss_fn(candidate_params):
            predictions = agent.actor.apply_fn(
                {"params": candidate_params}, observations, latents
            )
            return jnp.mean(jnp.square(predictions - targets))

        gradients = jax.grad(loss_fn)(params)
        updates, state = optimizer.update(gradients, state, params)
        return optax.apply_updates(params, updates), state

    params = agent.actor.params
    rng = jax.random.PRNGKey(seed + 91_000)
    for _ in range(steps):
        rng, step_key = jax.random.split(rng)
        params, optimizer_state = update(params, optimizer_state, step_key)
    actor = agent.actor.replace(params=params, opt_state=agent.actor.tx.init(params))
    return agent.replace(actor=actor)


def sample_initial_actions(agent: FourWayAgent, count: int, seed: int) -> np.ndarray:
    observations = jnp.zeros((count, 2), dtype=jnp.float32)
    return np.asarray(agent.sample_actions(observations, jax.random.PRNGKey(seed)))


def mode_summary(samples: np.ndarray, prefix: str) -> dict[str, object]:
    directions = np.asarray(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
        dtype=np.float32,
    )
    if len(samples) == 0:
        return {
            f"{prefix}_sample_count": 0,
            f"{prefix}_confident_fraction": 0.0,
            f"{prefix}_mode_counts": [0, 0, 0, 0],
            f"{prefix}_mode_probabilities": [0.0, 0.0, 0.0, 0.0],
            f"{prefix}_mode_entropy_normalized": 0.0,
            f"{prefix}_mode_tv_to_uniform": 0.5,
            f"{prefix}_mode_max_abs_bias": 0.25,
            f"{prefix}_uniform_chi_square_df3": 0.0,
            f"{prefix}_covered_modes": 0,
            f"{prefix}_mean": [0.0, 0.0],
            f"{prefix}_norm_mean": 0.0,
        }
    norms = np.linalg.norm(samples, axis=-1)
    scores = samples @ directions.T / np.maximum(norms[:, None], 1.0e-8)
    assignments = np.argmax(scores, axis=-1)
    confident = (norms >= 0.05) & (
        np.max(scores, axis=-1) >= np.cos(np.pi / 6.0)
    )
    counts = np.bincount(assignments[confident], minlength=4)
    probabilities = counts / max(counts.sum(), 1)
    positive = probabilities > 0.0
    entropy = -np.sum(probabilities[positive] * np.log(probabilities[positive]))
    uniform = np.full(4, 0.25)
    expected = max(counts.sum(), 1) / 4.0
    chi_square = float(np.sum(np.square(counts - expected) / expected))
    return {
        f"{prefix}_sample_count": int(len(samples)),
        f"{prefix}_confident_fraction": float(confident.mean()),
        f"{prefix}_mode_counts": counts.tolist(),
        f"{prefix}_mode_probabilities": probabilities.tolist(),
        f"{prefix}_mode_entropy_normalized": float(entropy / np.log(4.0)),
        f"{prefix}_mode_tv_to_uniform": float(
            0.5 * np.abs(probabilities - uniform).sum()
        ),
        f"{prefix}_mode_max_abs_bias": float(
            np.abs(probabilities - uniform).max()
        ),
        f"{prefix}_uniform_chi_square_df3": chi_square,
        f"{prefix}_covered_modes": int(
            np.sum(counts >= max(1, int(np.ceil(0.05 * len(samples)))))
        ),
        f"{prefix}_mean": samples.mean(axis=0).tolist(),
        f"{prefix}_norm_mean": float(norms.mean()),
    }


def collect_trajectories(
    agent: FourWayAgent, count: int, horizon: int, seed: int
) -> tuple[list[np.ndarray], np.ndarray]:
    environment = MultiGoalEnv(horizon=horizon)
    rng = jax.random.PRNGKey(seed)
    trajectories = []
    for episode in range(count):
        observation, _ = environment.reset(seed=seed + episode)
        trajectory = [observation.copy()]
        terminated = truncated = False
        while not (terminated or truncated):
            rng, action_key = jax.random.split(rng)
            action = np.asarray(
                agent.sample_actions(jnp.asarray(observation[None]), action_key)[0]
            )
            observation, _, terminated, truncated, _ = environment.step(action)
            trajectory.append(observation.copy())
        trajectories.append(np.asarray(trajectory))
    endpoints = np.asarray([trajectory[-1] for trajectory in trajectories])
    environment.close()
    return trajectories, endpoints


def trajectory_summary(endpoints: np.ndarray) -> dict[str, object]:
    goals = MultiGoalEnv().goal_positions
    distances = np.linalg.norm(endpoints[:, None] - goals[None], axis=-1)
    nearest = np.argmin(distances, axis=-1)
    successful = distances.min(axis=-1) < 1.0
    successful_directions = goals[nearest[successful]] / 5.0
    summary = mode_summary(successful_directions, "trajectory_goal")
    summary["trajectory_count"] = int(len(endpoints))
    summary["trajectory_success_rate"] = float(successful.mean())
    return summary


def plot_initial_actions(actions: np.ndarray, path: Path) -> None:
    directions = np.asarray(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
        dtype=np.float32,
    )
    assignments = np.argmax(actions @ directions.T, axis=-1)
    colors = ("#2775c9", "#df6b2f", "#2e9b59", "#8b55b5")
    figure, axis = plt.subplots(figsize=(5.4, 5.2))
    for mode, color in enumerate(colors):
        selected = actions[assignments == mode]
        axis.scatter(selected[:, 0], selected[:, 1], s=7, alpha=0.22, color=color)
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    axis.axvline(0.0, color="black", linewidth=0.8, alpha=0.35)
    axis.set(xlim=(-1, 1), ylim=(-1, 1), aspect="equal", xlabel="a_x", ylabel="a_y")
    axis.grid(alpha=0.1)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=tuple(METHODS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--init-steps", type=int, default=1_500)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--trajectory-count", type=int, default=400)
    parser.add_argument("--initial-action-count", type=int, default=4096)
    parser.add_argument("--proposal-std", type=float, default=0.2)
    parser.add_argument("--proposal-clip", type=float, default=0.5)
    parser.add_argument("--perpendicular-std-ratio", type=float, default=0.5)
    parser.add_argument("--gamma-shape", type=float, default=2.0)
    parser.add_argument("--gamma-scale", type=float, default=0.1)
    parser.add_argument("--gamma-perpendicular-std", type=float, default=0.1)
    parser.add_argument(
        "--include-anchor", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--clip-untruncated", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--log-interval", type=int, default=5_000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/four_way_multigoal/proposal_recovery"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    family, sampling_mode = METHODS[args.method]
    output_dir = args.output_dir / args.method / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    config = ExperimentConfig(
        proposal_family=family,
        proposal_sampling_mode=sampling_mode,
        proposal_std=args.proposal_std,
        proposal_clip=args.proposal_clip,
        perpendicular_std_ratio=args.perpendicular_std_ratio,
        gamma_shape=args.gamma_shape,
        gamma_scale=args.gamma_scale,
        gamma_perpendicular_std=args.gamma_perpendicular_std,
        include_anchor=args.include_anchor,
        clip_untruncated=args.clip_untruncated,
        temperature=args.temperature,
    )
    started = time.monotonic()
    agent = initialize_cardinal_policy(
        FourWayAgent.create(args.seed, config), args.seed, args.init_steps
    )
    environment = MultiGoalEnv(horizon=args.horizon)
    observation, _ = environment.reset(seed=args.seed)
    replay = ReplayBuffer(args.total_steps, args.seed + 1)
    numpy_rng = np.random.default_rng(args.seed + 2)
    action_rng = jax.random.PRNGKey(args.seed + 3)
    latest_metrics = {}

    for step in range(1, args.total_steps + 1):
        if step <= args.warmup_steps:
            action = environment.action_space.sample()
        else:
            action_rng, sample_key = jax.random.split(action_rng)
            action = np.asarray(
                agent.sample_actions(jnp.asarray(observation[None]), sample_key)[0]
            )
        next_observation, reward, terminated, truncated, _ = environment.step(action)
        replay.add(observation, action, reward, next_observation, terminated)
        observation = next_observation
        if step > args.warmup_steps:
            agent, latest_metrics = agent.update(replay.sample(256))
        if terminated or truncated:
            observation, _ = environment.reset(seed=int(numpy_rng.integers(2**31)))
        if step % args.log_interval == 0:
            current_actions = sample_initial_actions(
                agent, args.initial_action_count, args.seed + 1_000_000 + step
            )
            current_summary = mode_summary(current_actions, "s0_action")
            metrics_text = " ".join(
                f"{key}={float(value):.4f}"
                for key, value in sorted(latest_metrics.items())
            )
            print(
                f"step={step} elapsed={time.monotonic() - started:.1f}s "
                f"{metrics_text} "
                f"covered={current_summary['s0_action_covered_modes']} "
                f"tv={current_summary['s0_action_mode_tv_to_uniform']:.4f} "
                f"counts={current_summary['s0_action_mode_counts']}",
                flush=True,
            )

    actions = sample_initial_actions(agent, args.initial_action_count, args.seed + 2_000_000)
    trajectories, endpoints = collect_trajectories(
        agent, args.trajectory_count, args.horizon, args.seed + 10_000
    )
    del trajectories
    summary = {
        "method": args.method,
        "proposal_family": family,
        "proposal_sampling_mode": sampling_mode,
        "seed": args.seed,
        "total_steps": args.total_steps,
        "include_anchor": args.include_anchor,
        "clip_untruncated": args.clip_untruncated,
        "proposal_std": args.proposal_std,
        "proposal_clip": args.proposal_clip,
        "perpendicular_std_ratio": args.perpendicular_std_ratio,
        "gamma_shape": args.gamma_shape,
        "gamma_scale": args.gamma_scale,
        "gamma_perpendicular_std": args.gamma_perpendicular_std,
        "temperature": args.temperature,
        "wall_time_seconds": time.monotonic() - started,
        "final_training_metrics": {
            key: float(value) for key, value in latest_metrics.items()
        },
        **mode_summary(actions, "s0_action"),
        **trajectory_summary(endpoints),
    }
    np.savez_compressed(output_dir / "s0_actions.npz", actions=actions)
    np.savez_compressed(output_dir / "trajectory_endpoints.npz", endpoints=endpoints)
    plot_initial_actions(actions, output_dir / "s0_action_modes.png")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    environment.close()
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
