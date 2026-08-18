import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from experiments.four_way_multigoal.environment import MultiGoalEnv
from optiq.agent import OptiQ, OptiQConfig
from optiq.train import ReplayBuffer, save_checkpoint


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
    agent: OptiQ,
    seed: int,
    steps: int,
    batch_size: int,
    magnitude: float,
) -> OptiQ:
    optimizer = optax.adam(1.0e-3)
    optimizer_state = optimizer.init(agent.actor.params)

    @jax.jit
    def update(params, state, rng):
        latents = jax.random.normal(rng, (batch_size, 2))
        observations = jnp.zeros((batch_size, 2), dtype=jnp.float32)
        targets = cardinal_targets(latents, magnitude)

        def loss_fn(candidate_params):
            predictions = agent.actor.apply_fn(
                {"params": candidate_params}, observations, latents
            )
            return jnp.mean(jnp.square(predictions - targets))

        _, gradients = jax.value_and_grad(loss_fn)(params)
        updates, state = optimizer.update(gradients, state, params)
        return optax.apply_updates(params, updates), state

    params = agent.actor.params
    rng = jax.random.PRNGKey(seed + 91_000)
    for _ in range(steps):
        rng, step_rng = jax.random.split(rng)
        params, optimizer_state = update(params, optimizer_state, step_rng)
    actor = agent.actor.replace(params=params, opt_state=agent.actor.tx.init(params))
    return agent.replace(actor=actor)


def collect_trajectories(
    agent: OptiQ, count: int, horizon: int, seed: int
) -> list[np.ndarray]:
    environment = MultiGoalEnv(horizon=horizon)
    rng = jax.random.PRNGKey(seed)
    trajectories = []
    for episode in range(count):
        observation, _ = environment.reset(seed=seed + episode)
        trajectory = [observation.copy()]
        terminated = truncated = False
        while not (terminated or truncated):
            rng, action_rng = jax.random.split(rng)
            action = np.asarray(
                agent.sample_actions(
                    jnp.asarray(observation[None]), action_rng, deterministic=False
                )[0]
            )
            observation, _, terminated, truncated, _ = environment.step(action)
            trajectory.append(observation.copy())
        trajectories.append(np.asarray(trajectory))
    environment.close()
    return trajectories


def summarize_trajectories(trajectories: list[np.ndarray]) -> dict[str, object]:
    endpoints = np.asarray([trajectory[-1] for trajectory in trajectories])
    goals = MultiGoalEnv().goal_positions
    distances = np.linalg.norm(endpoints[:, None, :] - goals[None, :, :], axis=-1)
    nearest = np.argmin(distances, axis=1)
    successes = distances.min(axis=1) < 1.0
    counts = np.bincount(nearest[successes], minlength=len(goals))
    probabilities = counts / max(counts.sum(), 1)
    positive = probabilities > 0.0
    entropy = -np.sum(probabilities[positive] * np.log(probabilities[positive]))
    lengths = np.asarray([len(trajectory) - 1 for trajectory in trajectories])
    return {
        "trajectory_count": len(trajectories),
        "trajectory_success_rate": float(successes.mean()),
        "trajectory_goal_counts": counts.tolist(),
        "trajectory_goal_entropy": float(entropy),
        "trajectory_mean_length": float(lengths.mean()),
    }


def plot_trajectories(trajectories: list[np.ndarray], path: Path) -> None:
    environment = MultiGoalEnv()
    coordinates = np.linspace(-7.5, 7.5, 240)
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    points = np.stack((x_grid, y_grid), axis=-1)
    distances = np.square(
        points[:, :, None, :] - environment.goal_positions[None, None, :, :]
    ).sum(axis=-1)
    reward_surface = -distances.min(axis=-1)

    figure, axis = plt.subplots(figsize=(7.0, 7.0))
    axis.contour(x_grid, y_grid, reward_surface, levels=18, cmap="turbo", alpha=0.7)
    for trajectory in trajectories:
        axis.plot(trajectory[:, 0], trajectory[:, 1], color="#6d9dcc", alpha=0.18)
    axis.scatter(
        environment.goal_positions[:, 0],
        environment.goal_positions[:, 1],
        color="#c9352b",
        s=45,
        zorder=3,
    )
    axis.scatter(0.0, 0.0, color="black", s=35, zorder=3)
    axis.set(
        title="OptiQ Policy Rollout Trajectories",
        xlabel="State x",
        ylabel="State y",
        xlim=(-7.5, 7.5),
        ylim=(-7.5, 7.5),
        aspect="equal",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--init-steps", type=int, default=1_500)
    parser.add_argument("--trajectory-count", type=int, default=100)
    parser.add_argument("--num-policy-samples", type=int, default=16)
    parser.add_argument("--proposals-per-policy-sample", type=int, default=5)
    parser.add_argument("--proposal-std", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument(
        "--density-correction",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output-dir", default="outputs/four_way_multigoal/online")
    parser.add_argument("--save-replay-buffer", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = MultiGoalEnv(horizon=args.horizon)
    observation, _ = environment.reset(seed=args.seed)
    config = OptiQConfig(
        num_policy_samples=args.num_policy_samples,
        proposals_per_policy_sample=args.proposals_per_policy_sample,
        proposal_std=args.proposal_std,
        proposal_clip=2.5 * args.proposal_std,
        density_correction=args.density_correction,
        temperature=args.temperature,
        td_noise_std=args.proposal_std,
        td_noise_clip=2.5 * args.proposal_std,
    )
    agent = OptiQ.create(
        args.seed,
        observation_dim=2,
        action_low=environment.action_space.low,
        action_high=environment.action_space.high,
        config=config,
    )
    agent = initialize_cardinal_policy(agent, args.seed, args.init_steps, 512, 0.25)
    replay = ReplayBuffer(args.total_steps, 2, 2, args.seed + 1)
    numpy_rng = np.random.default_rng(args.seed + 2)
    action_rng = jax.random.PRNGKey(args.seed + 3)

    for step in range(1, args.total_steps + 1):
        if step <= args.warmup_steps:
            action = environment.action_space.sample()
        else:
            action_rng, sample_rng = jax.random.split(action_rng)
            action = np.asarray(
                agent.sample_actions(jnp.asarray(observation[None]), sample_rng)[0]
            )
        next_observation, reward, terminated, truncated, _ = environment.step(action)
        replay.add(observation, action, reward, next_observation, terminated)
        observation = next_observation

        if step > args.warmup_steps and replay.size >= 256:
            agent, _ = agent.update(replay.sample(256))
        if terminated or truncated:
            observation, _ = environment.reset(seed=int(numpy_rng.integers(2**31)))
        if step % 10_000 == 0:
            print(f"step={step}")

    trajectories = collect_trajectories(
        agent, args.trajectory_count, args.horizon, args.seed + 10_000
    )
    trajectory_summary = summarize_trajectories(trajectories)
    np.savez_compressed(
        output_dir / "trajectories.npz",
        **{
            f"trajectory_{index:03d}": trajectory
            for index, trajectory in enumerate(trajectories)
        },
    )
    plot_trajectories(trajectories, output_dir / "policy_rollouts.png")
    save_checkpoint(
        output_dir,
        args.total_steps,
        agent,
        action_rng,
        replay,
        args.save_replay_buffer,
    )
    summary = {
        "seed": args.seed,
        "density_correction": args.density_correction,
        **trajectory_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    environment.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
