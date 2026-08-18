import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import flax.serialization
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from .agent import OptiQ, OptiQConfig


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        observation_dim: int,
        action_dim: int,
        seed: int,
    ) -> None:
        self.capacity = capacity
        self.observations = np.empty((capacity, observation_dim), np.float32)
        self.actions = np.empty((capacity, action_dim), np.float32)
        self.rewards = np.empty((capacity,), np.float32)
        self.next_observations = np.empty((capacity, observation_dim), np.float32)
        self.masks = np.empty((capacity,), np.float32)
        self.index = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
    ) -> None:
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

    def save(self, path: Path) -> None:
        """Save valid transitions and ring-buffer state without compression."""
        valid_size = self.capacity if self.size == self.capacity else self.size
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        with temporary_path.open("wb") as handle:
            np.savez(
                handle,
                observations=self.observations[:valid_size],
                actions=self.actions[:valid_size],
                rewards=self.rewards[:valid_size],
                next_observations=self.next_observations[:valid_size],
                masks=self.masks[:valid_size],
                capacity=np.asarray(self.capacity, dtype=np.int64),
                index=np.asarray(self.index, dtype=np.int64),
                size=np.asarray(self.size, dtype=np.int64),
                rng_state=np.asarray(json.dumps(self.rng.bit_generator.state)),
            )
        temporary_path.replace(path)


class Logger:
    def __init__(
        self,
        output_dir: Path,
        config: dict[str, Any],
        wandb_project: str | None,
        run_name: str,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        self.csv_path = output_dir / "metrics.csv"
        with self.csv_path.open("w", newline="") as handle:
            csv.writer(handle).writerow(("step", "key", "value"))
        (output_dir / "config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n"
        )

        self.wandb_run = None
        if wandb_project:
            import wandb

            self.wandb_run = wandb.init(
                project=wandb_project,
                name=run_name,
                config=config,
            )

    def log(self, step: int, metrics: dict[str, Any]) -> None:
        scalar_metrics = {
            key: float(np.asarray(value)) for key, value in metrics.items()
        }
        with self.csv_path.open("a", newline="") as handle:
            writer = csv.writer(handle)
            for key, value in scalar_metrics.items():
                writer.writerow((step, key, value))
        if self.wandb_run is not None:
            self.wandb_run.log({"step": step, **scalar_metrics}, step=step)

    def finish(self) -> None:
        if self.wandb_run is not None:
            self.wandb_run.finish()


def flatten_observation(observation) -> np.ndarray:
    return np.asarray(observation, dtype=np.float32).reshape(-1)


def evaluate(
    agent: OptiQ,
    environment_name: str,
    episodes: int,
    seed: int,
    stochastic: bool,
) -> tuple[float, float]:
    environment = gym.make(environment_name)
    rng = jax.random.PRNGKey(seed)
    returns = []
    for episode in range(episodes):
        observation, _ = environment.reset(seed=seed + episode)
        observation = flatten_observation(observation)
        terminated = truncated = False
        episode_return = 0.0
        while not (terminated or truncated):
            rng, action_rng = jax.random.split(rng)
            action = np.asarray(
                agent.sample_actions(
                    jnp.asarray(observation[None]),
                    action_rng,
                    deterministic=not stochastic,
                )[0]
            )
            observation, reward, terminated, truncated, _ = environment.step(action)
            observation = flatten_observation(observation)
            episode_return += float(reward)
        returns.append(episode_return)
    environment.close()
    return float(np.mean(returns)), float(np.std(returns))


def save_checkpoint(
    output_dir: Path,
    step: int,
    agent: OptiQ,
    action_rng: jax.Array,
    replay: ReplayBuffer,
    save_replay_buffer: bool,
) -> None:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = f"step_{step:08d}"
    checkpoint_path = checkpoint_dir / f"{checkpoint_name}.msgpack"
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    payload = {
        "step": jnp.asarray(step, dtype=jnp.int32),
        "agent": agent,
        "action_rng": action_rng,
    }
    temporary_path.write_bytes(flax.serialization.to_bytes(payload))
    temporary_path.replace(checkpoint_path)

    if save_replay_buffer:
        replay.save(checkpoint_dir / f"{checkpoint_name}_replay.npz")


def train(args: argparse.Namespace) -> None:
    if args.checkpoint_interval < 0:
        raise ValueError("checkpoint_interval must be non-negative.")

    environment = gym.make(args.env)
    observation, _ = environment.reset(seed=args.seed)
    observation = flatten_observation(observation)
    action_space = environment.action_space
    if not isinstance(action_space, gym.spaces.Box) or len(action_space.shape) != 1:
        raise ValueError("OptiQ requires a one-dimensional continuous Box action space.")
    action_low = np.asarray(action_space.low, np.float32)
    action_high = np.asarray(action_space.high, np.float32)

    agent_config = OptiQConfig(
        actor_hidden_dims=args.hidden_dims,
        critic_hidden_dims=args.hidden_dims,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        discount=args.discount,
        target_tau=args.target_tau,
        actor_update_frequency=args.actor_update_frequency,
        num_particles=args.num_particles,
        proposals_per_particle=args.proposals_per_particle,
        proposal_std=args.proposal_std,
        proposal_clip=args.proposal_clip,
        include_anchor=args.include_anchor,
        temperature=args.temperature,
        sinkhorn_epsilon=args.sinkhorn_epsilon,
        sinkhorn_iterations=args.sinkhorn_iterations,
        td_noise_std=args.td_noise_std,
        td_noise_clip=args.td_noise_clip,
    )
    agent = OptiQ.create(
        args.seed,
        observation.size,
        action_low,
        action_high,
        agent_config,
    )
    replay = ReplayBuffer(
        args.replay_capacity,
        observation.size,
        action_low.size,
        args.seed + 1,
    )

    run_name = args.run_name or f"optiq_{args.env}_seed{args.seed}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{run_name}_{timestamp}"
    full_config = {**vars(args), "agent": asdict(agent.config)}
    full_config["output_dir"] = str(full_config["output_dir"])
    logger = Logger(output_dir, full_config, args.wandb_project, run_name)

    np_rng = np.random.default_rng(args.seed + 2)
    action_rng = jax.random.PRNGKey(args.seed + 3)
    episode_return = 0.0
    episode_length = 0
    last_update_metrics: dict[str, Any] = {}
    last_checkpoint_step = 0

    for step in range(1, args.total_steps + 1):
        if step <= args.warmup_steps:
            action = np_rng.uniform(action_low, action_high).astype(np.float32)
        else:
            action_rng, sample_rng = jax.random.split(action_rng)
            action = np.asarray(
                agent.sample_actions(jnp.asarray(observation[None]), sample_rng)[0]
            )
            if args.behavior_noise_std > 0.0:
                action += np_rng.normal(
                    0.0, args.behavior_noise_std, size=action.shape
                ).astype(np.float32)
                action = np.clip(action, action_low, action_high)

        next_observation, reward, terminated, truncated, _ = environment.step(action)
        next_observation = flatten_observation(next_observation)
        replay.add(observation, action, reward, next_observation, terminated)
        episode_return += float(reward)
        episode_length += 1

        if step > args.warmup_steps and replay.size >= args.batch_size:
            for _ in range(args.updates_per_step):
                agent, last_update_metrics = agent.update(replay.sample(args.batch_size))

        if terminated or truncated:
            logger.log(
                step,
                {
                    "train/episode_return": episode_return,
                    "train/episode_length": episode_length,
                },
            )
            observation, _ = environment.reset()
            observation = flatten_observation(observation)
            episode_return = 0.0
            episode_length = 0
        else:
            observation = next_observation

        if args.log_interval > 0 and step % args.log_interval == 0:
            logger.log(step, {"replay/size": replay.size, **last_update_metrics})

        if args.eval_interval > 0 and step % args.eval_interval == 0:
            deterministic_return, deterministic_std = evaluate(
                agent, args.env, args.eval_episodes, args.seed + 10_000 + step, False
            )
            stochastic_return, stochastic_std = evaluate(
                agent, args.env, args.eval_episodes, args.seed + 20_000 + step, True
            )
            logger.log(
                step,
                {
                    "eval/episode_return": deterministic_return,
                    "eval/episode_return_std": deterministic_std,
                    "eval/stochastic_episode_return": stochastic_return,
                    "eval/stochastic_episode_return_std": stochastic_std,
                },
            )

        if (
            args.save_checkpoints
            and args.checkpoint_interval > 0
            and step % args.checkpoint_interval == 0
        ):
            save_checkpoint(
                output_dir,
                step,
                agent,
                action_rng,
                replay,
                args.save_replay_buffer,
            )
            last_checkpoint_step = step

    if args.save_checkpoints and last_checkpoint_step != args.total_steps:
        save_checkpoint(
            output_dir,
            args.total_steps,
            agent,
            action_rng,
            replay,
            args.save_replay_buffer,
        )
    logger.finish()
    environment.close()


def parse_hidden_dims(value: str) -> tuple[int, ...]:
    dimensions = tuple(int(item) for item in value.split(",") if item)
    if not dimensions or any(dimension < 1 for dimension in dimensions):
        raise argparse.ArgumentTypeError("hidden dims must be comma-separated positives")
    return dimensions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--warmup-steps", type=int, default=10_000)
    parser.add_argument("--replay-capacity", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--behavior-noise-std", type=float, default=0.0)

    parser.add_argument("--hidden-dims", type=parse_hidden_dims, default=(256, 256, 256))
    parser.add_argument("--actor-lr", type=float, default=3.0e-4)
    parser.add_argument("--critic-lr", type=float, default=3.0e-4)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--target-tau", type=float, default=0.005)
    parser.add_argument("--actor-update-frequency", type=int, default=1)

    parser.add_argument("--num-particles", type=int, default=16)
    parser.add_argument("--proposals-per-particle", type=int, default=5)
    parser.add_argument("--proposal-std", type=float, default=0.2)
    parser.add_argument("--proposal-clip", type=float, default=0.5)
    parser.add_argument(
        "--include-anchor", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=0.05)
    parser.add_argument("--sinkhorn-iterations", type=int, default=30)
    parser.add_argument("--td-noise-std", type=float, default=0.2)
    parser.add_argument("--td-noise-clip", type=float, default=0.5)

    parser.add_argument("--eval-interval", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=5_000)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument(
        "--save-checkpoints", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=0,
        help="Periodic checkpoint interval; zero saves only the final state.",
    )
    parser.add_argument(
        "--save-replay-buffer", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--wandb-project")
    parser.add_argument("--run-name")
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
