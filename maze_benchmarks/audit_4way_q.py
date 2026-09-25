"""Compare the plotted 4-Way critic with on-policy Monte Carlo returns.

At the common origin, force one cardinal first action, then follow the frozen
policy. This tests the Q value of each first action without changing training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import flax.serialization
import numpy as np

from .agents import OptiQ
from .envs import TaskBatch


DIRECTIONS = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]], np.float32)
NAMES = ("east", "west", "north", "south")


def audit(checkpoint: Path, config_path: Path, episodes: int, seed: int) -> dict:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    config = json.loads(config_path.read_text())
    if config["task"] != "4way" or config["method"] != "optiq":
        raise ValueError("expected a wall-free OptiQ 4-Way run")
    # The first T=3 run predates the explicit CLI temperature flag, but its
    # resolved and saved actor config contains the actual training value.
    temperature = float(config.get("temperature", config["agent"]["alg"]["actor"]["temperature"]))
    with TemporaryDirectory(prefix="fourway-q-audit-") as temporary:
        agent = OptiQ(seed=config["seed"], folder=Path(temporary),
                      budget=config["steps"], observation_dim=2,
                      batch_size=config["batch_size"],
                      temperature=temperature)
        policy = agent.model.policy
        template = dict(actor=policy.actor_state, critic=policy.qf_state,
                        target_actor=policy.target_actor_state,
                        target_critic_params=policy.qf_state.target_params)
        restored = flax.serialization.from_bytes(template, checkpoint.read_bytes())
        policy.actor_state = restored["actor"]
        policy.qf_state = restored["critic"]
        policy.target_actor_state = restored["target_actor"]
        q_values = agent.q(np.zeros((4, 2), np.float32), DIRECTIONS)
        if not np.isfinite(q_values).all():
            raise FloatingPointError("nonfinite first-action Q")

        results = {}
        count = 4 * episodes
        first_actions = np.repeat(DIRECTIONS, episodes, axis=0)
        for mode in ("policy", "mu_only"):
            environment = TaskBatch("4way", count=count, seed=seed)
            active = np.ones(count, bool)
            discounted = np.zeros(count, np.float64)
            goal_ids = np.full(count, -1, np.int8)
            try:
                with agent.evaluation_rng(seed):
                    for t in range(environment.horizon):
                        actions = first_actions if t == 0 else agent.act(environment.current, mode=mode)
                        _, reward, terminated, truncated, goals = environment.step(actions, active=active)
                        discounted[active] += (.99 ** t) * reward[active]
                        finished = active & (terminated | truncated)
                        goal_ids[finished & terminated] = goals[finished & terminated]
                        active[finished] = False
                        if not active.any():
                            break
            finally:
                environment.close()
            mode_result = {}
            for index, name in enumerate(NAMES):
                sl = slice(index * episodes, (index + 1) * episodes)
                group_returns = discounted[sl]
                group_goals = goal_ids[sl]
                mode_result[name] = {
                    "first_action_q": float(q_values[index]),
                    "discounted_return_mean": float(group_returns.mean()),
                    "discounted_return_sd": float(group_returns.std(ddof=1)) if episodes > 1 else 0.,
                    "q_minus_mc": float(q_values[index] - group_returns.mean()),
                    "success_rate": float(np.mean(group_goals >= 0)),
                    "goal_counts": np.bincount(group_goals[group_goals >= 0], minlength=4).tolist(),
                }
            results[mode] = mode_result
        return {
            "checkpoint": str(checkpoint),
            "source_commit": config["source_commit"],
            "temperature": temperature,
            "gamma": .99,
            "forced_first_actions": DIRECTIONS.tolist(),
            "episodes_per_first_action": episodes,
            "seed": seed,
            "value_aggregation": agent.critic_label,
            "results": results,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.checkpoint, args.config, args.episodes, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
