"""Measure origin action sectors and final goals for one frozen OptiQ policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import flax.serialization
import numpy as np

from .agents import OptiQ
from .run import evaluate


def audit(checkpoint: Path, config_path: Path, destination: Path, episodes: int, seed: int):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    config = json.loads(config_path.read_text())
    if (config["task"], config["method"]) != ("4way", "optiq"):
        raise ValueError("expected OptiQ on wall-free 4-Way")
    temperature = float(config["agent"]["alg"]["actor"]["temperature"])
    destination.mkdir(parents=True, exist_ok=True)
    results = {}
    with TemporaryDirectory(prefix="fourway-action-mass-") as temporary:
        agent = OptiQ(config["seed"], Path(temporary), config["steps"], 2,
                      config["batch_size"], temperature)
        policy = agent.model.policy
        template = dict(actor=policy.actor_state, critic=policy.qf_state,
                        target_actor=policy.target_actor_state,
                        target_critic_params=policy.qf_state.target_params)
        restored = flax.serialization.from_bytes(template, checkpoint.read_bytes())
        policy.actor_state = restored["actor"]
        policy.qf_state = restored["critic"]
        policy.target_actor_state = restored["target_actor"]
        for mode in ("mu_only",):
            raw = destination / f"{mode}_{episodes}.npz"
            summary = evaluate(agent, "4way", seed, episodes, mode, raw)
            with np.load(raw) as data:
                xy = data["xy"]
                goals = data["goal_ids"]
            action = xy[:, 1] - xy[:, 0]
            major_x = np.abs(action[:, 0]) >= np.abs(action[:, 1])
            sector = np.where(major_x, np.where(action[:, 0] >= 0, 0, 1),
                              np.where(action[:, 1] >= 0, 2, 3))
            cross = np.zeros((4, 5), np.int32)
            np.add.at(cross, (sector, np.where(goals >= 0, goals, 4)), 1)
            results[mode] = dict(**summary,
                                 first_action_sectors=np.bincount(sector, minlength=4).tolist(),
                                 first_sector_to_final_goal=cross.tolist(),
                                 raw_rollouts=str(raw))
    result = dict(training_source_commit=config["source_commit"], checkpoint=str(checkpoint),
                  evaluation_seed=seed, episodes=episodes,
                  sector_order=["east", "west", "north", "south"],
                  final_goal_order=["east", "west", "north", "south", "failure"],
                  results=results)
    (destination / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args()
    value = audit(args.checkpoint, args.config, args.output, args.episodes, args.seed)
    print(args.output / "summary.json")
    for mode, result in value["results"].items():
        print(mode, result["first_action_sectors"], result["goals"])


if __name__ == "__main__":
    main()
