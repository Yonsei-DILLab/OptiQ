"""Online vector collection and checkpointed evaluation for four-goal tasks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np

from .envs import TaskBatch
from .agents import OptiQ, SAC, SQL, MEOW, MFPO, DIPO, TD3
from .visualize_4way import probe_policy
from .visualize_nway import render as render_nway
from .nway import GOAL_RADIUS, HORIZON as NWAY_HORIZON, STATE_LIMIT, SUCCESS_RADIUS, SUPPORTED_GOALS


AGENTS = {"optiq": OptiQ, "sac": SAC, "sql": SQL, "meow": MEOW,
          "mfpo": MFPO, "dipo": DIPO, "td3": TD3}
NWAY_TASKS = tuple(f"{count}way" for count in SUPPORTED_GOALS)


def atomic_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def evaluate(agent, task, seed, episodes, mode, destination):
    if episodes <= 0:
        raise ValueError(episodes)
    horizon = 300 if task == "pointmaze" else (20 if task == "4way" else NWAY_HORIZON)
    dim = 4 if task == "pointmaze" else 2
    histories = np.full((episodes, horizon + 1, 2), np.nan, np.float32)
    returns = np.zeros(episodes, np.float32)
    goal_ids = np.full(episodes, -1, np.int8)
    lengths = np.zeros(episodes, np.int32)
    for offset in range(0, episodes, 128):
        stop = min(offset + 128, episodes)
        env = TaskBatch(task, count=stop - offset, seed=seed + offset)
        active = np.ones(stop - offset, bool)
        history_chunk = histories[offset:stop]
        return_chunk = returns[offset:stop]
        goal_chunk = goal_ids[offset:stop]
        length_chunk = lengths[offset:stop]
        history_chunk[:, 0] = env.current[:, :2]
        try:
            with agent.evaluation_rng(seed + offset):
                for index in range(horizon):
                    if not active.any():
                        break
                    # Every active policy sees its own uninterrupted state path.
                    observations = env.current.copy()
                    actions = agent.act(observations, mode=mode)
                    next_obs, rewards, terminated, truncated, goals = env.step(actions, active)
                    history_chunk[active, index + 1] = next_obs[active, :2]
                    return_chunk[active] += rewards[active]
                    length_chunk[active] += 1
                    goal_chunk[active & terminated] = goals[active & terminated]
                    active &= ~(terminated | truncated)
        finally:
            env.close()
    if not np.isfinite(returns).all() or np.any(lengths == 0):
        raise FloatingPointError("invalid evaluation rollout")
    np.savez_compressed(destination, xy=histories, returns=returns,
                        lengths=lengths, goal_ids=goal_ids, mode=mode,
                        observation_dim=dim)
    goal_count = int(task[:-3]) if task in NWAY_TASKS else 4
    counts = np.bincount(goal_ids[goal_ids >= 0], minlength=goal_count)
    return dict(episodes=episodes, success=float(np.mean(goal_ids >= 0)),
                goals=counts.tolist(), failure=int(np.sum(goal_ids < 0)),
                mean_return=float(returns.mean()), mean_length=float(lengths.mean()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("4way", "pointmaze", *NWAY_TASKS), required=True)
    parser.add_argument("--method", choices=tuple(AGENTS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--updates-per-collect", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=1_024)
    parser.add_argument("--eval-every", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--final-eval-episodes", type=int, default=None)
    parser.add_argument("--render-each-eval", action="store_true")
    parser.add_argument("--temperature", type=float, default=3.)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if args.steps % args.num_envs or args.warmup % args.num_envs:
        raise ValueError("steps and warmup must be multiples of num-envs")
    if args.warmup < args.batch_size:
        raise ValueError("warmup must fill one update batch")
    if args.updates_per_collect <= 0 or args.eval_every <= 0 or not np.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError("invalid updates or evaluation cadence")
    if args.eval_episodes <= 0 or (args.final_eval_episodes is not None and args.final_eval_episodes <= 0):
        raise ValueError("evaluation episode counts must be positive")
    if args.render_each_eval and args.task not in NWAY_TASKS:
        raise ValueError("automatic N-Way figures require an N-Way task")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "evaluations").mkdir()
    (args.output / "checkpoints").mkdir()
    if args.render_each_eval:
        (args.output / "figures").mkdir()
    np.random.seed(args.seed)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    observation_dim = 4 if args.task == "pointmaze" else 2
    kwargs = dict(temperature=args.temperature) if args.method in ("optiq", "sql") else {}
    if args.method == "dipo":
        kwargs.update(task=args.task, horizon=20 if args.task == "4way" else
                      (300 if args.task == "pointmaze" else NWAY_HORIZON))
    agent = AGENTS[args.method](args.seed, args.output, args.steps,
                                observation_dim, args.batch_size, **kwargs)
    environment = TaskBatch(args.task, count=args.num_envs, seed=args.seed)
    config = vars(args).copy()
    config["output"] = str(args.output)
    config["agent"] = agent.config
    config["task_mode"] = ("existing wall-free symmetric four-goal" if args.task == "4way"
                           else "Farama PointMaze adapted four-goal" if args.task == "pointmaze"
                           else "wall-free symmetric N-goal ring")
    if args.task in NWAY_TASKS:
        config["geometry"] = dict(goal_positions=environment.goal_positions.tolist(),
                                  goal_radius=GOAL_RADIUS, success_radius=SUCCESS_RADIUS,
                                  state_limit=STATE_LIMIT, horizon=NWAY_HORIZON,
                                  reward="-30*||a||^2 - nearest_goal_squared_distance + 10*success")
    config["utd"] = args.updates_per_collect / args.num_envs
    atomic_json(args.output / "config.json", config)
    progress = dict(status="running", steps=0, updates=0,
                    latest_evaluation=None, started=time.time())
    atomic_json(args.output / "progress.json", progress)
    try:
        for step in range(args.num_envs, args.steps + 1, args.num_envs):
            observations = environment.current.copy()
            if step <= args.warmup:
                actions = rng.uniform(-1., 1., (args.num_envs, 2)).astype(np.float32)
            else:
                actions = agent.act(observations, mode="train")
            next_obs, rewards, terminated, truncated, goals = environment.step(actions)
            if not np.isfinite(rewards).all() or not np.isfinite(next_obs).all():
                raise FloatingPointError(f"nonfinite environment transition at {step}")
            done = terminated | truncated
            agent.replay.add(observations, actions, rewards, next_obs, done.astype(np.float32))
            if done.any():
                environment.reset(np.flatnonzero(done))
            if step > args.warmup:
                for _ in range(args.updates_per_collect):
                    agent.update(step)
            if step % args.eval_every < args.num_envs or step == args.steps:
                record = dict(step=step, updates=agent.updates, task=args.task,
                              method=args.method, source_commit=args.source_commit)
                episodes = (args.final_eval_episodes if step == args.steps and
                            args.final_eval_episodes is not None else args.eval_episodes)
                for mode in (("policy", "mu_only") if args.method == "optiq" else ("policy",)):
                    destination = args.output / "evaluations" / f"{step:09d}_{mode}.npz"
                    record[mode] = evaluate(agent, args.task,
                                            args.seed + 17_000 + step,
                                            episodes, mode, destination)
                if args.task == "4way" or args.task in NWAY_TASKS:
                    probe = args.output / "evaluations" / f"{step:09d}_probe.npz"
                    probe_policy(agent, probe)
                agent.save(args.output / "checkpoints", step, full=step == args.steps)
                if args.render_each_eval:
                    figure = args.output / "figures" / f"{step:09d}_policy_q_trajectories.png"
                    render_nway(args.task,
                                args.output / "evaluations" / f"{step:09d}_policy.npz",
                                probe, figure,
                                title=f"{args.task.upper()} · {args.method.upper()} · {step:,} steps · seed {args.seed}")
                    record["figure"] = str(figure)
                progress.update(steps=step, updates=agent.updates, latest_evaluation=record,
                                updated=time.time())
                atomic_json(args.output / "progress.json", progress)
                print(json.dumps(record, default=str), flush=True)
        progress["status"] = "complete"
        atomic_json(args.output / "progress.json", progress)
    except BaseException as error:
        atomic_json(args.output / "failure.json", dict(step=step, update=agent.updates,
                                                        error=repr(error), source_commit=args.source_commit))
        raise
    finally:
        environment.close()


if __name__ == "__main__":
    main()
