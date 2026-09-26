"""Audit learned OptiQ Q on the four cardinal corridors with policy Monte Carlo.

Each sampled first action is held fixed across independent continuations.
States away from the origin are diagnostic starts, not the training reset law.
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


def audit(checkpoint: Path, config_path: Path, *, samples: int = 8,
          continuations: int = 32, seed: int = 20260926):
    if samples < 1 or continuations < 2:
        raise ValueError("need at least one action and two continuations")
    config = json.loads(config_path.read_text())
    if config["task"] != "4way" or config["method"] != "optiq":
        raise ValueError("expected a wall-free 4-Way OptiQ checkpoint")
    temperature = float(config["agent"]["alg"]["actor"]["temperature"])
    positions = np.array([[0., 0.]] +
                         [[r, 0.] for r in (1., 2., 3., 4.)] +
                         [[-r, 0.] for r in (1., 2., 3., 4.)] +
                         [[0., r] for r in (1., 2., 3., 4.)] +
                         [[0., -r] for r in (1., 2., 3., 4.)], np.float32)
    with TemporaryDirectory(prefix="fourway-q-surface-") as temporary:
        agent = OptiQ(seed=config["seed"], folder=Path(temporary),
                      budget=config["steps"], observation_dim=2,
                      batch_size=config["batch_size"], temperature=temperature)
        policy = agent.model.policy
        template = dict(actor=policy.actor_state, critic=policy.qf_state,
                        target_actor=policy.target_actor_state,
                        target_critic_params=policy.qf_state.target_params)
        restored = flax.serialization.from_bytes(template, checkpoint.read_bytes())
        policy.actor_state = restored["actor"]
        policy.qf_state = restored["critic"]
        policy.target_actor_state = restored["target_actor"]
        starts = np.repeat(positions, samples, axis=0)
        with agent.evaluation_rng(seed):
            actions = agent.act(starts, mode="mu_only")
            q = agent.q(starts, actions)
            initial = np.repeat(starts, continuations, axis=0)
            first_actions = np.repeat(actions, continuations, axis=0)
            env = TaskBatch("4way", count=len(initial), seed=seed)
            env.batch.state[:] = initial
            env.current[:] = initial
            discounted = np.zeros(len(initial), np.float64)
            active = np.ones(len(initial), bool)
            try:
                for t in range(env.horizon):
                    action = first_actions if t == 0 else agent.act(env.current, mode="mu_only")
                    _, reward, terminated, truncated, _ = env.step(action, active=active)
                    discounted[active] += (.99 ** t) * reward[active]
                    active &= ~(terminated | truncated)
                    if not active.any():
                        break
            finally:
                env.close()
        mc = discounted.reshape(len(positions), samples, continuations)
        q = q.reshape(len(positions), samples)
        if not np.isfinite(q).all() or not np.isfinite(mc).all():
            raise FloatingPointError("nonfinite Q or Monte Carlo return")
        mean_mc = mc.mean(-1)
        error = q - mean_mc
        rows = []
        for state, state_q, state_mc, state_error in zip(positions, q, mean_mc, error):
            rows.append(dict(state_xy=state.tolist(), q_mean=float(state_q.mean()),
                             mc_mean=float(state_mc.mean()), q_minus_mc=float(state_error.mean()),
                             mean_absolute_error=float(np.abs(state_error).mean())))
        return dict(training_source_commit=config["source_commit"], checkpoint=str(checkpoint),
                    state_scope="origin and 1-4 units along each cardinal corridor",
                    action_scope="fresh random-z mu-only; MC continuation differs from full-policy training",
                    seed=seed, samples_per_state=samples,
                    continuations_per_action=continuations, gamma=.99,
                    mean_absolute_error=float(np.abs(error).mean()),
                    root_mean_square_error=float(np.sqrt(np.square(error).mean())),
                    mean_bias=float(error.mean()), rows=rows)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--continuations", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.checkpoint, args.config, samples=args.samples,
                   continuations=args.continuations, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output, result["mean_absolute_error"], result["mean_bias"])


if __name__ == "__main__":
    main()
