"""Read-only matched physical-state Monte Carlo and Q probe across checkpoints."""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("v3", "v4"), required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    source = Path("/home/heechan/OptiQ-ops/sources/23603a7e7696aa64e8e49a38b986d6b430b6d6f3")
    root = Path("/home/heechan/optiq-experiments/antmaze-optiq-critic-dynamics-500k-s0-20260925")
    run = root / "runs" / f"{args.task}-optiq-control-500k-s0"
    config = json.loads((run / "config.json").read_text())
    assert config["reward_profile"] == "progress100_euclidean_no_step_no_bonus"
    sys.path.insert(0, str(source))

    import flax.serialization as fs
    import gymnasium as gym
    import jax
    import jax.numpy as jnp
    import torch
    from omegaconf import OmegaConf
    from antmaze_experiments.envs import vector, transition

    torch.set_num_threads(1)

    class Descriptor(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, (29,), dtype=np.float32)
        action_space = gym.spaces.Box(-1, 1, (8,), dtype=np.float32)

    spec = importlib.util.spec_from_file_location(
        "frozen_trg", source / "analysis_tools/experiments/20260921_gmm_trg_sweep/train.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    native = OmegaConf.create(config["native"])
    native.output_root = "/tmp/optiq-antmaze-branch-state-probe-readonly"
    model = module.runner.OptiQDIME(
        "MlpPolicy", env=Descriptor(), cfg=native, model_save_path=None,
        save_every_n_steps=config["steps"]
    )
    policy = model.policy
    template = dict(actor=policy.actor_state, critic=policy.qf_state,
                    target_actor=policy.target_actor_state)
    states, hashes = {}, {}
    steps = (100096, 200192, 300032, 400128, 500224)
    for step in steps:
        file = run / "policy-checkpoints" / f"step_{step:010d}" / "policy.pt"
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        assert digest == json.loads((file.parent / "verification.json").read_text())["sha256"]
        payload = torch.load(file, map_location="cpu", weights_only=False)
        assert payload["config"] == config
        states[step] = fs.from_bytes(template, payload["learner"]["policy"])
        hashes[str(step)] = digest

    reference = states[300032]["actor"]
    capture_env = vector(args.task, 40, seed=87231, asynchronous=False, fixed=True,
                         reward_profile=config["reward_profile"], random_init=False)
    try:
        obs = np.asarray(capture_env.reset(), dtype=np.float32)
        initial = capture_env.envs[0].initial
        for i, single in enumerate(capture_env.envs):
            single.initial = initial
            obs[i] = single.restore(initial)
        active = np.ones(40, dtype=bool)
        captured = {"first": [], "second": []}
        captured_episode = {"first": set(), "second": set()}
        for t in range(700):
            action = np.asarray(policy.sample_action(
                reference, jnp.asarray(obs), jax.random.PRNGKey(812345 + t),
                deterministic=False, sample_conditional_noise=True,
            ))
            nxt, _, done, info = capture_env.step(action)
            final, _ = transition(nxt, done, info)
            for i in np.flatnonzero(active):
                x, y = final[i, :2]
                if args.task == "v3":
                    side = "first" if x < -8 else "second" if x > 8 else None
                else:
                    side = "first" if x < -4 and y > 2 else "second" if x < -4 and y < -2 else None
                if side and i not in captured_episode[side] and len(captured[side]) < 4:
                    captured[side].append({"state": capture_env.envs[i].state(),
                                           "observation": final[i].copy(),
                                           "reference_episode": int(i), "capture_t": t + 1})
                    captured_episode[side].add(i)
                if done[i]:
                    active[i] = False
            obs = nxt
            if len(captured["first"]) == 4 and len(captured["second"]) == 4:
                break
            if not active.any():
                break
        print(json.dumps({"phase": "capture", "task": args.task,
                          "captured": {k: len(v) for k, v in captured.items()}, "t": t + 1}), flush=True)
    finally:
        capture_env.close()
    assert all(len(v) == 4 for v in captured.values()), "Need four states per route"
    bank = [(side, item) for side in ("first", "second") for item in captured[side]]
    bank_obs = np.stack([item["observation"] for _, item in bank]).astype(np.float32)
    fixed_action = np.asarray(policy.sample_action(
        reference, jnp.asarray(bank_obs), jax.random.PRNGKey(931417),
        deterministic=False, sample_conditional_noise=True,
    ))
    repeat_index = np.repeat(np.arange(8), 2)
    batch_obs = bank_obs[repeat_index]
    batch_action = fixed_action[repeat_index]
    env = vector(args.task, 16, seed=87232, asynchronous=False, fixed=True,
                 reward_profile=config["reward_profile"], random_init=False)
    output = {
        "task": args.task,
        "source": config["source_commit"],
        "reward_profile": config["reward_profile"],
        "checkpoints": hashes,
        "read_only": True,
        "fixed_bank": [{"side": side,
                        "xy": item["observation"][:2].tolist(),
                        "capture_t": item["capture_t"],
                        "reference_episode": item["reference_episode"]}
                       for side, item in bank],
        "replicates_per_bank_state": 2,
        "results": {},
    }
    try:
        @jax.jit
        def evaluate_q(critic_state, observations, actions):
            return critic_state.apply_fn(
                {"params": critic_state.params, "batch_stats": critic_state.batch_stats},
                observations, actions, train=False,
            )[..., 0]

        for step in steps:
            critic = states[step]["critic"]
            qs = np.asarray(evaluate_q(critic, jnp.asarray(bank_obs),
                                       jnp.asarray(fixed_action)))
            assert qs.shape == (2, 8)
            obs = np.asarray(env.reset(), dtype=np.float32)
            for j, bank_id in enumerate(repeat_index):
                obs[j] = env.envs[j].restore(bank[bank_id][1]["state"])
            np.testing.assert_allclose(obs, batch_obs, atol=1e-5)
            active = np.ones(16, dtype=bool)
            returns = np.zeros(16, dtype=np.float64)
            lengths = np.zeros(16, dtype=np.int32)
            success = np.zeros(16, dtype=bool)
            min_dist = np.full(16, np.inf)
            last_dist = np.full(16, np.nan)
            final_xy = np.zeros((16, 2), dtype=np.float32)
            for t in range(700):
                if t == 0:
                    action = batch_action
                else:
                    action = np.asarray(policy.sample_action(
                        states[step]["actor"], jnp.asarray(obs),
                        jax.random.PRNGKey(940000 + t),
                        deterministic=False, sample_conditional_noise=True,
                    ))
                nxt, reward, done, info = env.step(action)
                final, _ = transition(nxt, done, info)
                for j in np.flatnonzero(active):
                    returns[j] += (.99 ** t) * float(reward[j])
                    lengths[j] += 1
                    distance = float(info[j]["distance"])
                    min_dist[j] = min(min_dist[j], distance)
                    last_dist[j] = distance
                    final_xy[j] = final[j, :2]
                    if done[j]:
                        active[j] = False
                        success[j] = bool(info[j].get("success", False))
                obs = nxt
                if not active.any():
                    break
            raw = []
            for j, (side, item) in enumerate(bank):
                raw.append({
                    "bank_id": j, "side": side,
                    "q1": float(qs[0, j]), "q2": float(qs[1, j]),
                    "q_mean": float(qs[:, j].mean()), "q_min": float(qs[:, j].min()),
                    "mc_returns": returns[2 * j:2 * j + 2].tolist(),
                    "successes": success[2 * j:2 * j + 2].astype(int).tolist(),
                    "closest_goal_m": min_dist[2 * j:2 * j + 2].tolist(),
                    "final_goal_m": last_dist[2 * j:2 * j + 2].tolist(),
                    "lengths": lengths[2 * j:2 * j + 2].tolist(),
                    "final_xy": final_xy[2 * j:2 * j + 2].tolist(),
                })
            groups = {}
            for side in ("first", "second"):
                rows = [r for r in raw if r["side"] == side]
                groups[side] = {
                    "q_mean": float(np.mean([r["q_mean"] for r in rows])),
                    "mc_mean": float(np.mean([v for r in rows for v in r["mc_returns"]])),
                    "q_minus_mc": float(np.mean([
                        r["q_mean"] - v for r in rows for v in r["mc_returns"]
                    ])),
                    "successes": int(sum(sum(r["successes"]) for r in rows)),
                    "closest_goal_m": float(np.mean([
                        v for r in rows for v in r["closest_goal_m"]
                    ])),
                }
            output["results"][str(step)] = {"groups": groups, "raw": raw}
            print(json.dumps({"phase": "checkpoint", "task": args.task,
                              "step": step, "groups": groups}), flush=True)
        output["limitations"] = [
            "Four route states per side captured from one 300k policy and two common-random continuations per state.",
            "Frozen first action is sampled from 300k actor; continuation actor changes by checkpoint.",
            "The terminal time-limit convention bootstraps whereas these Monte Carlo returns end at timeout.",
            "Different side states have different coordinates; compare Q to same-state MC rather than raw Q across sides.",
            "This is post-hoc inference and does not establish the learning-time initiator of a policy change."
        ]
        args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"phase": "completed", "task": args.task,
                          "result_path": str(args.output)}), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
