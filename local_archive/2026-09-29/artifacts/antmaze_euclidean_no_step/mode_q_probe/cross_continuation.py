"""Read-only intervention: hold initial teacher actions fixed, swap future actors."""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("v3", "v4"), required=True)
    parser.add_argument("--replicates", type=int, default=2)
    args = parser.parse_args()
    assert 1 <= args.replicates <= 4
    source = Path("/home/heechan/OptiQ-ops/sources/f953d28456d3800860dddb9b9cb91b6bd520ae00")
    root = Path("/home/heechan/optiq-experiments/antmaze-optiq-euclidean-no-step-B0-T1-s0-20260925-r2")
    run = root / "runs" / f"{args.task}-optiq-euclidean-no-step-B0-T1-s0"
    config = json.loads((run / "config.json").read_text())
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
    native.output_root = "/tmp/optiq-antmaze-cross-continuation-readonly"
    model = module.runner.OptiQDIME(
        "MlpPolicy", env=Descriptor(), cfg=native, model_save_path=None,
        save_every_n_steps=config["steps"]
    )
    policy = model.policy
    from optiq_dime.semi_implicit import ConditionalGaussianProposal
    template = dict(actor=policy.actor_state, critic=policy.qf_state,
                    target_actor=policy.target_actor_state)
    states = {}
    hashes = {}
    for step in (250112, 500224):
        file = run / "policy-checkpoints" / f"step_{step:010d}" / "policy.pt"
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        assert digest == json.loads((file.parent / "verification.json").read_text())["sha256"]
        payload = torch.load(file, map_location="cpu", weights_only=False)
        assert payload["config"] == config
        states[step] = fs.from_bytes(template, payload["learner"]["policy"])
        hashes[str(step)] = digest

    env = vector(args.task, 64, seed=87231, asynchronous=False, fixed=True,
                 reward_profile=config["reward_profile"], random_init=False)
    try:
        observations = np.asarray(env.reset(), dtype=np.float32)
        first = jnp.asarray(observations[:1])
        actor = states[250112]["actor"]
        _, latent_key, proposal_key, _ = jax.random.split(jax.random.PRNGKey(887231), 4)
        z_key, _ = jax.random.split(latent_key)
        z = jax.random.normal(z_key, (64, 8))
        mu, log_std = actor.apply_fn(
            {"params": actor.params}, jnp.repeat(first, 64, axis=0), z
        )
        proposal = ConditionalGaussianProposal(
            mu[None], log_std[None], float(native.alg.actor.proposal_std)
        )
        sampled, _, _ = proposal.sample(proposal_key, 1, "exact")
        candidates = np.asarray(sampled[0])

        @jax.jit
        def next_actions(actor_state, obs, key):
            return jax.vmap(
                lambda one: policy.sample_action(
                    actor_state, one[None], key,
                    deterministic=False, sample_conditional_noise=True,
                )[0]
            )(obs)

        output = {"task": args.task, "source": config["source_commit"],
                  "checkpoint_sha256": hashes,
                  "candidate_actor_step": 250112,
                  "candidate_count": 64,
                  "continuation_mode": "random-z and conditional sigma",
                  "frozen_initial_full_state": True,
                  "read_only": True,
                  "results": {}}
        for continuation_step in (250112, 500224):
            rep_rows = []
            for rep in range(args.replicates):
                observations = np.asarray(env.reset(), dtype=np.float32)
                common = env.envs[0].initial
                for i, single in enumerate(env.envs):
                    single.initial = common
                    observations[i] = single.restore(common)
                paths = [[o[:2].copy()] for o in observations]
                active = np.ones(64, dtype=bool)
                for t in range(700):
                    if t == 0:
                        actions = candidates
                    else:
                        actions = np.asarray(next_actions(
                            states[continuation_step]["actor"],
                            jnp.asarray(observations),
                            jax.random.PRNGKey(900001 + rep * 1000 + t),
                        ))
                    nxt, _, done, infos = env.step(actions)
                    final, _ = transition(nxt, done, infos)
                    for i in np.flatnonzero(active):
                        paths[i].append(final[i, :2].copy())
                        if done[i]:
                            active[i] = False
                    observations = nxt
                    if not active.any():
                        break
                labels = []
                for path in paths:
                    xy = np.asarray(path)
                    if args.task == "v3":
                        left = bool(np.any(xy[:, 0] < -8))
                        right = bool(np.any(xy[:, 0] > 8))
                        labels.append("both" if left and right else
                                      "left" if left else "right" if right else "uncommitted")
                    else:
                        crossings = np.flatnonzero((xy[:-1, 0] > -4) & (xy[1:, 0] <= -4))
                        if len(crossings):
                            j = int(crossings[0])
                            y = xy[j, 1] + (-4 - xy[j, 0]) / (xy[j + 1, 0] - xy[j, 0]) * (xy[j + 1, 1] - xy[j, 1])
                            labels.append("upper" if y > 2 else "lower" if y < -2 else "uncommitted")
                        else:
                            labels.append("uncommitted")
                counts = {label: labels.count(label) for label in sorted(set(labels))}
                row = {"replicate": rep, "route_counts": counts}
                rep_rows.append(row)
                print(json.dumps({"continuation_step": continuation_step, **row}), flush=True)
            output["results"][str(continuation_step)] = rep_rows
        output["limitation"] = (
            "One 64-action proposal cloud from the earlier actor; changing the future actor "
            "is a post-hoc counterfactual, not a causal intervention during training."
        )
        print(json.dumps({"final_result": output}), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
