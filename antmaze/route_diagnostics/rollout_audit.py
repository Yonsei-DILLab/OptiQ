"""Read-only checkpoint rollouts; never updates a learner or writes its run directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np

TRAINING_SHA = "19fc37a7e51be2bb54d41ab9a980225ecd9f5ba5"
TRAINING_ROOT = Path("/home/heechan/OptiQ-ops/sources") / TRAINING_SHA
CAMPAIGN = Path("/home/heechan/optiq-experiments/antmaze-dense-noveld-1m-s0-20260922")
sys.path.insert(0, str(TRAINING_ROOT))

from antmaze.evaluation import atomic_json, isolated_rng
from antmaze.multimodal.env import make_env
from antmaze.multimodal.policy import PolicyView
from antmaze.multimodal.analysis import summarize
from antmaze.multimodal.resume import restore_model, model_state, digest


def rollout(agent, task, mode, out, episodes=100):
    batch = 10
    envs = [make_env(task) for _ in range(batch)]
    horizon = envs[0].horizon
    rng = np.random.default_rng(np.random.SeedSequence([20260922, 700001, int(task[1:])]))
    env_seeds = rng.integers(0, 2**30, episodes)
    policy_seeds = rng.integers(0, 2**30, episodes // batch)
    xy = np.full((episodes, horizon+1, 2), np.nan, np.float32)
    observations = np.full((episodes, horizon+1, 29), np.nan, np.float32)
    actions = np.full((episodes, horizon, 8), np.nan, np.float32)
    returns = np.zeros(episodes)
    lengths = np.zeros(episodes, int)
    goals = np.zeros(episodes, int)
    initial_simulator = []
    view = PolicyView(agent)
    for start in range(0, episodes, batch):
        obs = []
        for i, env in enumerate(envs):
            o, info = env.reset(seed=int(env_seeds[start+i]), options={"fixed_start": True})
            obs.append(o)
            xy[start+i, 0] = info["xy"]
            observations[start+i, 0] = o
            initial_simulator.append(env.simulator_state())
        obs = np.stack(obs)
        done = np.zeros(batch, bool)
        ps = int(policy_seeds[start // batch])
        with isolated_rng(ps), view.evaluation("native" if mode == "episode_latent_mu" else mode, ps):
            if mode == "episode_latent_mu":
                p = agent.model.policy
                p.reset_noise()
                held_key = p.noise_key
            while not done.all():
                if mode == "episode_latent_mu":
                    a = np.asarray(p.sample_action(p.actor_state, obs, held_key,
                        deterministic=False, sample_conditional_noise=False))
                else:
                    a = np.asarray(view.act(obs))
                assert a.shape == (batch, 8) and np.isfinite(a).all()
                for i, env in enumerate(envs):
                    if done[i]:
                        continue
                    ix, t = start+i, lengths[start+i]
                    actions[ix, t] = a[i]
                    obs[i], reward, term, trunc, info = env.step(a[i])
                    lengths[ix] += 1
                    returns[ix] += reward
                    goals[ix] = info["goal_id"]
                    xy[ix, t+1] = info["xy"]
                    observations[ix, t+1] = obs[i]
                    done[i] = term or trunc
        atomic_json(out / "progress.json", dict(task=task, mode=mode, episodes_done=start+batch))
    states = np.asarray(initial_simulator)
    np.testing.assert_array_equal(states, np.broadcast_to(states[0], states.shape))
    summary = summarize(task, xy, lengths, goals, returns)
    summary.update(task=task, training_seed=0, checkpoint_step=1000000, mode=mode,
        evaluation_rng_seed=[20260922, 700001, int(task[1:])], fixed_full_state=True,
        added_action_noise=False, intrinsic_reward_in_evaluation=False,
        altered_policy_diagnostic=mode == "episode_latent_mu")
    np.savez_compressed(out / f"{mode}.npz", xy=xy, observations=observations, actions=actions,
        returns=returns, lengths=lengths, goal_ids=goals, initial_simulator_state=states,
        env_seeds=env_seeds, policy_batch_seeds=policy_seeds)
    atomic_json(out / f"{mode}.json", summary)
    for env in envs:
        env.close()
    print(json.dumps({k:v for k,v in summary.items() if k != "routes"}), flush=True)
    return summary, observations[0, 0]


def probe(agent, obs, out):
    modes = ["policy", "native"] if getattr(agent, "method", None) == "optiq" else ["policy"]
    arrays, summary = {}, {}
    view = PolicyView(agent)
    repeated = np.broadcast_to(obs, (64, 29)).copy()
    for mode in modes:
        with isolated_rng(830019), view.evaluation(mode, 830019):
            a = np.concatenate([np.asarray(view.act(repeated)) for _ in range(16)])
        assert a.shape == (1024, 8) and np.isfinite(a).all()
        arrays[mode] = a
        summary[mode] = dict(unique_actions=int(len(np.unique(a, axis=0))),
            mean=a.mean(0).tolist(), std=a.std(0).tolist(),
            covariance_trace=float(np.trace(np.cov(a.T))),
            max_abs=float(np.max(np.abs(a))))
    np.savez_compressed(out / "initial_state_actions.npz", observation=obs, **arrays)
    atomic_json(out / "initial_state_actions.json", summary)


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--method", choices=["optiq", "sac", "mfpo", "meow"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--diagnostic-source-sha", required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    random.seed(0)
    np.random.seed(0)
    assert torch.cuda.is_available()
    assert subprocess.check_output(["git", "-C", str(TRAINING_ROOT), "rev-parse", "HEAD"], text=True).strip() == TRAINING_SHA
    from antmaze.agents import SB3, MEOW, MFPO
    started = time.monotonic()
    results = {}
    for task in (["v3", "v1"] if a.method == "optiq" else ["v3"]):
        out = a.output / task
        out.mkdir()
        run = CAMPAIGN / "runs" / f"{task}-{a.method}-s0"
        config = json.loads((run / "config.json").read_text())
        checkpoint = run / "resume" / "step_0001000000"
        manifest = json.loads((checkpoint / "manifest.json").read_text())
        assert manifest["source_commit"] == config["source_commit"] == TRAINING_SHA
        assert manifest["step"] == 1000000 and config["seed"] == 0
        file = checkpoint / "state.pt"
        with file.open("rb") as f:
            sha = hashlib.file_digest(f, "sha256").hexdigest()
        assert sha == manifest["files"]["state.pt"]["sha256"]
        env = make_env(task, 0)
        agent = SB3(a.method, env, 0, out) if a.method in ("optiq", "sac") else {"meow": MEOW, "mfpo": MFPO}[a.method](env, 0, out)
        state = torch.load(file, map_location="cpu", weights_only=False)
        assert state["extra"]["task"] == task and state["extra"]["method"] == a.method
        restore_model(agent, state["model"])
        before = digest(model_state(agent))
        assert before == digest(state["model"])
        atomic_json(out / "provenance.json", dict(training_source=TRAINING_SHA,
            diagnostic_source=a.diagnostic_source_sha, checkpoint=str(checkpoint),
            checkpoint_sha256=sha, config=config, model_digest=before,
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        modes = ["policy", "native", "episode_latent_mu"] if a.method == "optiq" and task == "v3" else ["policy"]
        for mode in modes:
            summary, initial_obs = rollout(agent, task, mode, out)
            results[f"{task}-{mode}"] = summary
        probe(agent, initial_obs, out)
        assert digest(model_state(agent)) == before, "Evaluation changed trained model or its RNG"
        atomic_json(out / "verification.json", dict(passed=True, model_unchanged=True,
            checkpoint_step=1000000, training_seed=0, source_commit=TRAINING_SHA,
            npz_sha256={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in out.glob("*.npz")}))
        env.close()
        del agent, state
    atomic_json(a.output / "result.json", dict(completed=True, training_unchanged=True,
        method=a.method, seconds=time.monotonic()-started, results=results))


if __name__ == "__main__":
    main()
