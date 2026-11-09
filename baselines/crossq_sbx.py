"""CrossQ baseline (reference SBX implementation) under the iBOLT MuJoCo protocol.

Algorithm: sbx.CrossQ with its defaults (BatchRenorm critic and actor, no target
network, Adam lr 1e-3 with b1=0.5, policy delay 3, auto entropy, UTD 1, batch 256),
critic width 2048x2 as in the CrossQ paper, actor 256x2.
Protocol (same as iBOLT): Gymnasium v4 tasks, 1M interactions, 5K random warmup,
evaluation every 5K steps with 10 episodes whose reset seeds are generated exactly
as in ibolt/dual_evaluation.py. Two evaluation modes are logged:
  deterministic: tanh of the Gaussian mean (the usual CrossQ/SAC evaluation)
  stochastic:    an action sampled from the tanh-Gaussian policy
Runs in a separate environment with sbx-rl installed (see baselines/README.md).
Usage: python baselines/crossq_sbx.py --task hopper --seed 0 [--output-root outputs]
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import gymnasium as gym
import jax
import numpy as np
from sbx import CrossQ
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure

ENVS = dict(hopper="Hopper-v4", walker2d="Walker2d-v4", halfcheetah="HalfCheetah-v4",
            ant="Ant-v4", humanoid="Humanoid-v4")
MODES = ("deterministic", "stochastic")


class PairedEval(BaseCallback):
    def __init__(self, eval_env, seed, directory, eval_freq=5000, n_episodes=10):
        super().__init__()
        self.eval_env, self.seed, self.eval_freq, self.n_episodes = eval_env, seed, eval_freq, n_episodes
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.timesteps = []
        self.hist = {m: {k: [] for k in ("results", "ep_lengths", "env_seeds", "policy_seeds")} for m in MODES}

    def _on_step(self):
        if not (self.n_calls == 1 or self.n_calls % self.eval_freq == 0):
            return True
        policy = self.model.policy
        old = (policy.key, policy.noise_key)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.num_timesteps, 4404]))
        env_seeds = rng.integers(0, 2**30, self.n_episodes)
        policy_seeds = rng.integers(0, 2**30, self.n_episodes)
        try:
            for mode in MODES:
                rewards, lengths = [], []
                for es, ps in zip(env_seeds, policy_seeds):
                    self.eval_env.seed(int(es))
                    policy.key = jax.random.PRNGKey(int(ps))
                    r, l = evaluate_policy(self.model, self.eval_env, n_eval_episodes=1,
                                           deterministic=(mode == "deterministic"),
                                           return_episode_rewards=True, warn=False)
                    rewards += r; lengths += l
                h = self.hist[mode]
                for k, v in (("results", rewards), ("ep_lengths", lengths),
                             ("env_seeds", env_seeds.tolist()), ("policy_seeds", policy_seeds.tolist())):
                    h[k].append(v)
                self.logger.record(f"eval/{mode}/mean_reward", float(np.mean(rewards)))
                self.logger.record(f"eval/{mode}/mean_ep_length", float(np.mean(lengths)))
        finally:
            policy.key, policy.noise_key = old
        self._log_sigma()
        self.timesteps.append(self.num_timesteps)
        for mode in MODES:
            tmp = self.directory / f"evaluations_{mode}.tmp.npz"
            np.savez(tmp, timesteps=self.timesteps, **self.hist[mode])
            tmp.replace(self.directory / f"evaluations_{mode}.npz")
        self.logger.dump(self.num_timesteps)
        return True

    def _log_sigma(self, n_states=256, n_samples=64):
        """Policy width on replay states: pre-tanh sigma (the Gaussian scale) and the empirical std of
        sampled actions in action space (comparable to iBOLT's z + sigma policy)."""
        buf = self.model.replay_buffer
        if buf is None or buf.size() < n_states:
            return
        obs = jax.numpy.asarray(np.asarray(buf.sample(n_states).observations))
        st = self.model.policy.actor_state
        dist = st.apply_fn({"params": st.params, "batch_stats": st.batch_stats}, obs, train=False)
        sigma = np.asarray(dist.distribution.stddev())
        acts = np.asarray(dist.sample(seed=jax.random.PRNGKey(self.num_timesteps), sample_shape=(n_samples,)))
        a_std = acts.std(0)
        for k, v in (("sigma_pre_tanh_mean", sigma.mean()), ("sigma_pre_tanh_min", sigma.min()),
                     ("sigma_pre_tanh_max", sigma.max()), ("action_std_mean", a_std.mean()),
                     ("action_std_min", a_std.mean(0).min()), ("action_std_max", a_std.mean(0).max()),
                     ("abs_mean_action", np.abs(np.tanh(np.asarray(dist.distribution.mean()))).mean())):
            self.logger.record(f"policy/{k}", float(v))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=sorted(ENVS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-steps", type=int, default=1_000_000)
    p.add_argument("--critic-width", type=int, default=2048)
    p.add_argument("--output-root", default="outputs")
    p.add_argument("--utd", type=int, default=1, help="gradient steps per environment step")
    args = p.parse_args()

    utd_tag = f"-utd{args.utd}" if args.utd != 1 else ""
    name = (f"{args.task}-crossq-q{args.critic_width}{utd_tag}-s{args.seed}_"
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}")
    out = (Path(args.output_root) / name).resolve()
    out.mkdir(parents=True, exist_ok=False)
    hp = dict(learning_starts=5000, batch_size=256, buffer_size=1_000_000, gamma=0.99,
              gradient_steps=args.utd, train_freq=1, policy_delay=3, learning_rate=1e-3,
              policy_kwargs=dict(net_arch=dict(pi=[256, 256], qf=[args.critic_width] * 2)))
    import sbx, stable_baselines3
    (out / "config.json").write_text(json.dumps(dict(
        algorithm="CrossQ (sbx reference implementation)", env=ENVS[args.task], seed=args.seed,
        total_steps=args.total_steps, hyperparameters=hp, eval=dict(interval=5000, episodes=10, modes=MODES),
        versions=dict(sbx=sbx.__version__, sb3=stable_baselines3.__version__, jax=jax.__version__,
                      gymnasium=gym.__version__), devices=[d.device_kind for d in jax.devices()]),
        indent=2, default=str))

    env = gym.make(ENVS[args.task])
    eval_env = make_vec_env(ENVS[args.task], n_envs=1, seed=args.seed)
    model = CrossQ("MlpPolicy", env, seed=args.seed, verbose=0, **hp)
    model.set_logger(configure(str(out / "logs"), ["csv"]))
    model.learn(total_timesteps=args.total_steps, log_interval=10,
                callback=PairedEval(eval_env, args.seed, out / "eval"))
    (out / "completed.json").write_text(json.dumps(dict(timesteps=model.num_timesteps, completed=True)))
    print(f"Completed: {out}", flush=True)


if __name__ == "__main__":
    main()
