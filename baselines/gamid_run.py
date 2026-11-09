"""Train Gamid-PG with its official algorithm class (github.com/Pi-Star-Lab/gamid-pg, rl_zoo3/ddpg2,
unchanged) under SB3 1.7.0 / gym 0.21 (SB3 1.8.0 breaks DDPG2's extract_features call).

The repo's hyperparams/ddpg2.yml covers only some envs and uses MuJoCo-v3 (mujoco-py), so:
- env: gymnasium {Env}-v4 behind a gym-0.21 adapter (done = terminated or truncated, with
  info["TimeLimit.truncated"] so SB3 bootstraps through time limits);
- hyperparameters from the Gamid paper (TMLR 2024), per env (PAPER_HP below): Table 3 (shared TD3:
  lr, batch, train_freq, gradient_steps; learning_starts 1e4, gamma 0.99, normal action noise 0.1,
  net_arch [256, 256] for all) and Table 5 (N actors, temperature schedule).
  The epsilon-greedy actor weights of Table 5 are not implemented in the released code (DDPG2.predict
  picks by target-critic Q); we run the code as released.
- variant: DDPG2 subclasses SB3 DDPG, which hard-sets one critic, policy_delay 1 and no target smoothing,
  although the paper (Sec. 3.2, Table 3) reports Gamid built on TD3. DDPG2's update already implements
  min over critics, delayed actor updates and target smoothing, so --variant td3 (default) only sets the
  TD3 values (n_critics 2, policy_delay 2, target_policy_noise 0.2, target_noise_clip 0.5; SB3 TD3
  defaults); --variant released keeps the DDPG settings of the released class.
  One line of DDPG2.train assumes a single critic: the target-actor selection does th.cat over the critic
  outputs (dim 0), which with 2 critics stacks them into the batch and crashes. For td3 it is replaced by
  the min over critics (TD3's target), which is identical to the released line when there is 1 critic.
Evaluation follows the iBOLT protocol: every 5,000 steps, 10 episodes, deterministic predict.

Usage: ~/gamid-venv/bin/python baselines/gamid_run.py --seed 0 [--gamid-src ~/baselines_src/gamid-pg]
Output: outputs/{task}-gamid-s{seed}_{UTC}_{hex}/ eval/evaluations_deterministic.npz, checkpoints/, ...
"""
import argparse, json, sys, time, types, uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--env", default="Humanoid-v4")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--total-steps", type=int, default=1_000_000)
p.add_argument("--n-actors", type=int, default=None, help="default: paper Table 5")
p.add_argument("--variant", choices=["td3", "released"], default="td3")
p.add_argument("--eval-freq", type=int, default=5000)
p.add_argument("--checkpoint-every", type=int, default=250_000)
p.add_argument("--gamid-src", default=str(Path.home() / "baselines_src" / "gamid-pg"))
p.add_argument("--output-root", default="outputs")
args = p.parse_args()

import gym
import gymnasium
import torch
pkg = types.ModuleType("rl_zoo3"); pkg.__path__ = [str(Path(args.gamid_src) / "rl_zoo3")]
sys.modules["rl_zoo3"] = pkg  # import rl_zoo3.ddpg2 without the zoo's __init__
from rl_zoo3.ddpg2 import ddpg2 as _ddpg2_mod
from rl_zoo3.ddpg2.ddpg2 import DDPG2
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise


class GymnasiumToGym021(gym.Env):
    def __init__(self, env_id, seed):
        self.e = gymnasium.make(env_id)
        self.observation_space = gym.spaces.Box(self.e.observation_space.low, self.e.observation_space.high,
                                                dtype=np.float64)
        self.action_space = gym.spaces.Box(self.e.action_space.low, self.e.action_space.high, dtype=np.float32)
        self._seed = seed
    def seed(self, seed=None):
        self._seed = seed; return [seed]
    def reset(self):
        if self._seed is not None:
            obs, _ = self.e.reset(seed=self._seed); self._seed = None
        else:
            obs, _ = self.e.reset()
        return obs
    def step(self, a):
        obs, r, term, trunc, info = self.e.step(a)
        info = dict(info); info["TimeLimit.truncated"] = bool(trunc and not term)
        return obs, r, bool(term or trunc), info
    def close(self):
        self.e.close()


# Gamid paper Table 3 (lr, batch, train_freq, gradient_steps) and Table 5 (N, temperature init/final/fraction)
PAPER_HP = {
    "halfcheetah": dict(learning_rate=1e-3, batch_size=100, train_freq=(1, "episode"), gradient_steps=-1,
                        n_actors=5, temperature_initial=0.1, temperature_final=0.1, temperature_fraction=1.0),
    "hopper": dict(learning_rate=3e-4, batch_size=256, train_freq=1, gradient_steps=1,
                   n_actors=2, temperature_initial=0.1, temperature_final=0.01, temperature_fraction=1.0),
    "ant": dict(learning_rate=1e-3, batch_size=100, train_freq=(1, "episode"), gradient_steps=-1,
                n_actors=2, temperature_initial=0.1, temperature_final=0.1, temperature_fraction=1.0),
    "walker2d": dict(learning_rate=1e-3, batch_size=100, train_freq=(1, "episode"), gradient_steps=-1,
                     n_actors=3, temperature_initial=0.1, temperature_final=0.1, temperature_fraction=1.0),
    "humanoid": dict(learning_rate=3e-4, batch_size=256, train_freq=1, gradient_steps=1,
                     n_actors=4, temperature_initial=0.3, temperature_final=0.1, temperature_fraction=0.3),
}
task = args.env.split("-")[0].lower()
hp = dict(PAPER_HP[task], learning_starts=10_000, gamma=0.99, policy_kwargs=dict(net_arch=[256, 256]))
if args.n_actors is not None:
    hp["n_actors"] = args.n_actors
args.n_actors = hp["n_actors"]
run = f"{task}-gamid-{args.variant}-n{args.n_actors}-s{args.seed}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
out = Path(args.output_root) / run
for d in ("eval", "logs", "checkpoints"):
    (out / d).mkdir(parents=True, exist_ok=True)
td3 = dict(policy_delay=2, target_policy_noise=0.2, target_noise_clip=0.5) if args.variant == "td3" else {}
if td3:
    hp["policy_kwargs"]["n_critics"] = 2
(out / "config.json").write_text(json.dumps(dict(vars(args), algo="Gamid-PG (DDPG2)", hyperparameters=hp, td3_overrides=td3,
                                                 action_noise="normal 0.1", source="github.com/Pi-Star-Lab/gamid-pg"),
                                            indent=2, default=str))

env = Monitor(GymnasiumToGym021(args.env, args.seed))
eval_env = Monitor(GymnasiumToGym021(args.env, args.seed + 100))
ad = env.action_space.shape[0]
model = DDPG2("MlpPolicy", env, seed=args.seed, device="cuda", verbose=0,
              action_noise=NormalActionNoise(np.zeros(ad), 0.1 * np.ones(ad)), **hp)
if td3:  # selection Q with twin critics: min over critics (== released line for 1 critic)
    import inspect, re, textwrap
    src = textwrap.dedent(inspect.getsource(DDPG2.train))
    pat = re.compile(r"th\.cat\(\s*self\.critic_target\(\s*replay_data\.next_observations, next_actions\s*\)\s*\)")
    assert len(pat.findall(src)) == 1
    src = pat.sub("th.min(th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1), "
                  "dim=1, keepdim=True)[0]", src)
    ns = {}; exec(compile(src, _ddpg2_mod.__file__, "exec"), _ddpg2_mod.__dict__, ns)
    DDPG2.train = ns["train"]
for k, v in td3.items():  # DDPG.__init__ hard-sets the DDPG values; override after construction
    setattr(model, k, v)
assert len(model.critic.q_networks) == (2 if td3 else 1)
cb = CallbackList([
    EvalCallback(eval_env, eval_freq=args.eval_freq, n_eval_episodes=10, deterministic=True,
                 log_path=str(out / "eval"), best_model_save_path=str(out / "checkpoints"), verbose=0),
    CheckpointCallback(save_freq=args.checkpoint_every, save_path=str(out / "checkpoints"), name_prefix="step"),
])
from stable_baselines3.common.logger import configure
model.set_logger(configure(str(out / "logs"), ["csv"]))
t0 = time.time()
model.learn(total_timesteps=args.total_steps, callback=cb, log_interval=10)
model.save(str(out / "checkpoints" / "final"))
ev = out / "eval" / "evaluations.npz"
if ev.exists():  # same file name as the other baselines
    z = np.load(ev); np.savez(out / "eval" / "evaluations_deterministic.npz", **{k: z[k] for k in z.files})
(out / "completed.json").write_text(json.dumps(dict(timesteps=model.num_timesteps, wall_s=round(time.time() - t0)), indent=2))
print("Completed:", out)
