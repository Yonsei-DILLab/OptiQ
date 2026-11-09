"""Train AEAP with its official agent (github.com/kk0mo/AEAP, AEAP.py unchanged).

The repo's main.py imports modules that are not in the repository (utils, TD3, multi_actors,
gamid_vae2, gymnasium_robotics), so this script reproduces its training loop for AEAP only:
same defaults (start_timesteps 1e4, expl_noise 1.0 * max_action, batch 512, discount 0.99,
tau 0.005, policy_noise 0.2, noise_clip 0.5, policy_freq 2), the same done/time-limit handling and
the standard TD3 replay buffer (Fujimoto et al.) that `utils.ReplayBuffer` refers to.
Evaluation follows the iBOLT protocol: every 5,000 steps, 10 episodes, the repo's eval rule
`select_action(state, is_eval=True)`, eval env seeded with seed+100 as in main.py.

Usage: ~/torch-venv/bin/python baselines/aeap_run.py --env Humanoid-v4 --seed 0 [--aeap-src ~/baselines_src/AEAP]
Output: outputs/{task}-aeap-s{seed}_{UTC}_{hex}/  eval/evaluations_deterministic.npz, logs/progress.csv,
        checkpoints/{best,final,step_XXXXXXX}_*, config.json, completed.json
"""
import argparse, csv, json, os, sys, time, uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import gymnasium as gym

p = argparse.ArgumentParser()
p.add_argument("--env", default="Humanoid-v4")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--max-timesteps", type=int, default=1_000_000)
p.add_argument("--start-timesteps", type=int, default=10_000)
p.add_argument("--expl-noise", type=float, default=1.0)
p.add_argument("--batch-size", type=int, default=512)
p.add_argument("--eval-freq", type=int, default=5000)
p.add_argument("--eval-episodes", type=int, default=10)
p.add_argument("--checkpoint-every", type=int, default=250_000)
p.add_argument("--aeap-src", default=str(Path.home() / "baselines_src" / "AEAP"))
p.add_argument("--output-root", default="outputs")
args = p.parse_args()

sys.path.insert(0, args.aeap_src)
import torch
import AEAP  # official agent; sets torch default tensor type to CUDA

device = AEAP.device


class ReplayBuffer:  # utils.ReplayBuffer of the TD3 reference code
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size, self.ptr, self.size = max_size, 0, 0
        self.state = np.zeros((max_size, state_dim)); self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim)); self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

    def add(self, state, action, next_state, reward, done):
        i = self.ptr
        self.state[i], self.action[i], self.next_state[i] = state, action, next_state
        self.reward[i], self.not_done[i] = reward, 1. - done
        self.ptr = (self.ptr + 1) % self.max_size; self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        f = lambda x: torch.FloatTensor(x[ind]).to(device)
        return f(self.state), f(self.action), f(self.next_state), f(self.reward), f(self.not_done)


def eval_policy(policy):  # main.py eval_policy, AEAP branch
    eval_env = gym.make(args.env, render_mode=None)
    eval_env.reset(seed=args.seed + 100)
    rewards, lengths = [], []
    for _ in range(args.eval_episodes):
        state, _ = eval_env.reset()
        ep_r, ep_l, done, trunc = 0., 0, False, False
        while not (done or trunc):
            with torch.no_grad():
                action = policy.select_action(np.array(state), is_eval=True)
            state, r, done, trunc, _ = eval_env.step(action)
            ep_r += r; ep_l += 1
        rewards.append(ep_r); lengths.append(ep_l)
    eval_env.close()
    return rewards, lengths


task = args.env.split("-")[0].lower()
run = f"{task}-aeap-s{args.seed}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
out = Path(args.output_root) / run
for d in ("eval", "logs", "checkpoints"):
    (out / d).mkdir(parents=True, exist_ok=True)
(out / "config.json").write_text(json.dumps(dict(vars(args), algo="AEAP", source="github.com/kk0mo/AEAP"), indent=2))

env = gym.make(args.env, render_mode=None)
env.reset(seed=args.seed); env.action_space.seed(args.seed)
torch.manual_seed(args.seed); np.random.seed(args.seed)
sd, ad = env.observation_space.shape[0], env.action_space.shape[0]
max_action = float(env.action_space.high[0])
policy = AEAP.AEAP(state_dim=sd, action_dim=ad, max_action=max_action, discount=0.99, tau=0.005)
buf = ReplayBuffer(sd, ad)

ev_t, ev_r, ev_l, best = [], [], [], -np.inf
logf = open(out / "logs" / "progress.csv", "w", newline=""); log = csv.writer(logf)
log.writerow(["env_steps", "eval_mean", "active_actors", "time_elapsed"])
t0 = time.time()
state, _ = env.reset(); ep_t = 0
for t in range(args.max_timesteps):
    ep_t += 1
    if t < args.start_timesteps:
        action = env.action_space.sample()
    else:
        action = (policy.select_action(np.array(state))
                  + np.random.normal(0, max_action * args.expl_noise, size=ad)).clip(-max_action, max_action)
    next_state, reward, done, trunc, _ = env.step(action)
    done_bool = float(done) if ep_t < env._max_episode_steps else 0
    buf.add(state, action, next_state, reward, done_bool)
    state = next_state
    if t >= args.start_timesteps:
        policy.train(buf, args.batch_size)
    if done or trunc:
        state, _ = env.reset(); ep_t = 0
    if (t + 1) % args.eval_freq == 0:
        r, l = eval_policy(policy)
        ev_t.append(t + 1); ev_r.append(r); ev_l.append(l)
        np.savez(out / "eval" / "evaluations_deterministic.npz", timesteps=np.array(ev_t),
                 results=np.array(ev_r), ep_lengths=np.array(ev_l))
        n_act = int(policy.active_actors.sum().item())
        log.writerow([t + 1, float(np.mean(r)), n_act, round(time.time() - t0, 1)]); logf.flush()
        print(f"step {t+1} eval {np.mean(r):.1f} actors {n_act} {time.time()-t0:.0f}s", flush=True)
        if np.mean(r) > best:  # main.py saves the best-evaluated model
            best = float(np.mean(r)); policy.save(str(out / "checkpoints" / "best"))
    if (t + 1) % args.checkpoint_every == 0:
        policy.save(str(out / "checkpoints" / f"step_{t+1:07d}"))
policy.save(str(out / "checkpoints" / "final"))
(out / "completed.json").write_text(json.dumps(dict(timesteps=args.max_timesteps, best_eval=best,
                                                     active_actors=int(policy.active_actors.sum().item()),
                                                     wall_s=round(time.time() - t0)), indent=2))
print("Completed:", out)
