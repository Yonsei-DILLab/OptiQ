"""Evaluate Gamid-PG / AEAP final checkpoints on the same episodes as bok_eval_ibolt.py
(env.reset(seed=10000 + i), each method's own official evaluation rule).

Usage: ~/gamid-venv/bin/python results/best_of_k/bok_eval_baselines.py gamid <run_dir> [--episodes 20]
       ~/torch-venv/bin/python results/best_of_k/bok_eval_baselines.py aeap <run_dir> [--episodes 20]
"""
import argparse, json, sys, time, types
from pathlib import Path
import numpy as np
import gymnasium as gym

p = argparse.ArgumentParser()
p.add_argument("method", choices=["gamid", "aeap"])
p.add_argument("run")
p.add_argument("--episodes", type=int, default=20)
p.add_argument("--src", default=str(Path.home() / "baselines_src"))
p.add_argument("--out", default="results/best_of_k/out")
a = p.parse_args()
run = Path(a.run); env = gym.make("Humanoid-v4")
import torch
if a.method == "gamid":
    pkg = types.ModuleType("rl_zoo3"); pkg.__path__ = [str(Path(a.src) / "gamid-pg" / "rl_zoo3")]
    sys.modules["rl_zoo3"] = pkg
    from rl_zoo3.ddpg2.ddpg2 import DDPG2
    import gym as old_gym
    from stable_baselines3.common.save_util import load_from_zip_file
    from stable_baselines3.common.vec_env import DummyVecEnv
    class _Spaces(old_gym.Env):  # spaces only; DDPG2 lacks the _init_setup_model kwarg DDPG2.load needs
        observation_space = old_gym.spaces.Box(env.observation_space.low, env.observation_space.high, dtype=np.float64)
        action_space = old_gym.spaces.Box(env.action_space.low, env.action_space.high, dtype=np.float32)
        def reset(self): return self.observation_space.sample()
        def step(self, a): return self.observation_space.sample(), 0.0, False, {}
    path = str(run / "checkpoints" / "final.zip")
    data, _, _ = load_from_zip_file(path, device="cpu")
    model = DDPG2("MlpPolicy", DummyVecEnv([_Spaces]), n_actors=data["n_actors"], train_freq=1, gradient_steps=1,
                  policy_kwargs=data["policy_kwargs"], device="cuda")
    model.set_parameters(path, exact_match=True, device="cuda")
    act = lambda o: model.predict(o, deterministic=True)[0]          # DDPG2Policy.predict
else:
    sys.path.insert(0, str(Path(a.src) / "AEAP")); import AEAP
    ag = AEAP.AEAP(env.observation_space.shape[0], env.action_space.shape[0], float(env.action_space.high[0]))
    ag.load(str(run / "checkpoints" / "final"))
    torch.set_grad_enabled(False)
    act = lambda o: ag.select_action(np.array(o), is_eval=True)       # AEAP eval rule
rets, t0 = [], time.time()
for i in range(a.episodes):
    obs, _ = env.reset(seed=10_000 + i); done, R = False, 0.0
    while not done:
        obs, r, term, trunc, _ = env.step(act(obs)); R += r; done = term or trunc
    rets.append(R)
name = run.name.split("_")[0]
print(f"{name} final {np.mean(rets):7.0f} ± {np.std(rets):5.0f} ({time.time()-t0:.0f}s)", flush=True)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
(out / f"{a.method}_{name}_final.json").write_text(json.dumps(dict(run=run.name, method=a.method, episodes=a.episodes, returns=rets)))
