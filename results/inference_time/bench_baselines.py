"""Inference latency of the Table 2 baselines with their OFFICIAL code (no training needed).

Each method runs in its own framework/venv on the server (official repos cloned to ~/baselines_src):
  ~/tf-venv/bin/python    bench_baselines.py sql   [--hidden 256]          # TensorFlow, softqlearning
  ~/torch-venv/bin/python bench_baselines.py meow                          # PyTorch, cleanrl MEow
  ~/torch-venv/bin/python bench_baselines.py dipo                          # PyTorch, DIPO
  ~/mfpo-venv/bin/python  bench_baselines.py mfpo                          # JAX, MFPO (jaxrl5)
  ~/gamid-venv/bin/python bench_baselines.py gamid                         # PyTorch, SB3 1.8, Gamid-PG
  ~/torch-venv/bin/python bench_baselines.py aeap                          # PyTorch, AEAP
(R1 also names Gamid-PG [Pi-Star-Lab/gamid-pg, DDPG2.predict: all N actors + target-critic Q,
pick by Q] and AEAP [kk0mo/AEAP, select_action(is_eval=True): every active actor + twin target-critic
min-Q via .item() per actor, argmax, then the chosen actor again].)
MFPO pins jax 0.4.30, whose XLA cannot compile for the RTX 5090 (sm_120: "ptxas ... cannot be
compiled to future architecture"); its venv uses jax 0.4.33, the version of iBOLT/SAC/TD3, with
the repo's other pins (flax 0.8.5, optax 0.2.3, ...).

What is timed: the method's own evaluation call, NumPy observation in -> NumPy action out
(host->device, every network evaluation / sampling step, device->host; reading the result on the
host waits for the GPU). Same protocol as bench_interleaved.py (iBOLT/SAC/TD3): Humanoid-v4,
batch 1, 5 seeds (network init + seeded observation stream), compile/graph build + warm-up excluded,
interleaved blocks over the configurations of one method, mean ± std across the 5 seed means.
Configurations: "paper" = the architecture of Appendix C.6 (2x256 actor/critic MLPs where
applicable, MEow's own flow), "official" = the repo default (only where it differs).
"""
import argparse, json, sys, time, types
from pathlib import Path
import numpy as np
import gymnasium as gym

SRC = Path.home() / "baselines_src"
ENV_ID, SEEDS, N_OBS = "Humanoid-v4", (0, 1, 2, 3, 4), 1200


def obs_stream(env, seed):
    obs, out = env.reset(seed=seed)[0], []
    env.action_space.seed(seed)
    while len(out) < N_OBS:
        out.append(np.asarray(obs, dtype=np.float64))
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            obs = env.reset()[0]
    return out


# ---------------------------------------------------------------- SQL (softqlearning, TF)
def _sql_shims():
    """softqlearning targets TF 1.9 + garage (Python 3.6). Run its code unchanged on TF 2 in v1
    graph mode: `tensorflow` -> tf.compat.v1 (+ contrib xavier = glorot uniform), and the few garage
    names it imports (bookkeeping only: Serializable, overrides, Policy base, flat_dim, config)."""
    import tensorflow as tf2
    tf1 = tf2.compat.v1
    tf1.disable_v2_behavior()
    class _TF(types.ModuleType):
        def __getattr__(self, name):
            return getattr(tf1, name)
    tf = _TF("tensorflow")
    tf.contrib = types.SimpleNamespace(layers=types.SimpleNamespace(xavier_initializer=tf1.glorot_uniform_initializer))
    sys.modules["tensorflow"] = tf
    def mod(name, **attrs):
        m = types.ModuleType(name); m.__dict__.update(attrs); sys.modules[name] = m
    class Serializable:
        def quick_init(self, locals_): pass
    class Parameterized:
        def __init__(self, *a, **k): pass
    class Policy:
        def __init__(self, env_spec): self._env_spec = env_spec
    for n in ("garage", "garage.core", "garage.misc", "garage.tf", "garage.tf.core", "garage.tf.policies", "garage.envs"):
        mod(n)
    mod("garage.config", USE_GPU=False)
    mod("garage.core.serializable", Serializable=Serializable)
    mod("garage.misc.overrides", overrides=lambda f: f)
    mod("garage.tf.policies.base", Policy=Policy)
    mod("garage.tf.core.parameterized", Parameterized=Parameterized)
    mod("garage.envs.util", flat_dim=lambda space: int(np.prod(space.shape)))
    sys.path.insert(0, str(SRC / "softqlearning"))
    return tf1


_SQL_CONTEXTS = []


def build_sql(env, seed, hidden):
    tf1 = _sql_shims() if "garage" not in sys.modules else sys.modules["tensorflow"]
    from softqlearning.policies import StochasticNNPolicy
    graph = tf1.Graph()
    with graph.as_default():
        tf1.set_random_seed(seed)
        spec = types.SimpleNamespace(observation_space=env.observation_space, action_space=env.action_space)
        policy = StochasticNNPolicy(env_spec=spec, hidden_layer_sizes=(hidden, hidden))
        sess = tf1.Session(graph=graph)
        sess.run(tf1.global_variables_initializer())
    # Official get_action uses tf.get_default_session(): enter this graph/session and keep the
    # context managers referenced (they exit if garbage collected).
    _SQL_CONTEXTS.clear()
    for c in (graph.as_default(), sess.as_default()):
        c.__enter__()
        _SQL_CONTEXTS.append(c)
    return {f"{hidden}x2": lambda o: policy.get_action(o)[0]}


# ---------------------------------------------------------------- MEow (cleanrl, PyTorch)
def build_meow(env, seed):
    import torch
    sys.path.insert(0, str(SRC / "meow" / "cleanrl" / "cleanrl"))
    import meow_continuous_action as m
    torch.manual_seed(seed)
    a = m.Args()  # official defaults: alpha 0.2, sigma_max -0.3, sigma_min -5, deterministic eval
    policy = m.FlowPolicy(alpha=a.alpha, sigma_max=a.sigma_max, sigma_min=a.sigma_min,
                          action_sizes=env.action_space.shape[0],
                          state_sizes=env.observation_space.shape[0], device="cuda")
    policy.eval()
    def act(o):  # body of m.evaluate() for one observation
        act_t, _ = policy.sample(num_samples=1, obs=o[None], deterministic=a.deterministic_action)
        return act_t.cpu().detach().numpy()[0]
    return {"official flow": act}


# ---------------------------------------------------------------- DIPO (PyTorch)
def build_dipo(env, seed):
    import torch, torch.nn as nn
    sys.path.insert(0, str(SRC / "DIPO"))
    from agent.DiPo import DiPo
    from agent.helpers import init_weights
    args = types.SimpleNamespace(policy_type="Diffusion", noise_ratio=1.0, beta_schedule="cosine",
                                 n_timesteps=100, diffusion_lr=3e-4, action_gradient_steps=20, ratio=0.1,
                                 ac_grad_norm=2.0, tau=0.005, update_actor_target_every=1,
                                 critic_lr=3e-4, action_lr=0.03)  # main.py defaults
    S, A = env.observation_space.shape[0], env.action_space.shape[0]
    out = {}
    for name in ("paper 2x256", "official 3x256"):
        torch.manual_seed(seed)
        agent = DiPo(args, S, env.action_space, None, None, "cuda")
        if name.startswith("paper"):  # Appendix C.6: two hidden layers of 256 (Mish kept)
            m = agent.actor.model
            layer = nn.Sequential(nn.Linear(S + A + 32, 256), nn.Mish(), nn.Linear(256, 256), nn.Mish(),
                                  nn.Linear(256, A))
            layer.apply(init_weights)  # official init, applied on CPU as in the repo
            m.layer = layer.to("cuda")
        out[name] = (lambda ag: (lambda o: ag.sample_action(o, eval=True)))(agent)
    return out


# ---------------------------------------------------------------- MFPO (JAX, jaxrl5)
def build_mfpo(env, seed):
    import gym as old_gym
    sys.path.insert(0, str(SRC / "MFPO"))
    from jaxrl5.agents.mean_flow_learner import MeanFlowLearner
    osp = old_gym.spaces.Box(env.observation_space.low, env.observation_space.high, dtype=np.float32)
    asp = old_gym.spaces.Box(env.action_space.low, env.action_space.high, dtype=np.float32)
    cfg = dict(T=2, temp=0.01, eval_action_selection=True, eval_candidate_num=10, actor_layer_norm=True,
               critic_layer_norm=True, atoms_num=101, v_min=-1600, v_max=1600)  # configs/mfpo_config.py
    out = {}
    for name, dims in (("paper 2x256", (256, 256)), ("official 3x256", (256, 256, 256))):
        holder = {"agent": MeanFlowLearner.create(seed, osp, asp, actor_hidden_dims=dims,
                                                   critic_hidden_dims=dims, **cfg)}
        def act(o, h=holder):  # body of jaxrl5.evaluation.evaluate()
            a, h["agent"] = h["agent"].eval_actions(o)
            return np.asarray(a)
        out[name] = act
    return out


# ---------------------------------------------------------------- Gamid-PG (rl_zoo3 fork, PyTorch, SB3 1.8)
class _SpacesEnv:
    """Minimal gym-0.21 env carrying Humanoid-v4's spaces; only used to construct the algorithm."""
    def __init__(self, osp, asp):
        import gym as old_gym
        self.observation_space, self.action_space = osp, asp
        self.metadata, self.reward_range, self.spec = {}, (-float("inf"), float("inf")), None
        self.unwrapped = self
    def reset(self):
        return self.observation_space.sample()
    def step(self, a):
        return self.observation_space.sample(), 0.0, False, {}
    def seed(self, s=None):
        return [s]
    def render(self, mode="human"):
        pass
    def close(self):
        pass


def build_gamid(env, seed):
    import gym as old_gym
    import torch
    # Import rl_zoo3.ddpg2 without running rl_zoo3/__init__.py (it pulls in the whole zoo).
    pkg = types.ModuleType("rl_zoo3"); pkg.__path__ = [str(SRC / "gamid-pg" / "rl_zoo3")]
    sys.modules.setdefault("rl_zoo3", pkg)
    from rl_zoo3.ddpg2.ddpg2 import DDPG2
    from stable_baselines3.common.vec_env import DummyVecEnv
    osp = old_gym.spaces.Box(env.observation_space.low, env.observation_space.high, dtype=np.float32)
    asp = old_gym.spaces.Box(env.action_space.low, env.action_space.high, dtype=np.float32)
    out = {}
    # Humanoid N=4 (paper Table 5), net [256, 256] (Table 3). The trained runs use the paper's TD3 variant
    # (2 critics, baselines/gamid_run.py --variant td3); the released class uses 1 critic. predict() scores
    # every actor with the target critic, so the critic count changes the inference cost.
    for name, nc in (("N=4, 2 critics (TD3, trained)", 2), ("N=4, 1 critic (released)", 1)):
        model = DDPG2("MlpPolicy", DummyVecEnv([lambda: _SpacesEnv(osp, asp)]), n_actors=4, seed=seed, train_freq=1, gradient_steps=1,
                      policy_kwargs=dict(net_arch=[256, 256], n_critics=nc), device="cuda")
        out[name] = (lambda m: (lambda o: m.predict(o.astype(np.float32), deterministic=True)[0]))(model)
    return out


# ---------------------------------------------------------------- AEAP (PyTorch)
def build_aeap(env, seed):
    import torch
    sys.path.insert(0, str(SRC / "AEAP"))
    import AEAP as aeap_mod  # sets torch's default tensor type to CUDA, as in the repo
    out = {}
    for k in (3, 2, 1):  # 3 actors at start; pruning can remove actors (our Humanoid runs ended with 3)
        torch.manual_seed(seed)
        agent = aeap_mod.AEAP(env.observation_space.shape[0], env.action_space.shape[0],
                              float(env.action_space.high[0]))
        agent.active_actors[k:] = False
        out[f"{k} active actors"] = (lambda ag: (lambda o: ag.select_action(np.array(o), is_eval=True)))(agent)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("method", choices=["sql", "meow", "dipo", "mfpo", "gamid", "aeap"])
    p.add_argument("--hidden", type=int, default=256, help="SQL hidden width (paper 256, official 128)")
    p.add_argument("--blocks", type=int, default=30)
    p.add_argument("--per", type=int, default=100)
    p.add_argument("--warm", type=int, default=300)
    a = p.parse_args()
    if a.method == "dipo":  # ~35 ms per call
        a.blocks, a.per, a.warm = min(a.blocks, 10), min(a.per, 20), min(a.warm, 20)
    if a.method in ("meow", "dipo", "aeap"):
        import torch
        torch.set_grad_enabled(False)  # official evaluate()/sample_action/eval_policy run under no_grad
    build = {"sql": lambda e, s: build_sql(e, s, a.hidden), "meow": build_meow,
             "dipo": build_dipo, "mfpo": build_mfpo, "gamid": build_gamid, "aeap": build_aeap}[a.method]
    rng = np.random.default_rng(0)
    seed_means, first_call = {}, {}
    for seed in SEEDS:
        env = gym.make(ENV_ID)
        stream = obs_stream(env, seed)
        calls = build(env, seed)
        for k, f in calls.items():
            t0 = time.perf_counter(); f(stream[0]); dt = time.perf_counter() - t0
            first_call.setdefault(k, dt * 1e3)
            for i in range(a.warm):
                f(stream[1 + i % (N_OBS - 1)])
        blocks = {k: [] for k in calls}
        for b in range(a.blocks):
            for k in rng.permutation(list(calls)):
                f, t = calls[k], np.empty(a.per)
                for i in range(a.per):
                    o = stream[(b * a.per + i) % N_OBS]
                    t0 = time.perf_counter(); f(o); t[i] = time.perf_counter() - t0
                blocks[k].append(t.mean() * 1e3)
        for k in calls:
            seed_means.setdefault(k, []).append(float(np.mean(blocks[k])))
        env.close()
        print(f"seed {seed} done: " + ", ".join(f"{k}={seed_means[k][-1]:.3f}ms" for k in calls), flush=True)
    rows = []
    print(f"\n{a.method}: {ENV_ID}, batch 1, {len(SEEDS)} seeds x {a.blocks} blocks x {a.per} calls, warm-up {a.warm}")
    for k, v in seed_means.items():
        v = np.array(v)
        rows.append(dict(method=a.method, config=k, mean_ms=v.mean(), std_ms=v.std(ddof=1), seed_means=v.tolist(),
                         first_call_ms=first_call[k]))
        print(f"  {k:16s} {v.mean():8.3f} ± {v.std(ddof=1):.3f} ms   (first call {first_call[k]:.0f} ms)")
    tag = a.method + (f"_{a.hidden}" if a.method == "sql" else "")
    Path(__file__).with_name(f"latency_baseline_{tag}.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
