"""Native online learners, with only environment, logging and evaluation adapters."""
from contextlib import contextmanager
from dataclasses import asdict
import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASELINES = ROOT / "gmm40-baseline"


class TorchEvaluation:
    @contextmanager
    def evaluation(self, mode, seed):
        old = self.eval_mode
        self.eval_mode = mode
        try: yield
        finally: self.eval_mode = old


class DIPO(TorchEvaluation):
    modes = ("native",)
    warmup = 5000
    def __init__(self, env, seed, folder):
        import torch
        sys.path.insert(0, str(BASELINES / "DIPO"))
        from agent.DiPo import DiPo
        from agent.replay_memory import ReplayMemory, DiffusionMemory
        class CorrectedMemory(DiffusionMemory):
            # Upstream np.copyto(array[indices], values) modifies an advanced-index copy.
            def replace(self, indices, values): self.best_actions[indices] = values
        od, ad = env.observation_space.shape[0], env.action_space.shape[0]
        self.memory = ReplayMemory(od, ad, 1000000, "cuda")
        self.dmemory = CorrectedMemory(od, ad, 1000000, "cuda")
        self.config = dict(policy_type="Diffusion", noise_ratio=1., beta_schedule="cosine",
            n_timesteps=100, diffusion_lr=3e-4, critic_lr=3e-4, action_gradient_steps=20,
            ratio=.1, ac_grad_norm=2., tau=.005, update_actor_target_every=1, action_lr=.03)
        self.agent = DiPo(SimpleNamespace(**self.config), od, env.action_space,
                          self.memory, self.dmemory, torch.device("cuda"))
        self.updates = 0; self.eval_mode = None
    def act(self, obs):
        import torch
        with torch.no_grad():
            return self.agent.actor(torch.as_tensor(obs, device="cuda"), eval=self.eval_mode is not None).cpu().numpy()
    def store(self, s, a, r, ns, terminal): self.agent.append_memory(s, a, r, ns, .99 * (1-terminal))
    def update(self):
        self.agent.train(1, 256); self.updates += 1
        return {"updates": self.updates}
    def save(self, path):
        import torch
        torch.save({k: getattr(self.agent, k).state_dict() for k in
            ("actor", "actor_target", "critic", "critic_target", "actor_optimizer", "critic_optimizer")}, path)


class MEOW(TorchEvaluation):
    modes = ("native",)
    warmup = 5000
    def __init__(self, env, seed, folder):
        import torch
        from stable_baselines3.common.buffers import ReplayBuffer
        sys.path.insert(0, str(BASELINES / "meow/cleanrl/cleanrl"))
        from meow_continuous_action import FlowPolicy
        od, ad = env.observation_space.shape[0], env.action_space.shape[0]
        self.config = dict(alpha=.2, sigma_max=-.3, sigma_min=-5., q_lr=1e-3,
                           grad_clip=30., gamma=.99, tau=.005, batch_size=256,
                           flow_hidden_dims=[64, 64], scale_hidden_dims=[256, 256], flow_layers=2,
                           deterministic_action=True, autotune=False)
        self.policy = FlowPolicy(.2, -.3, -5., ad, od, "cuda").to("cuda")
        self.target = copy.deepcopy(self.policy)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=.001)
        self.buffer = ReplayBuffer(1000000, env.observation_space, env.action_space, "cuda", handle_timeout_termination=False)
        self.updates = 0; self.eval_mode = None
    def act(self, obs):
        import torch
        self.policy.eval()
        with torch.no_grad():
            a, _ = self.policy.sample(len(obs), obs, deterministic=self.eval_mode is not None)
            return a.cpu().numpy()
    def store(self, s, a, r, ns, terminal):
        self.buffer.add(s[None], ns[None], a[None], np.array([r]), np.array([terminal]), [{}])
    def update(self):
        import torch
        from torch.nn import functional as F
        d = self.buffer.sample(256)
        with torch.no_grad():
            self.target.eval()
            v = self.target.get_v(torch.cat([d.next_observations, d.next_observations]))
            v = torch.minimum(v[:256], v[256:])
            target = d.rewards + (1-d.dones) * .99 * v
        self.policy.train()
        q, _ = self.policy.get_qv(torch.cat([d.observations]*2), torch.cat([d.actions]*2))
        loss = F.mse_loss(q, torch.cat([target]*2))
        if not torch.isfinite(loss): raise FloatingPointError("MEOW Q loss")
        self.optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 30.)
        self.optimizer.step()
        with torch.no_grad():
            for p, t in zip(self.policy.parameters(), self.target.parameters()): t.lerp_(p, .005)
        self.updates += 1
        return {"q_loss": float(loss.detach()), "updates": self.updates}
    def save(self, path):
        import torch
        torch.save(dict(policy=self.policy.state_dict(), target=self.target.state_dict(),
                        optimizer=self.optimizer.state_dict(), updates=self.updates), path)


class MFPO:
    modes = ("native",)
    warmup = 10000
    def __init__(self, env, seed, folder):
        import gym
        import jax
        sys.path.insert(0, str(BASELINES / "MFPO"))
        from jaxrl5.agents.mean_flow_learner import MeanFlowLearner
        from jaxrl5.data.replay_buffer import ReplayBuffer
        from configs.mfpo_config import get_config
        self.config = get_config().to_dict(); self.config.pop("model_cls")
        spaces = [gym.spaces.Box(s.low, s.high, dtype=np.float32) for s in (env.observation_space, env.action_space)]
        for s in spaces: s.seed(seed)
        self.agent = MeanFlowLearner.create(seed, *spaces, **self.config)
        self.buffer = ReplayBuffer(*spaces, 1000000); self.buffer.seed(seed)
        self.updates = 0; self.eval_agent = None
        @jax.jit
        def batched_eval(agent, obs):
            keys = jax.random.split(agent.rng, len(obs)+1)
            actions = jax.vmap(lambda o,k: agent.replace(rng=k).eval_actions(o)[0])(obs, keys[1:])
            return actions, agent.replace(rng=keys[0])
        self.batched_eval = batched_eval
    def act(self, obs):
        if self.eval_agent is not None:
            a, self.eval_agent = self.batched_eval(self.eval_agent, obs)
        else:
            assert len(obs) == 1
            a, self.agent = self.agent.sample_actions(obs[0]); a = a[None]
        return np.asarray(a)
    @contextmanager
    def evaluation(self, mode, seed):
        import jax
        self.eval_agent = self.agent.replace(rng=jax.random.PRNGKey(seed))
        try: yield
        finally: self.eval_agent = None
    def store(self, s, a, r, ns, terminal):
        self.buffer.insert(dict(observations=s, actions=a, rewards=np.float32(r),
            masks=np.float32(1-terminal), dones=bool(terminal), next_observations=ns))
    def update(self):
        self.agent, info = self.agent.update(self.buffer.sample(256), utd_ratio=1)
        self.updates += 1
        return {k: float(v) for k,v in info.items()}
    def save(self, path):
        import flax.serialization
        path.write_bytes(flax.serialization.to_bytes(self.agent))


class SQL:
    modes = ("native",)
    warmup = 5000
    def __init__(self, env, seed, folder):
        from gmm40.sql import SQLOnline
        from gmm40.sql_jax import SQLConfig
        self.inner = SQLOnline(env, seed)
        self.config = asdict(SQLConfig())
    def act(self, obs): return self.inner.act(obs)
    def store(self, *args): return self.inner.store(*args)
    def update(self): return self.inner.update()
    @property
    def updates(self): return self.inner.updates
    @contextmanager
    def evaluation(self, mode, seed):
        import jax
        self.inner.eval_key = jax.random.PRNGKey(seed)
        try: yield
        finally: self.inner.eval_key = None
    def save(self, path): self.inner.learner.save(path)


class SB3:
    """Use the unchanged SAC/OptiQ training loops and an independent evaluator."""
    def __init__(self, method, env, seed, folder, smoke=False):
        self.method = method
        self.mode = None
        if method == "sac":
            from stable_baselines3 import SAC
            self.config = dict(policy="MlpPolicy", learning_rate=3e-4, buffer_size=1000000,
                learning_starts=5000, batch_size=256, tau=.005, gamma=.99, train_freq=1,
                gradient_steps=1, ent_coef="auto", policy_kwargs=dict(net_arch=[256,256]), device="cuda")
            self.model = SAC(env=env, seed=seed, **self.config)
            self.modes = ("native",)
        else:
            spec = importlib.util.spec_from_file_location("antmaze_trg", ROOT / "analysis_tools/experiments/20260921_gmm_trg_sweep/train.py")
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            from omegaconf import OmegaConf
            cfg = module.compose_config(["benchmark=ant", f"seed={seed}",
                "alg.actor.mean_output_init_scale=1.0", "dacer.enabled=true", "dacer.noise_scale=0.1",
                f"output_root={folder}"])
            cfg.env_name = "AntMaze_UMaze-v5"; cfg.task = "antmaze"
            if smoke:
                cfg.alg.learning_starts = 256
                cfg.alg.actor.learning_starts = 256
            self.config = OmegaConf.to_container(cfg, resolve=True)
            self.model = module.runner.OptiQDIME("MlpPolicy", env=env, cfg=cfg,
                model_save_path=str(folder / "checkpoints"), save_every_n_steps=100000)
            self.modes = ("stochastic_z", "zero_z")
        if smoke: self.model.learning_starts = 256
        self.warmup = int(self.model.learning_starts)
    def act(self, obs):
        if self.method == "optiq":
            # Legacy public predict() keeps only batch element zero. Use the
            # native batched sampler for paired evaluation of all episodes.
            p = self.model.policy
            p.reset_noise()
            return np.asarray(p.sample_action(p.actor_state, obs, p.noise_key,
                deterministic=self.mode == "zero_z", sample_conditional_noise=False))
        a, _ = self.model.predict(obs, deterministic=self.mode != "stochastic_z")
        return a
    @property
    def updates(self): return self.model._n_updates
    @contextmanager
    def evaluation(self, mode, seed):
        self.mode = mode
        if self.method == "optiq":
            import jax
            p = self.model.policy
            saved = p.key, p.noise_key, getattr(p, "evaluation_mu_only", False)
            p.key = jax.random.PRNGKey(seed); p.evaluation_mu_only = True
        try: yield
        finally:
            if self.method == "optiq": p.key, p.noise_key, p.evaluation_mu_only = saved
            self.mode = None
    def save(self, path):
        if self.method == "sac": self.model.save(str(path))
        else:
            self.model._save_model()
            import flax.serialization
            path.write_bytes(flax.serialization.to_bytes(dict(actor=self.model.policy.actor_state,
                critic=self.model.policy.qf_state)))


def parameter_audit(agent):
    """Verify actual policy and learned-Q parameters, not just update counters."""
    import hashlib
    import jax
    if isinstance(agent, SB3):
        if agent.method == "optiq":
            groups = dict(actor=agent.model.policy.actor_state.params, critic=agent.model.policy.qf_state.params)
        else:
            groups = dict(actor=list(agent.model.actor.parameters()), critic=list(agent.model.critic.parameters()))
    elif isinstance(agent, DIPO):
        groups = dict(actor=list(agent.agent.actor.parameters()), critic=list(agent.agent.critic.parameters()))
    elif isinstance(agent, MEOW):
        groups = dict(flow_q_and_policy=list(agent.policy.parameters()))
    elif isinstance(agent, MFPO):
        groups = dict(actor=agent.agent.actor.params, critic=agent.agent.critic_1.params)
    else:
        groups = dict(actor=agent.inner.learner.state.actor.params, critic=agent.inner.learner.state.critic.params)
    result = {}
    for name, params in groups.items():
        digest = hashlib.sha256(); count = 0
        for leaf in jax.tree_util.tree_leaves(params):
            array = np.asarray(leaf.detach().cpu() if hasattr(leaf, "detach") else leaf)
            if not np.isfinite(array).all(): raise FloatingPointError(f"Nonfinite {name} parameters")
            digest.update(array.tobytes()); count += array.size
        result[name] = dict(sha256=digest.hexdigest(), parameters=count)
    return result
