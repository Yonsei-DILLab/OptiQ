"""Batched adapter around the pinned, byte-for-byte upstream DIPO learner."""
from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import gymnasium as gym
import numpy as np
import torch

VENDOR = Path(__file__).resolve().parents[1] / 'vendor/dipo'


def verify_vendor(root=VENDOR):
    manifest = json.loads((root / 'SOURCE.json').read_text())
    for name, digest in manifest['files'].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Pinned DIPO source changed: {name}')
    return manifest


def upstream():
    verify_vendor()
    if 'agent' in sys.modules:
        paths = list(getattr(sys.modules['agent'], '__path__', []))
        if str(VENDOR / 'agent') not in paths:
            raise RuntimeError('Upstream DIPO agent namespace conflicts; use a separate process')
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    learner = importlib.import_module('agent.DiPo').DiPo
    memory = importlib.import_module('agent.replay_memory')
    return learner, memory.ReplayMemory, memory.DiffusionMemory


def vector_memories():
    _, ReplayMemory, DiffusionMemory = upstream()

    class VectorReplay(ReplayMemory):
        def append(self, state, action, reward, next_state, mask):
            count = len(state)
            if not 0 < count <= self.capacity:
                raise ValueError(count)
            indices = (np.arange(count) + self.idx) % self.capacity
            self.states[indices], self.actions[indices] = state, action
            self.rewards[indices, 0] = np.asarray(reward).reshape(-1)
            self.next_states[indices] = next_state
            self.masks[indices, 0] = np.asarray(mask).reshape(-1)
            self.full |= self.idx + count >= self.capacity
            self.idx = (self.idx + count) % self.capacity

    class VectorDiffusion(DiffusionMemory):
        def append(self, state, action):
            count = len(state)
            if not 0 < count <= self.capacity:
                raise ValueError(count)
            indices = (np.arange(count) + self.idx) % self.capacity
            self.states[indices], self.best_actions[indices] = state, action
            self.full |= self.idx + count >= self.capacity
            self.idx = (self.idx + count) % self.capacity
        # Deliberately inherit replace(): upstream np.copyto on advanced indexing
        # does NOT persist improved actions across minibatches. Do not "fix" it.

    return VectorReplay, VectorDiffusion


class Agent:
    method = 'dipo'
    evaluation_mode = 'policy'

    def __init__(self, config, output, device='cuda'):
        learner, _, native_memory = upstream()
        memory, diffusion = vector_memories()
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('DIPO CUDA requested but PyTorch GPU is unavailable')
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        self.batch_size, self.count = config['batch_size'], 0
        capacity = config['replay_capacity']
        self.memory = memory(4, 2, capacity, self.device)
        self.diffusion_memory = diffusion(4, 2, capacity, self.device)
        args = SimpleNamespace(policy_type='Diffusion', noise_ratio=1.,
            beta_schedule='cosine', n_timesteps=100, diffusion_lr=3e-4,
            critic_lr=3e-4, action_gradient_steps=20, ratio=.1,
            ac_grad_norm=2., tau=.005, update_actor_target_every=1, action_lr=.03)
        self.agent = learner(args, 4, gym.spaces.Box(-1., 1., (2,), np.float32),
                             self.memory, self.diffusion_memory, self.device)
        self.config = dict(upstream=verify_vendor(), device=str(self.device),
            diffusion_steps=100, action_gradient_steps=20, action_lr=.03,
            actor_lr=3e-4, critic_lr=3e-4, adam_eps=1e-5,
            actor_critic_grad_clip=2., action_grad_clip=.2, gamma=.99, tau=.005,
            network='upstream 256x3 Mish; time embedding 32',
            replay_capacity=capacity, batch_size=self.batch_size,
            replay_replace_is_upstream=(diffusion.replace is native_memory.replace),
            evaluation='fresh initial Gaussian; reverse-process noise disabled')

    @property
    def updates(self):
        return self.count

    def observe(self, obs, actions, rewards, next_obs, done):
        self.agent.append_memory(obs, actions, rewards, next_obs,
                                 .99 * (1. - np.asarray(done, np.float32)))

    def act(self, observations, mode='train'):
        if mode not in ('train', 'policy'):
            raise ValueError('DIPO uses its native policy evaluation, never iBOLT mu_only')
        with torch.no_grad():
            obs = torch.as_tensor(np.asarray(observations, np.float32), device=self.device)
            actions = self.agent.actor(obs, eval=mode == 'policy')
        return np.clip(actions.cpu().numpy(), -1., 1.).astype(np.float32)

    def update(self, step):
        self.agent.train(1, batch_size=self.batch_size)
        self.count += 1
        if self.agent.step != self.count:
            raise RuntimeError('DIPO update accounting mismatch')

    @contextmanager
    def evaluation_rng(self, seed):
        devices = [torch.cuda.current_device()] if self.device.type == 'cuda' else []
        ratios = self.agent.actor.noise_ratio, self.agent.actor_target.noise_ratio
        try:
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(seed)
                if devices:
                    torch.cuda.manual_seed_all(seed)
                yield
        finally:
            self.agent.actor.noise_ratio, self.agent.actor_target.noise_ratio = ratios

    def assert_finite(self):
        for model in (self.agent.actor, self.agent.critic):
            if not all(torch.isfinite(p).all().item() for p in model.parameters()):
                raise FloatingPointError('Nonfinite DIPO parameters')

    def save(self, path):
        state = {name: getattr(self.agent, name).state_dict() for name in
                 ('actor', 'actor_target', 'critic', 'critic_target',
                  'actor_optimizer', 'critic_optimizer')}
        state.update(updates=self.count, config=self.config)
        torch.save(state, path)

    def save_replay(self, path):
        size = self.memory.capacity if self.memory.full else self.memory.idx
        fields = ('states', 'actions', 'rewards', 'next_states', 'masks')
        np.savez_compressed(path, **{k: getattr(self.memory, k)[:size] for k in fields},
            best_actions=self.diffusion_memory.best_actions[:size], size=size,
            position=self.memory.idx)

    def close(self):
        pass

    def flush_logs(self, step):
        # Native train() returns no loss values. The runner records counts and
        # evaluations; keep the original learner unchanged.
        pass
