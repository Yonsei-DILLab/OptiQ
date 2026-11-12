"""Native learner interfaces and explicit demonstration replay storage."""
from contextlib import contextmanager
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
from ..demo_replay import DemonstrationReplay


class NativeReplay:
    def __init__(self, replay):
        self.replay = replay

    def batch(self, size):
        values = self.replay.sample(size)
        data = {k: v.detach().cpu().numpy().astype(np.float32, copy=False)
                for k, v in zip(('observations', 'actions', 'next_observations', 'terminals', 'rewards'), values)}
        for key in ('rewards', 'terminals'):
            data[key] = data[key].reshape(-1)
        return data


class DIPOReplay(DemonstrationReplay):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_online = np.empty_like(self.actions)
        self.best_offline = self.demo['actions'].copy()

    def add(self, obs, next_obs, action, reward, done, infos):
        self.best_online[self.pos] = action
        return super().add(obs, next_obs, action, reward, done, infos)


class DIPOTransitions:
    def __init__(self, replay):
        self.replay = replay

    def sample(self, size):
        x = self.replay.sample(size)
        return tuple(v.to('cuda', dtype=torch.float32) for v in
                     (x.observations, x.actions, x.rewards, x.next_observations, .99*(1.-x.dones)))


class DIPOActions:
    def __init__(self, replay):
        self.replay = replay

    def sample(self, size):
        r = self.replay
        assert size == 256
        off = r.demo_rng.integers(0, len(r.best_offline), size//2)
        slots = r.demo_rng.integers(0, r.size(), size//2)
        envs = r.demo_rng.integers(0, r.n_envs, size//2)
        states = np.concatenate((r.demo['observations'][off], r.observations[slots, envs]))
        labels = np.concatenate((r.best_offline[off], r.best_online[slots, envs]))
        indices = np.concatenate((-1-off, slots*r.n_envs+envs))
        return (torch.as_tensor(states, device='cuda', dtype=torch.float32),
                torch.tensor(labels, device='cuda', dtype=torch.float32, requires_grad=True), indices)

    def replace(self, indices, actions):
        # Explicit indexed assignment persists the improved labels in both pools.
        # Critic transitions continue to retain the actions actually executed.
        r = self.replay
        for index, action in zip(indices, actions):
            if index < 0:
                r.best_offline[-1-index] = action
            else:
                r.best_online[index//r.n_envs, index % r.n_envs] = action


class DIPO:
    def __init__(self, seed, observation_dim, action_space, replay):
        root = Path(__file__).parent/'vendor/dipo'
        sys.path.insert(0, str(root))
        from agent.DiPo import DiPo
        torch.manual_seed(seed)
        self.config = dict(policy_type='Diffusion', noise_ratio=1., beta_schedule='cosine',
            n_timesteps=100, diffusion_lr=3e-4, critic_lr=3e-4, action_lr=.03,
            action_gradient_steps=20, ratio=.1, ac_grad_norm=2., tau=.005,
            update_actor_target_every=1)
        self.native = DiPo(SimpleNamespace(**self.config), observation_dim, action_space,
                           DIPOTransitions(replay), DIPOActions(replay), torch.device('cuda'))
        from .architecture import reduce_dipo
        reduce_dipo(self.native)
        self.config.update(actor_hidden=[256, 256], critic_hidden=[256, 256],
                           architecture_profile='matched_256x2')

    @property
    def updates(self):
        return self.native.step

    def act(self, observations, evaluation=False):
        with torch.no_grad():
            x = torch.as_tensor(observations, dtype=torch.float32, device='cuda')
            return self.native.actor(x, eval=evaluation).cpu().numpy().clip(-1, 1)

    def update(self, step):
        self.native.train(1, batch_size=256)
        return dict(updates=self.updates)

    @contextmanager
    def evaluation_rng(self, seed):
        old_noise = self.native.actor.noise_ratio
        try:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(seed)
                yield
        finally:
            self.native.actor.noise_ratio = old_noise

    def finite(self):
        return all(torch.isfinite(p).all().item() for module in (self.native.actor, self.native.critic) for p in module.parameters())

    def save(self, folder, step):
        path = folder/f'policy_{step:09d}.pt'
        torch.save(dict(actor=self.native.actor.state_dict(), critic=self.native.critic.state_dict(),
            actor_target=self.native.actor_target.state_dict(), critic_target=self.native.critic_target.state_dict(),
            actor_optimizer=self.native.actor_optimizer.state_dict(), critic_optimizer=self.native.critic_optimizer.state_dict(),
            updates=self.updates), path)
        return path


def create(method, seed, folder, budget, observation_dim, action_space, replay):
    if method == 'dipo':
        return DIPO(seed, observation_dim, action_space, replay)
    from .native import MEOW, MFPO
    cls = MEOW if method == 'meow' else MFPO
    options = dict(device='cuda') if method == 'meow' else {}
    agent = cls(seed, folder, budget, observation_dim, batch_size=256,
                action_dim=action_space.shape[0], **options)
    agent.replay = NativeReplay(replay)
    return agent


def action(agent, method, observations, evaluation=False):
    if method == 'dipo':
        return agent.act(observations, evaluation)
    if evaluation and method == 'meow':
        agent.policy.eval()
        with torch.no_grad():
            values, _ = agent.policy.sample(num_samples=len(observations), obs=observations, deterministic=True)
        return values.cpu().numpy().clip(-1, 1)
    if evaluation and method == 'mfpo':
        rows = []
        for obs in observations:
            a, agent.agent = agent.agent.eval_actions(obs)
            rows.append(np.asarray(a))
        return np.asarray(rows).clip(-1, 1)
    return agent.act(observations, mode='train')
