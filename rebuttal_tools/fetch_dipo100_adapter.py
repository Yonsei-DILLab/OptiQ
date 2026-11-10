"""Bind shared vector HER to the unchanged native 100-step DiPo learner."""
import numpy as np
from rebuttal_tools.fetch_suite_vector_replay import VectorHERReplayBuffer


class DIPOReplayBuffer(VectorHERReplayBuffer):
    def draw(self, batch_size):
        result = super().draw(batch_size)
        self.last_draw = result
        return result


def match_hidden_layers(native):
    """Use two hidden layers without editing the pinned diffusion/critic code."""
    import torch
    assert native.updates == native.agent.step == 0
    for actor in (native.agent.actor, native.agent.actor_target):
        modules = list(actor.model.layer.children())
        assert len(modules)==7
        actor.model.layer = torch.nn.Sequential(*modules[:4],modules[-1])
    for critic in (native.agent.critic,native.agent.critic_target):
        for name in ('q1_model','q2_model'):
            modules = list(getattr(critic,name).children())
            assert len(modules)==7
            setattr(critic,name,torch.nn.Sequential(*modules[:4],modules[-1]))
    for optimizer_name,model_name in [('actor_optimizer','actor'),('critic_optimizer','critic')]:
        old = getattr(native.agent,optimizer_name)
        assert not old.state
        setattr(native.agent,optimizer_name,type(old)(getattr(native.agent,model_name).parameters(),**dict(old.defaults)))
    native.config.update(network='256x2 Mish; original time embedding 32',
        actor_hidden_dims=[256,256],critic_hidden_dims=[256,256],
        actor_layer_norm=False,critic_layer_norm=False)
    for actor in (native.agent.actor,native.agent.actor_target):
        linear=[m for m in actor.model.layer if isinstance(m,torch.nn.Linear)]
        assert [m.out_features for m in linear[:-1]]==[256,256]
        assert not any(isinstance(m,torch.nn.LayerNorm) for m in actor.modules())
    for critic in (native.agent.critic,native.agent.critic_target):
        for name in ('q1_model','q2_model'):
            linear=[m for m in getattr(critic,name) if isinstance(m,torch.nn.Linear)]
            assert [m.out_features for m in linear[:-1]]==[256,256]
    for name,model_name in [('actor_optimizer','actor'),('critic_optimizer','critic')]:
        optimized={id(p) for group in getattr(native.agent,name).param_groups for p in group['params']}
        assert optimized=={id(p) for p in getattr(native.agent,model_name).parameters()}


def bind_replay(native, replay, normalizer):
    import torch
    from fetch_rebuttal.dipo import upstream
    _, _, OriginalDiffusionMemory = upstream()

    class CriticReplay:
        def sample(self, batch_size):
            batch = replay.sample(batch_size, env=normalizer)
            fields = (batch.observations, batch.actions, batch.rewards,
                batch.next_observations, .99*(1.-batch.dones))
            return tuple(x.to(native.device) for x in fields)

    class ActorReplay(OriginalDiffusionMemory):
        def __init__(self):
            self.device = native.device
            self.capacity = replay.buffer_size*replay.n_envs
            self.states = replay.observations.reshape(-1, replay.obs_shape[0])
            self.best_actions = replay.actions.reshape(-1, replay.action_dim)

        def sample(self, batch_size):
            batch = replay.sample(batch_size, env=normalizer)
            times, envs, _, _ = replay.last_draw
            indices = (times % replay.buffer_size)*replay.n_envs+envs
            actions = batch.actions.to(self.device).requires_grad_(True)
            return batch.observations.to(self.device), actions, indices

        # Inherit the author's advanced-indexing replace() verbatim. Its updates
        # do not persist in the stored bank; preserve that upstream behavior.

    native.memory = native.agent.memory = CriticReplay()
    native.diffusion_memory = native.agent.diffusion_memory = ActorReplay()
    assert type(native.diffusion_memory).replace is OriginalDiffusionMemory.replace
    native.config.update(diffusion_steps=100, HER='204/256 separately in critic and actor minibatches',
        actual_action_grad_clip=float(native.agent.action_grad_norm),
        replay_replace_is_upstream=True)


class DIPOView:
    def __init__(self, config, output, replay, normalizer):
        from fetch_rebuttal.dipo import Agent
        self.native = Agent(config, output, device='cuda')
        match_hidden_layers(self.native)
        self.normalizer = normalizer
        bind_replay(self.native, replay, normalizer)
        self.config = self.native.config
        assert self.native.agent.actor.n_timesteps == 100
        assert self.native.agent.actor_target.n_timesteps == 100
        assert self.native.agent.action_gradient_steps == 20

    @property
    def updates(self):
        return self.native.updates

    def act(self, observations, mode='train'):
        if mode != 'train':
            observations = self.normalizer.normalize_obs(observations)
        return self.native.act(observations, mode)

    def update(self, step):
        self.native.update(step)
        return {}

    def evaluation_rng(self, seed):
        return self.native.evaluation_rng(seed)

    def save(self, folder, step):
        self.native.save(folder/f'policy_{step:09d}.pt')

    def close(self):
        self.native.close()
