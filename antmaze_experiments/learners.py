"""Thin adapters using original learners and the upstream NovelD implementation."""
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager
import importlib.util
import sys
import copy
import numpy as np
import torch
from .settings import BUDGETS, NUM_ENVS, UPDATES, WARMUP, total_budget, DIPO_DENSE_V_MIN, WANDB_ENTITY, WANDB_PROJECT
from .numerics import DisabledIntrinsic, stable_dipo_class

ROOT = Path(__file__).resolve().parents[1]
BATCH = 4096


class Native:
    def __init__(self, method, spaces, task, folder, reward_profile='sparse',noveld=True,seed=0):
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        from ddiffpg.utils.common import preprocess_cfg
        from ddiffpg.algo.sac import AgentSAC
        from ddiffpg.replay.simple_replay import ReplayBuffer
        import gym
        with initialize_config_dir(config_dir=str(ROOT/'antmaze/ddiffpg/cfg'), version_base=None):
            cfg = compose(config_name='default', overrides=[f'algo={method}_algo',
                f'env.name=antmaze-{task}', f'seed={seed}'])
        cfg = preprocess_cfg(cfg, if_ddiffpg=False)
        assert cfg.max_step == BUDGETS[task]
        assert cfg.num_envs == NUM_ENVS and cfg.algo.update_times == UPDATES
        assert cfg.algo.batch_size == BATCH and cfg.algo.warm_up * NUM_ENVS == WARMUP
        assert cfg.env.reward_type == 'sparse'
        self.reward_profile,self.noveld_enabled = reward_profile,noveld
        cfg.env.reward_type = reward_profile
        if method == 'dipo' and reward_profile == 'dense':
            cfg.algo.v_min = DIPO_DENSE_V_MIN
        cfg.env.env_kwargs = dict(gym.spec('antmaze-' + task).kwargs,
            reward_type=cfg.env.reward_type, random_init=cfg.env.random_init)
        self.config = OmegaConf.to_container(cfg, resolve=True)
        self.config['intrinsic']['enabled'] = noveld
        self.config['intrinsic']['effective_type'] = 'noveld' if noveld else 'off'
        if method == 'dipo':self.config['projection_implementation'] = 'float64 normalized bounded C51; guarded BCE'
        proxy = SimpleNamespace(observation_space=spaces[0], action_space=spaces[1],
            max_episode_length=500 if task in ('v1', 'v2') else 700)
        self.agent = (AgentSAC if method=='sac' else stable_dipo_class())(proxy, cfg)
        if not noveld:self.agent.intrinsic = DisabledIntrinsic()
        self.replay = ReplayBuffer(1000000, (29,), 8, device='cuda')
        self.intrinsic = self.agent.intrinsic
        self.method = method
        self.updates = 0

    def act(self, obs, mode='train'):
        with torch.no_grad():
            obs = torch.as_tensor(obs, device='cuda', dtype=torch.float32)
            if self.method == 'sac':
                result = self.agent.get_actions(obs, sample=mode != 'native')
            else:
                # Official diffusion policy includes initial and reverse noise.
                # Only the extra mixed Gaussian exploration noise is omitted.
                result = self.agent.get_actions(obs, sample=mode == 'train')
            return result.cpu().numpy()

    def store(self, obs, act, reward, nxt, terminal):
        data = [torch.as_tensor(x, device='cuda', dtype=torch.float32)
                for x in (obs, act, reward, nxt, terminal)]
        self.replay.add_to_buffer(data)

    def update(self, step):
        info = self.agent.update_net(self.replay)
        self.updates += UPDATES
        info.update(getattr(self.agent,'projection_diagnostics',{}))
        if not self.noveld_enabled:info.update(noveld_mean=0.,rnd_loss=0.,rnd_grad=0.)
        return info

    def record_collection(self, obs, reward, done):
        self.agent.update_tracker(torch.as_tensor(reward,device='cuda'),
                                  torch.as_tensor(done,device='cuda'))
        self.agent.pos_history.update_mat(torch.as_tensor(obs[:,:2]))

    def state(self):
        names = ['actor', 'critic', 'critic_target', 'actor_target',
                 'actor_optimizer', 'critic_optimizer']
        result = {k: getattr(self.agent, k).state_dict() for k in names}
        if self.method == 'sac':
            result.update(log_alpha=self.agent.log_alpha.detach(),
                          alpha_optimizer=self.agent.alpha_optim.state_dict())
        return result


class ReplayView:
    def __init__(self, memory, intrinsic):
        self.memory, self.intrinsic = memory, intrinsic
        self.diagnostic = False
        self.metrics = {}

    def sample(self, batch_size, env=None):
        from stable_baselines3.common.type_aliases import ReplayBufferSamples
        diagnostic, self.diagnostic = self.diagnostic, False
        if diagnostic:
            batch_size = 256  # preserve existing DACER state sample count
        obs, act, _, reward, nxt, done = self.memory.sample_batch(batch_size)
        if not diagnostic and getattr(self.intrinsic,'enabled',True):
            bonus = self.intrinsic.compute_reward(obs, nxt)
            loss, grad = self.intrinsic.update(torch.cat([obs, nxt]))
            reward = reward + bonus
            self.metrics = dict(noveld_mean=float(bonus.mean()), rnd_loss=loss, rnd_grad=grad)
        elif not diagnostic:
            self.metrics = dict(noveld_mean=0.,rnd_loss=0.,rnd_grad=0.)
        return ReplayBufferSamples(obs.cpu(), act.cpu(), nxt.cpu(), done.cpu(), reward.cpu())


class SpaceOnlyEnv:
    """The SB3 constructor needs Gymnasium spaces; no substitute physics."""
    @staticmethod
    def create(spaces):
        import gymnasium as gym
        class Descriptor(gym.Env):
            metadata = {}
            def __init__(self):
                self.observation_space, self.action_space = [gym.spaces.Box(s.low, s.high,
                    dtype=np.float32) for s in spaces]
            def reset(self, seed=None, options=None):
                raise RuntimeError('Space descriptor cannot collect experience')
            def step(self, action):
                raise RuntimeError('Use the upstream vector environment')
        return Descriptor()


class JaxLearner:
    def __init__(self, method, spaces, task, folder, temperature=None, budget=None,
                 reward_profile='sparse',noveld=True,seed=0,dacer=True):
        import jax
        from ddiffpg.utils.intrinsic import IntrinsicM
        from ddiffpg.replay.simple_replay import ReplayBuffer
        self.method, self.updates = method, 0
        self.reward_profile,self.noveld_enabled = reward_profile,noveld
        self.budget = total_budget(task) if budget is None else budget
        self.replay = ReplayBuffer(1000000, (29,), 8, device='cuda')
        self.intrinsic = (IntrinsicM((29,), env_name='antmaze-'+task,
                                   normalize=False, pos_enc=True, L=10)
                          if noveld else DisabledIntrinsic())
        self.view = ReplayView(self.replay, self.intrinsic)
        if method == 'optiq':
            spec = importlib.util.spec_from_file_location('antmaze_trg',
                ROOT/'analysis_tools/experiments/20260921_gmm_trg_sweep/train.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            from omegaconf import OmegaConf
            # AntMaze-only defaults approved on 2026-09-23. Other benchmarks
            # retain their own model and optimizer configuration.
            upstream_lr = OmegaConf.load(ROOT/'antmaze/ddiffpg/cfg/algo/actor_critic.yaml')
            overrides = ['benchmark=ant', f'seed={seed}',
                'alg.actor.mean_output_init_scale=1.0', f'dacer.enabled={str(dacer).lower()}',
                'dacer.noise_scale=0.1', f'output_root={folder}']
            if temperature is not None:
                overrides.append(f'alg.actor.temperature={temperature}')
            cfg = module.compose_config(overrides)
            # The shared MuJoCo loader validates its 256x2 base profile. Apply
            # AntMaze's approved overrides after that validation, before model
            # construction; verify_profile checks the actual modules/optimizers.
            cfg.alg.actor.hidden_dims = [256, 256, 256]
            cfg.alg.critic.hs = [256, 256, 256]
            cfg.alg.optimizer.lr_actor = float(upstream_lr.actor_lr)
            cfg.alg.optimizer.lr_critic = float(upstream_lr.critic_lr)
            cfg.env_name = 'DDiffPG-' + task + '-upstream-' + reward_profile
            cfg.task = 'antmaze'
            cfg.wandb.entity = WANDB_ENTITY
            cfg.wandb.project = WANDB_PROJECT
            cfg.alg.batch_size = BATCH
            cfg.alg.learning_starts = WARMUP
            cfg.alg.actor.learning_starts = WARMUP
            # Constructor does not collect data. Common scheduler controls updates.
            self.model = module.runner.OptiQDIME('MlpPolicy',
                env=SpaceOnlyEnv.create(spaces), cfg=cfg,
                model_save_path=None, save_every_n_steps=self.budget)
            from stable_baselines3.common.logger import configure
            self.model.set_logger(configure(str(folder/'learner'), ['csv']))
            self.model.replay_buffer = self.view
            self.model._total_timesteps = self.budget
            self.config = OmegaConf.to_container(cfg, resolve=True)
            from .optiq_profile import verify_profile
            verify_profile(self, folder)
        else:
            from .dependencies import load_mfpo_config
            self.config = load_mfpo_config(ROOT).to_dict()
            sys.path.insert(0, str(ROOT/'gmm40-baseline/MFPO'))
            from jaxrl5.agents.mean_flow_learner import MeanFlowLearner
            self.config.pop('model_cls')
            for space in spaces:
                space.seed(seed)
            self.agent = MeanFlowLearner.create(seed, *spaces, **self.config)
            @jax.jit
            def sample(agent, obs):
                keys = jax.random.split(agent.rng, len(obs)+1)
                actions = jax.vmap(lambda o,k: agent.replace(rng=k).sample_actions(o)[0])(obs, keys[1:])
                return actions, agent.replace(rng=keys[0])
            @jax.jit
            def select(agent, obs):
                keys = jax.random.split(agent.rng, len(obs)+1)
                actions = jax.vmap(lambda o,k: agent.replace(rng=k).eval_actions_select(o, agent.eval_candidate_num)[0])(obs, keys[1:])
                return actions, agent.replace(rng=keys[0])
            self.sample, self.select = sample, select

    def act(self, obs, mode='train'):
        if self.method == 'optiq':
            p = self.model.policy
            p.reset_noise()
            action = np.asarray(p.sample_action(p.actor_state, obs, p.noise_key,
                deterministic=mode == 'zero_z', sample_conditional_noise=mode in ('train','policy')))
            if mode == 'train':
                action = action + self.model.regulator_noise_std * self.model.regulator_rng.normal(size=action.shape)
            return np.clip(action, -1, 1)
        action, self.agent = (self.select if mode == 'native' else self.sample)(self.agent, obs)
        return np.asarray(action)

    def store(self, *values):
        self.replay.add_to_buffer([torch.as_tensor(x, device='cuda', dtype=torch.float32) for x in values])

    def update(self, step):
        info = {}
        for _ in range(UPDATES):
            if self.method == 'optiq':
                m = self.model
                m.num_timesteps = step
                m._current_progress_remaining = 1-step/self.budget
                self.view.diagnostic = m.regulator_enabled and m._n_updates >= m.regulator_next_update
                m.train(batch_size=BATCH, gradient_steps=1)
                info = {k:float(v) for k,v in m.logger.name_to_value.items()
                        if isinstance(v,(int,float,np.number))}
            else:
                d = self.view.sample(BATCH)
                data = dict(observations=d.observations.numpy(), actions=d.actions.numpy(),
                    next_observations=d.next_observations.numpy(), rewards=d.rewards.numpy().ravel(),
                    masks=1-d.dones.numpy().ravel(), dones=d.dones.numpy().ravel())
                self.agent, info = self.agent.update(data, utd_ratio=1)
            self.updates += 1
        return {**{k:float(v) for k,v in info.items()}, **self.view.metrics}

    def state(self):
        import flax.serialization
        if self.method == 'mfpo':
            return dict(agent=flax.serialization.to_bytes(self.agent))
        m, p = self.model, self.model.policy
        return dict(policy=flax.serialization.to_bytes(dict(actor=p.actor_state,
            critic=p.qf_state, target_actor=p.target_actor_state)),
            entropy=flax.serialization.to_bytes(m.ent_coef_state), key=np.asarray(m.key),
            policy_key=np.asarray(p.key), noise_key=np.asarray(p.noise_key),
            regulator=flax.serialization.to_bytes(dict(log_alpha=m.regulator_log_alpha,
                optimizer=m.regulator_state, key=m.regulator_key)),
            regulator_rng=copy.deepcopy(m.regulator_rng.bit_generator.state),
            regulator_next_update=m.regulator_next_update, regulator_count=m.regulator_count,
            regulator_entropy=m.regulator_entropy)


@contextmanager
def evaluation_rng(learner, seed):
    import random
    saved = random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all()
    private = None
    if isinstance(learner, JaxLearner):
        import jax
        if learner.method == 'mfpo':
            private = learner.agent
            learner.agent = learner.agent.replace(rng=jax.random.PRNGKey(seed))
        else:
            p = learner.model.policy
            private = p.key, p.noise_key
            p.key = jax.random.PRNGKey(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(saved[0]); np.random.set_state(saved[1])
        torch.set_rng_state(saved[2]); torch.cuda.set_rng_state_all(saved[3])
        if isinstance(learner,JaxLearner):
            if learner.method == 'mfpo': learner.agent = private
            else: learner.model.policy.key, learner.model.policy.noise_key = private


def audit(learner):
    import hashlib
    if isinstance(learner, Native):
        params = dict(actor=learner.agent.actor.state_dict(), critic=learner.agent.critic.state_dict())
    else:
        if learner.method == 'mfpo':
            params = dict(actor=learner.agent.actor.params, critic=learner.agent.critic_1.params)
        else:
            params = dict(actor=learner.model.policy.actor_state.params, critic=learner.model.policy.qf_state.params)
    if learner.noveld_enabled:
        params.update(rnd_predictor=learner.intrinsic.rnd_model.predictor.state_dict(),
                      rnd_target=learner.intrinsic.rnd_model.target.state_dict())
    def leaves(x):
        if hasattr(x,'items'):
            for k in sorted(x): yield from leaves(x[k])
        elif isinstance(x,(list,tuple)):
            for v in x: yield from leaves(v)
        else: yield x
    result = {}
    for name, group in params.items():
        h = hashlib.sha256(); n = 0
        for leaf in leaves(group):
            arr = leaf.detach().cpu().numpy() if isinstance(leaf,torch.Tensor) else np.asarray(leaf)
            assert np.isfinite(arr).all(), name
            h.update(arr.tobytes()); n += arr.size
        result[name] = dict(sha256=h.hexdigest(), parameters=n)
    return result
