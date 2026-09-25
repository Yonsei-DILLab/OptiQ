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
from .progress_reward import PROFILES, value_support
from .numerics import DisabledIntrinsic, stable_dipo_class

ROOT = Path(__file__).resolve().parents[1]
BATCH = 4096


class Native:
    def __init__(self, method, spaces, task, folder, reward_profile='sparse',noveld=True):
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        from ddiffpg.utils.common import preprocess_cfg
        from ddiffpg.algo.sac import AgentSAC
        from ddiffpg.replay.simple_replay import ReplayBuffer
        import gym
        with initialize_config_dir(config_dir=str(ROOT/'antmaze/ddiffpg/cfg'), version_base=None):
            cfg = compose(config_name='default', overrides=[f'algo={method}_algo',
                f'env.name=antmaze-{task}', 'seed=0'])
        cfg = preprocess_cfg(cfg, if_ddiffpg=False)
        assert cfg.max_step == BUDGETS[task]
        assert cfg.num_envs == NUM_ENVS and cfg.algo.update_times == UPDATES
        assert cfg.algo.batch_size == BATCH and cfg.algo.warm_up * NUM_ENVS == WARMUP
        assert cfg.env.reward_type == 'sparse'
        self.reward_profile,self.noveld_enabled = reward_profile,noveld
        cfg.env.reward_type = reward_profile
        if method == 'dipo' and reward_profile == 'dense':
            cfg.algo.v_min = DIPO_DENSE_V_MIN
        if method == 'dipo' and reward_profile in PROFILES:
            cfg.algo.v_min,cfg.algo.v_max=value_support(task,reward_profile)
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
                 reward_profile='sparse',noveld=False,temperature_schedule=None,
                 dacer_target_entropy_per_dim=None,dacer_enabled=False,dynamics_profile=None,
                 dacer_interval_updates=None,discount=None,teacher_std_floor=None,latent_profile=None,
                 collection_profile=None,actor_sigma_profile=None,optiq_config_profile='basic'):
        import jax
        from ddiffpg.utils.intrinsic import IntrinsicM
        from ddiffpg.replay.simple_replay import ReplayBuffer
        self.method, self.updates = method, 0
        self.reward_profile,self.noveld_enabled = reward_profile,noveld
        self.dynamics_profile = dynamics_profile
        self.latent_profile = latent_profile
        self.actor_sigma_profile = actor_sigma_profile
        assert optiq_config_profile in ('basic', 'legacy')
        self.optiq_config_profile = optiq_config_profile
        if actor_sigma_profile is not None:
            from .actor_sigma_profile import settings as sigma_settings
            sigma_settings(actor_sigma_profile)
            assert method == 'optiq' and latent_profile is None
        from .collection_profile import get_profile
        self.updates_per_collection = get_profile(collection_profile)['updates_per_vector_step']
        if collection_profile is not None:
            assert method == 'optiq'
        assert latent_profile in (None, 'fixed64')
        if latent_profile is not None:
            assert method == 'optiq'
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
            # Compose the committed Direct GMM/TRG configuration first. The
            # basic AntMaze profile keeps its model/init/LRs and explicitly
            # selects T=1. Historical experiments retain their frozen sources.
            overrides = ['benchmark=ant', 'seed=0',
                f'dacer.enabled={str(bool(dacer_enabled)).lower()}',
                'dacer.noise_scale=0.1', f'output_root={folder}']
            if optiq_config_profile == 'legacy':
                overrides.append('alg.actor.mean_output_init_scale=1.0')
            if temperature is None and optiq_config_profile == 'basic':
                temperature = 1.0
            if temperature is not None:
                overrides.append(f'alg.actor.temperature={temperature}')
            if teacher_std_floor is not None:
                from .teacher_proposal import validate_floor
                overrides.append(f'alg.actor.teacher_std_floor={validate_floor(teacher_std_floor)}')
            if latent_profile == 'fixed64':
                overrides += ['++alg.actor.latent_prior=finite',
                              '++alg.actor.latent_components=64',
                              '++alg.actor.latent_codebook_seed=20260911',
                              # The unused standalone MuJoCo evaluator requires
                              # a normal prior. AntMaze's own evaluator runs both
                              # direct and mu-only modes with the finite prior.
                              'dual_mu_eval=false']
            if dacer_target_entropy_per_dim is not None:
                assert dacer_enabled, 'A DACER target requires an enabled regulator'
                assert np.isfinite(dacer_target_entropy_per_dim)
                overrides.append(f'dacer.target_entropy_per_dim={dacer_target_entropy_per_dim}')
            if dacer_interval_updates is not None:
                assert dacer_enabled and isinstance(dacer_interval_updates, int) and dacer_interval_updates > 0
                overrides.append(f'dacer.interval_updates={dacer_interval_updates}')
            if temperature_schedule is not None:
                schedule=temperature_schedule
                overrides.append('++alg.actor.temperature_schedule={enabled:true,'
                    f'final_temperature:{schedule["final_temperature"]},'
                    f'anneal_steps:{schedule["anneal_steps"]},decay:{schedule["decay"]}'+'}')
            cfg = module.compose_config(overrides)
            if teacher_std_floor is not None:
                cfg.experiment.teacher_extra_floor = True
            # Older AntMaze campaigns used a deeper DDiffPG-matched model.
            # Only the explicitly named legacy profile reproduces that adapter.
            if optiq_config_profile == 'legacy':
                upstream_lr = OmegaConf.load(ROOT/'antmaze/ddiffpg/cfg/algo/actor_critic.yaml')
                cfg.alg.actor.hidden_dims = [256, 256, 256]
                cfg.alg.critic.hs = [256, 256, 256]
                cfg.alg.optimizer.lr_actor = float(upstream_lr.actor_lr)
                cfg.alg.optimizer.lr_critic = float(upstream_lr.critic_lr)
            if actor_sigma_profile is not None:
                for name, value in sigma_settings(actor_sigma_profile).items():
                    cfg.alg.actor[name] = value
            if discount is not None:
                assert np.isfinite(discount) and 0 < discount < 1
                cfg.alg.gamma = float(discount)
            if dynamics_profile is not None:
                from .dynamics_profiles import get_profile
                profile = get_profile(dynamics_profile)
                assert task in ('v3', 'v4') and not noveld and not dacer_enabled
                assert temperature_schedule is None and reward_profile == profile['reward_profile']
                assert temperature == profile['temperature']
                cfg.alg.tau = profile['tau']
                cfg.alg.policy_delay = profile['policy_delay']
                cfg.alg.optimizer.lr_critic = profile['critic_lr']
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
            if discount is not None:
                assert float(self.model.gamma) == discount
            self.config = OmegaConf.to_container(cfg, resolve=True)
            from .optiq_profile import verify_profile
            verify_profile(self, folder)
            if actor_sigma_profile is not None:
                from .actor_sigma_profile import verify as verify_actor_sigma
                verify_actor_sigma(self, folder, 'initial')
            if teacher_std_floor is not None:
                from .teacher_proposal import verify_teacher_floor
                verify_teacher_floor(self, folder, teacher_std_floor)
            if latent_profile is not None:
                from .latent_profile import verify_fixed_latent
                verify_fixed_latent(self, folder, 'initial')
            if dacer_target_entropy_per_dim is not None:
                from .dacer_target import verify_target
                verify_target(self, folder, dacer_target_entropy_per_dim,
                              10000 if dacer_interval_updates is None else dacer_interval_updates)
            if not dacer_enabled:
                from .dacer_mode import verify_disabled
                verify_disabled(self, folder)
        else:
            from .dependencies import load_mfpo_config
            self.config = load_mfpo_config(ROOT).to_dict()
            sys.path.insert(0, str(ROOT/'gmm40-baseline/MFPO'))
            from jaxrl5.agents.mean_flow_learner import MeanFlowLearner
            self.config.pop('model_cls')
            for space in spaces:
                space.seed(0)
            self.agent = MeanFlowLearner.create(0, *spaces, **self.config)
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
            if mode == 'train' and self.model.regulator_enabled:
                action = action + self.model.regulator_noise_std * self.model.regulator_rng.normal(size=action.shape)
                self.model.logger.record('exploration/behavior_noise_std', self.model.regulator_noise_std)
                self.model.logger.record('exploration/behavior_clip_fraction', float(np.mean(np.abs(action) >= 1.)))
            return np.clip(action, -1, 1)
        action, self.agent = (self.select if mode == 'native' else self.sample)(self.agent, obs)
        return np.asarray(action)

    def store(self, *values):
        self.replay.add_to_buffer([torch.as_tensor(x, device='cuda', dtype=torch.float32) for x in values])

    def update(self, step):
        info = {}
        for _ in range(self.updates_per_collection):
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
        state = dict(policy=flax.serialization.to_bytes(dict(actor=p.actor_state,
            critic=p.qf_state, target_actor=p.target_actor_state)),
            entropy=flax.serialization.to_bytes(m.ent_coef_state), key=np.asarray(m.key),
            policy_key=np.asarray(p.key), noise_key=np.asarray(p.noise_key))
        if m.regulator_enabled:
            state.update(regulator=flax.serialization.to_bytes(dict(log_alpha=m.regulator_log_alpha,
                optimizer=m.regulator_state, key=m.regulator_key)),
                regulator_rng=copy.deepcopy(m.regulator_rng.bit_generator.state),
                regulator_next_update=m.regulator_next_update, regulator_count=m.regulator_count,
                regulator_entropy=m.regulator_entropy)
        else:
            state.update(regulator=None,regulator_rng=None,regulator_next_update=None,
                         regulator_count=0,regulator_entropy=None,regulator_enabled=False)
        return state


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
