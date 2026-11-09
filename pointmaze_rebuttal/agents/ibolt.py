"""PointMaze iBOLT adapter; leave the shared learner and original runner intact."""
from pathlib import Path
import jax
import numpy as np
import gymnasium as gym
from omegaconf import OmegaConf
from stable_baselines3.common.logger import configure
from contextlib import contextmanager
from ..ibolt_core.algorithm import IBOLT
from .ibolt_logging import DeferredMetricIBOLT
from ..envs.replay import Replay


class Agent:
    method = 'ibolt'
    evaluation_mode = 'mu_only'

    def __init__(self, config, output, device='cuda', defer_metrics=True):
        if device == 'cuda' and jax.default_backend() != 'gpu':
            raise RuntimeError('iBOLT CUDA requested but JAX GPU is unavailable')
        if device == 'cpu' and jax.default_backend() != 'cpu':
            raise RuntimeError('CPU mode requires JAX_PLATFORMS=cpu before process startup')
        cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / 'profiles/ibolt-learner.yaml')
        del cfg.defaults
        cfg.seed, cfg.temperature = config['seed'], config['temperature']
        cfg.require_gpu = device == 'cuda'
        cfg.task, cfg.env_name = f"pointmaze-{config['maze']}", f"DrAC-PointMaze-{config['maze']}"
        cfg.dacer.enabled = cfg.wandb.activate = False
        cfg.diagnostic_interval = 0
        cfg.alg.batch_size = config['batch_size']
        cfg.total_steps = config['steps']
        cfg.alg.utd = config['updates_per_collect'] / config['num_envs']
        cfg.alg.buffer_size = config['replay_capacity']
        cfg.alg.learning_starts = cfg.alg.actor.learning_starts = config['warmup']
        cfg.eval_interval = cfg.checkpoint_interval = config['eval_every']
        cfg.num_eval_episodes = config['eval_episodes']
        cfg.output_root = str(output)
        class SpaceOnlyEnv(gym.Env):
            observation_space = gym.spaces.Box(-np.inf, np.inf, (4,), np.float32)
            action_space = gym.spaces.Box(-1., 1., (2,), np.float32)
            def reset(self, **kwargs):
                raise RuntimeError('TaskBatch owns the simulator')
            def step(self, action):
                raise RuntimeError('TaskBatch owns the simulator')
        model_class = DeferredMetricIBOLT if defer_metrics else IBOLT
        self.model = model_class('MlpPolicy', SpaceOnlyEnv(), cfg=cfg,
            model_save_path=None, save_every_n_steps=0, gradient_steps=1)
        self.model.set_logger(configure(str(output / 'learner'), ['csv']))
        self.replay = Replay(capacity=config['replay_capacity'], seed=config['seed'])
        self.model.replay_buffer = self.replay
        self.config = OmegaConf.to_container(cfg, resolve=True)
        self.config['metric_logging'] = ('deferred-until-csv-flush' if defer_metrics
                                         else 'reference-eager')
        self.config['minibatch_transport'] = ('packed-float32-device-inputs' if defer_metrics
                                              else 'reference-numpy-inputs')
        self.config['update_execution'] = ('sequential-device-loop'
            if hasattr(self.model, 'train_collection') else 'python-loop')

    @property
    def updates(self):
        return self.model._n_updates

    def observe(self, obs, actions, rewards, next_obs, done):
        self.replay.add(obs, actions, rewards, next_obs, done.astype(np.float32))

    def act(self, observations, mode='train'):
        if mode not in ('train', 'mu_only'):
            raise ValueError('iBOLT evaluation requires fresh-z mu_only')
        policy = self.model.policy
        policy.reset_noise()
        actions = policy.sample_action(policy.actor_state, jax.numpy.asarray(observations),
            policy.noise_key, deterministic=False, sample_conditional_noise=mode == 'train')
        return np.clip(np.asarray(actions), -1., 1.).astype(np.float32)

    @contextmanager
    def evaluation_rng(self, seed):
        policy = self.model.policy
        original = policy.key, policy.noise_key
        policy.key, policy.noise_key = jax.random.PRNGKey(seed), jax.random.PRNGKey(seed + 1)
        try:
            yield
        finally:
            policy.key, policy.noise_key = original

    def update(self, step):
        self.model.num_timesteps = step
        self.model.train(batch_size=int(self.model.cfg.alg.batch_size), gradient_steps=1)

    def update_many(self, step, count):
        self.model.num_timesteps = step
        if hasattr(self.model, 'train_collection'):
            self.model.train_collection(batch_size=int(self.model.cfg.alg.batch_size), count=count)
        else:
            for _ in range(count):
                self.update(step)

    def save(self, path):
        import flax.serialization
        policy = self.model.policy
        path.write_bytes(flax.serialization.to_bytes(
            dict(actor=policy.actor_state, critic=policy.qf_state)))

    def assert_finite(self):
        states = (self.model.policy.actor_state.params, self.model.policy.qf_state.params)
        if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(states)):
            raise FloatingPointError('Nonfinite iBOLT parameters')

    def save_replay(self, path):
        np.savez_compressed(path, **{k: v[:self.replay.size] for k, v in self.replay.data.items()},
                            size=self.replay.size, position=self.replay.position)

    def close(self):
        self.model.get_env().close()
        self.model.logger.close()

    def flush_logs(self, step):
        if isinstance(self.model, DeferredMetricIBOLT):
            self.model.flush_training_metrics()
        self.model.logger.record('env_steps', step)
        self.model.logger.dump(step)
