"""Small SB3/JAX learner adapter; no generative sampler or diffusion model."""
from pathlib import Path
import flax.serialization
import jax
import optax
from gymnasium import spaces
from common.off_policy_algorithm import OffPolicyAlgorithmJax


class JaxLearner(OffPolicyAlgorithmJax):
    def __init__(self, policy, env, model_save_path, save_every_n_steps, cfg,
                 tensorboard_log=None, replay_buffer_class=None, **kwargs):
        self.cfg = cfg
        self.model_save_path = model_save_path
        self.save_every_n_steps = save_every_n_steps
        self.policy_delay = cfg.alg.policy_delay
        super().__init__(policy=policy, env=env,
            learning_rate=cfg.alg.optimizer.lr_actor,
            qf_learning_rate=cfg.alg.optimizer.lr_critic,
            buffer_size=cfg.alg.buffer_size, learning_starts=cfg.alg.learning_starts,
            batch_size=cfg.alg.batch_size, tau=cfg.alg.tau, gamma=cfg.alg.gamma,
            train_freq=1, gradient_steps=cfg.alg.utd,
            replay_buffer_class=replay_buffer_class, tensorboard_log=tensorboard_log,
            seed=cfg.seed, support_multi_env=True, supported_action_spaces=(spaces.Box,),
            **kwargs)
        self._setup_model()

    def _setup_model(self):
        super()._setup_model()
        self.policy = self.policy_class(self.observation_space, self.action_space, self.cfg)
        self.key = self.policy.build(self.key, self.lr_schedule, self.qf_learning_rate)
        # Preserve the training RNG stream used by the reference experiments.
        self.key, _ = jax.random.split(self.key)
        self.qf = self.policy.qf

    @staticmethod
    @jax.jit
    def soft_update(tau, state):
        return state.replace(
            target_params=optax.incremental_update(state.params, state.target_params, tau),
            target_batch_stats=optax.incremental_update(state.batch_stats, state.target_batch_stats, tau))

    def predict_critic(self, observation, action):
        return self.policy.predict_critic(observation, action)

    def _save_model(self):
        directory = Path(self.model_save_path)
        directory.mkdir(parents=True, exist_ok=True)
        for name, state in [('actor_state',self.policy.actor_state),('critic_state',self.policy.qf_state)]:
            path = directory / f'{name}_{self.num_timesteps}.msgpack'
            path.write_bytes(flax.serialization.to_bytes(state))

    def load_model(self, path, n_steps_actor, n_steps_critic):
        for attr, name, step in [('actor_state','actor_state',n_steps_actor),('qf_state','critic_state',n_steps_critic)]:
            state = getattr(self.policy, attr)
            data = (Path(path) / f'{name}_{step}.msgpack').read_bytes()
            setattr(self.policy, attr, flax.serialization.from_bytes(state, data))
