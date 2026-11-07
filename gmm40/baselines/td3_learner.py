"""Unmodified Stable-Baselines3 TD3 on the terminal GMM40 bandit."""
from pathlib import Path
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise

from gmm40.target import SCALE
from .env import GMM40Bandit


class TD3BanditLearner:
    def __init__(self, target, seed=0, batch=256, learning_rate=3e-4,
                 warmup=1000, buffer_size=1000000, device='cpu'):
        self.env = GMM40Bandit(target)
        self.warmup = warmup
        self.model = TD3(
            'MlpPolicy',self.env,learning_rate=learning_rate,buffer_size=buffer_size,
            learning_starts=warmup,batch_size=batch,gamma=0.,tau=.005,
            train_freq=(1,'step'),gradient_steps=1,policy_delay=2,
            target_policy_noise=.2,target_noise_clip=.5,
            action_noise=NormalActionNoise(np.zeros(2),.1*np.ones(2)),
            seed=seed,device=device,verbose=0)
        self.updates = 0
        self._evaluation_only = False
        self.settings = dict(policy_delay=2,target_policy_noise=.2,target_noise_clip=.5,
                             exploration_noise_std=.1,tau=.005,net_arch=[400,300],
                             evaluation_noise_std=0.,entropy_coefficient=0.,
                             checkpoint_supports_training_resume=False)

    def advance(self, count):
        if count < 1 or self._evaluation_only:
            raise ValueError('Advance requires a positive count and a fresh training learner')
        # Two critic updates per actor update, preserving TD3's delayed policy.
        steps = 2*count+(self.warmup if self.updates == 0 else 0)
        self.model.learn(total_timesteps=steps,reset_num_timesteps=self.updates==0,
                         progress_bar=False,log_interval=None)
        self.updates = self.model._n_updates//2
        return dict(actor_updates=self.updates,critic_updates=self.model._n_updates,
                    Q_evaluations=self.env.reward_evaluations,
                    replay_size=self.model.replay_buffer.size())

    def evaluate_samples(self, n, seed):
        # TD3's policy is deterministic. Repeat its action without evaluation noise.
        was_training = self.model.policy.training
        try:
            action,_ = self.model.predict(np.zeros((1,1),dtype=np.float32),deterministic=True)
            return SCALE*np.repeat(action,n,axis=0)
        finally:
            self.model.policy.set_training_mode(was_training)

    def save(self, path):
        path = Path(path).with_suffix('.zip')
        self.model.save(path)
        return path

    def load_for_evaluation(self, path):
        self.model = TD3.load(path,device=self.model.device)
        self.updates = self.model._n_updates//2
        self._evaluation_only = True
