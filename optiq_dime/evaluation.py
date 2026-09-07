"""MuJoCo evaluation with random streams independent of training."""

from pathlib import Path

import jax
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy


class MujocoEvalCallback(BaseCallback):
    def __init__(self, eval_env, cfg, directory):
        super().__init__()
        self.eval_env = eval_env
        self.cfg = cfg
        self.directory = Path(directory)
        self.steps, self.returns, self.lengths = [], [], []

    def evaluate(self):
        if self.steps and self.steps[-1] == self.num_timesteps:
            return
        # Independent action and environment seeds, paired across sweep arms.
        key = jax.random.fold_in(jax.random.PRNGKey(self.cfg.seed + 10000), self.num_timesteps)
        action_key, env_key = jax.random.split(key)
        self.eval_env.seed(int(jax.random.randint(env_key, (), 0, 2**30)))
        training_key = self.model.policy.key
        training_noise_key = self.model.policy.noise_key
        self.model.policy.key = action_key
        try:
            returns, lengths = evaluate_policy(
                self.model, self.eval_env,
                n_eval_episodes=self.cfg.num_eval_episodes,
                deterministic=not self.cfg.stochastic_eval,
                return_episode_rewards=True,
            )
        finally:
            self.model.policy.key = training_key
            self.model.policy.noise_key = training_noise_key
        if not np.isfinite(returns).all():
            raise FloatingPointError("Nonfinite evaluation return")
        self.steps.append(self.num_timesteps)
        self.returns.append(returns)
        self.lengths.append(lengths)
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = self.directory / "evaluations.tmp.npz"
        np.savez(temporary, timesteps=self.steps, results=self.returns, ep_lengths=self.lengths)
        temporary.replace(self.directory / "evaluations.npz")
        self.logger.record("eval/mean_reward", float(np.mean(returns)))
        self.logger.record("eval/std_reward", float(np.std(returns)))
        self.logger.record("eval/mean_ep_length", float(np.mean(lengths)))
        self.logger.dump(self.num_timesteps)

    def _on_step(self):
        if (self.cfg.eval_at_start and self.n_calls == 1) or (
            self.cfg.eval_interval > 0 and self.num_timesteps % self.cfg.eval_interval == 0
        ):
            self.evaluate()
        elif self.num_timesteps % self.cfg.log_interval == 0 and self.logger.name_to_value:
            self.logger.dump(self.num_timesteps)
        return True

    def _on_training_end(self):
        self.evaluate()
