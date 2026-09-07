"""Persist evaluations while retaining critic-dime's evaluation protocol."""

from pathlib import Path

import numpy as np

from models.actor_critic_evaluation_callback import EvalCallback


class MujocoEvalCallback(EvalCallback):
    """Use the original seed list, policy RNG consumption and evaluation timing."""

    def __init__(self, eval_env, cfg, directory):
        self.directory = Path(directory)
        super().__init__(
            eval_env,
            jax_random_key_for_seeds=int(cfg.seed),
            best_model_save_path=None,
            log_path=str(self.directory),
            eval_freq=int(cfg.eval_interval),
            n_eval_episodes=int(cfg.num_eval_episodes),
            deterministic=not bool(cfg.stochastic_eval),
            render=False,
        )

    @property
    def returns(self):
        return self.evaluations_results

    def _on_step(self):
        previous_count = len(self.evaluations_timesteps)
        continue_training = super()._on_step()
        if len(self.evaluations_timesteps) != previous_count:
            if not np.isfinite(self.evaluations_results[-1]).all():
                raise FloatingPointError("Nonfinite evaluation return")
            # The parent collects these arrays but never writes them to disk.
            temporary = self.directory / "evaluations.tmp.npz"
            np.savez(
                temporary,
                timesteps=self.evaluations_timesteps,
                results=self.evaluations_results,
                ep_lengths=self.evaluations_length,
            )
            temporary.replace(self.directory / "evaluations.npz")
        return continue_training
