"""Persist iBOLT evaluations with isolated evaluation randomness."""

from pathlib import Path

import numpy as np

from models.actor_critic_evaluation_callback import EvalCallback


class MujocoEvalCallback(EvalCallback):
    """Use the original seed list, policy RNG consumption and evaluation timing."""

    def __init__(self, eval_env, cfg, directory):
        self.directory = Path(directory)
        self.mu_only_eval = bool(cfg.get("mu_only_eval", False))
        self.successful_steps = cfg.get("successful_steps")
        self._solved_counts = {}
        self._episode_steps = {}
        self._evaluation_solved_counts = []
        self.evaluations_solved_steps = []
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

    def _log_success_callback(self, locals_, globals_):
        if self.successful_steps is None:
            return super()._log_success_callback(locals_, globals_)
        info = locals_["info"]
        if "solved" not in info:
            raise KeyError("MyoSuite evaluation requires the per-step solved flag")
        index = locals_.get("i", 0)
        self._solved_counts[index] = self._solved_counts.get(index, 0) + int(bool(info["solved"]))
        self._episode_steps[index] = self._episode_steps.get(index, 0) + 1
        if locals_["done"]:
            solved = self._solved_counts.pop(index)
            length = self._episode_steps.pop(index)
            # Match MyoSuite evaluate_success: strict >, total solved steps;
            # neither a final-step flag nor a consecutive-success requirement.
            self._is_success_buffer.append(solved > self.successful_steps)
            self._per_time_is_success_buffer.append(solved / length)
            self._evaluation_solved_counts.append(solved)

    def _on_step(self):
        if self.n_calls == 1 or self.n_calls % self.eval_freq == 0:
            self._solved_counts.clear()
            self._episode_steps.clear()
            self._evaluation_solved_counts = []
        previous_count = len(self.evaluations_timesteps)
        policy = self.model.policy
        previous_mu_only = bool(getattr(policy, "evaluation_mu_only", False))
        policy.evaluation_mu_only = self.mu_only_eval
        try:
            continue_training = super()._on_step()
        finally:
            policy.evaluation_mu_only = previous_mu_only
        if len(self.evaluations_timesteps) != previous_count:
            if not np.isfinite(self.evaluations_results[-1]).all():
                raise FloatingPointError("Nonfinite evaluation return")
            # The parent collects these arrays but never writes them to disk.
            temporary = self.directory / "evaluations.tmp.npz"
            success_arrays = {}
            if self.successful_steps is not None:
                self.evaluations_solved_steps.append(list(self._evaluation_solved_counts))
                success_arrays = {
                    "successes": self.evaluations_successes,
                    "solved_steps": self.evaluations_solved_steps,
                }
            np.savez(
                temporary,
                timesteps=self.evaluations_timesteps,
                results=self.evaluations_results,
                ep_lengths=self.evaluations_length,
                **success_arrays,
            )
            temporary.replace(self.directory / "evaluations.npz")
        return continue_training
