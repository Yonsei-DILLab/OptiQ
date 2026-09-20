"""Paired deterministic/stochastic evaluations using the policy latent prior."""
import jax
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import sync_envs_normalization

from .evaluation import MujocoEvalCallback


def dual_mu_evaluation_spec(cfg):
    """Describe the same latent support used by training and action sampling."""
    finite = cfg.alg.actor.get("latent_prior", "normal") == "finite"
    deterministic_mode = "zero_z"
    spec = {
        deterministic_mode: "z=0 vector; a=tanh(mu(s,0)); epsilon=0",
        "stochastic_z": ("uniform choice from the same fixed training codebook per action; a=tanh(mu(s,z)); epsilon=0" if finite
                         else "z~N(0,I) per action; a=tanh(mu(s,z)); epsilon=0"),
        "episodes_per_mode": int(cfg.num_eval_episodes),
        "legacy_eval_alias": "zero_z",
        "schema_version": 2,
        "zero_z_role": "out-of-support diagnostic" if finite else "zero-latent diagnostic",
        "paired_episode_reset_seeds": True,
        "rng_isolated_from_collection": True,
    }
    if finite:
        spec.update(latent_components=int(cfg.alg.actor.latent_components),
                    latent_codebook_seed=int(cfg.alg.actor.latent_codebook_seed))
    return deterministic_mode, spec


class DualMuEvalCallback(MujocoEvalCallback):
    """Both modes use epsilon=0; each episode receives a paired reset seed.

    Evaluation RNG is independent of collection and restored even on failure.
    For finite policies stochastic-z samples the training codebook; zero-z is
    the literal zero vector (an out-of-support diagnostic). Legacy reward
    aliases zero-z for both finite and continuous policies.
    """

    MODES = ("zero_z", "stochastic_z")

    def __init__(self, eval_env, cfg, directory):
        super().__init__(eval_env, cfg, directory)
        if cfg.alg.actor.get("type") != "semi_implicit":
            raise ValueError("Dual mu evaluation requires a semi-implicit actor")
        self.deterministic_mode, self.evaluation_spec = dual_mu_evaluation_spec(cfg)
        self.MODES = (self.deterministic_mode, "stochastic_z")
        self.primary_mode = self.evaluation_spec["legacy_eval_alias"]
        self.best_rewards = {mode: -np.inf for mode in self.MODES}
        self.training_seed = int(cfg.seed)
        self.histories = {mode: {key: [] for key in
            ("results", "ep_lengths", "env_seeds", "policy_seeds", "successes", "solved_steps")}
            for mode in self.MODES}

    def _on_step(self):
        if self.eval_freq <= 0 or not (self.n_calls == 1 or self.n_calls % self.eval_freq == 0):
            return True
        if self.model.get_vec_normalize_env() is not None:
            sync_envs_normalization(self.training_env, self.eval_env)
        policy = self.model.policy
        old_key, old_noise_key = policy.key, policy.noise_key
        old_mu_only = getattr(policy, "evaluation_mu_only", False)
        old_zero_latent = getattr(policy, "evaluation_zero_latent", False)
        rng = np.random.default_rng(np.random.SeedSequence([
            self.training_seed, self.num_timesteps, 4404]))
        env_seeds = rng.integers(0, 2**30, self.n_eval_episodes)
        policy_seeds = rng.integers(0, 2**30, self.n_eval_episodes)
        current = {}
        try:
            policy.evaluation_mu_only = True
            for mode in self.MODES:
                policy.evaluation_zero_latent = mode == "zero_z"
                rewards, lengths, successes, solved_steps = [], [], [], []
                for env_seed, policy_seed in zip(env_seeds, policy_seeds):
                    self.eval_env.seed(int(env_seed))
                    policy.key = jax.random.PRNGKey(int(policy_seed))
                    self._solved_counts.clear()
                    self._episode_steps.clear()
                    self._evaluation_solved_counts = []
                    self._is_success_buffer = []
                    self._per_time_is_success_buffer = []
                    reward, length = evaluate_policy(
                        self.model, self.eval_env, n_eval_episodes=1,
                        deterministic=(mode == self.deterministic_mode), return_episode_rewards=True,
                        warn=self.warn, callback=self._log_success_callback)
                    rewards.extend(reward)
                    lengths.extend(length)
                    successes.extend(self._is_success_buffer)
                    solved_steps.extend(self._evaluation_solved_counts)
                if not np.isfinite(rewards).all():
                    raise FloatingPointError(f"Nonfinite {mode} evaluation return")
                current[mode] = dict(results=rewards, ep_lengths=lengths,
                    env_seeds=env_seeds.tolist(), policy_seeds=policy_seeds.tolist(),
                    successes=successes, solved_steps=solved_steps)
        finally:
            policy.key, policy.noise_key = old_key, old_noise_key
            policy.evaluation_mu_only = old_mu_only
            policy.evaluation_zero_latent = old_zero_latent

        self.evaluations_timesteps.append(self.num_timesteps)
        for mode, result in current.items():
            history = self.histories[mode]
            for key, value in result.items():
                history[key].append(value)
            prefix = f"eval/{mode}"
            self.best_rewards[mode] = max(self.best_rewards[mode], float(np.mean(result["results"])))
            self.logger.record(f"{prefix}/best_mean_reward", self.best_rewards[mode])
            self.logger.record(f"{prefix}/num_episodes", len(result["results"]))
            self.logger.record(f"{prefix}/std_ep_length", float(np.std(result["ep_lengths"])))
            self.logger.record(f"{prefix}/mean_reward", float(np.mean(result["results"])))
            self.logger.record(f"{prefix}/std_reward", float(np.std(result["results"])))
            self.logger.record(f"{prefix}/mean_ep_length", float(np.mean(result["ep_lengths"])))
            if result["successes"]:
                self.logger.record(f"{prefix}/success_rate", float(np.mean(result["successes"])))
            temporary = self.directory / f"evaluations_{mode}.tmp.npz"
            np.savez(temporary, timesteps=self.evaluations_timesteps, **history)
            temporary.replace(self.directory / f"evaluations_{mode}.npz")
            if self.verbose:
                print(f"Eval {mode}, step={self.num_timesteps}, "
                      f"reward={np.mean(result['results']):.2f}, "
                      f"length={np.mean(result['ep_lengths']):.2f}", flush=True)
        self.evaluations_results = self.histories[self.primary_mode]["results"]
        self.evaluations_length = self.histories[self.primary_mode]["ep_lengths"]
        self.last_mean_reward = float(np.mean(current[self.primary_mode]["results"]))
        self.best_mean_reward = max(self.best_mean_reward, self.last_mean_reward)
        self.logger.record("eval/mean_reward", self.last_mean_reward)
        self.logger.record("eval/mean_ep_length", float(np.mean(current[self.primary_mode]["ep_lengths"])))
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        self.logger.dump(self.num_timesteps)
        return True
