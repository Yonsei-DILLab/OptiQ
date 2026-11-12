"""Paired zero-latent and stochastic-latent conditional-mean evaluations."""
import jax
import numpy as np
from pathlib import Path
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import sync_envs_normalization



class DualMuEvalCallback(EvalCallback):
    """Both modes use epsilon=0; each episode receives a paired reset seed.

    Evaluation RNG is independent of collection and restored even on failure.
    MuJoCo evaluation records returns and episode lengths, not success metrics.
    """

    MODES = ("zero_z", "stochastic_z")

    def __init__(self, eval_env, cfg, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        super().__init__(eval_env,n_eval_episodes=int(cfg.num_eval_episodes),
                         eval_freq=int(cfg.eval_interval),log_path=str(self.directory))
        self.training_seed = int(cfg.seed)
        self.histories = {mode: {key: [] for key in
            ("results", "ep_lengths", "env_seeds", "policy_seeds")}
            for mode in self.MODES}

    def _on_step(self):
        if self.eval_freq <= 0 or not (self.n_calls == 1 or self.n_calls % self.eval_freq == 0):
            return True
        if self.model.get_vec_normalize_env() is not None:
            sync_envs_normalization(self.training_env, self.eval_env)
        policy = self.model.policy
        old_key, old_noise_key = policy.key, policy.noise_key
        old_mu_only = getattr(policy, "evaluation_mu_only", False)
        rng = np.random.default_rng(np.random.SeedSequence([
            self.training_seed, self.num_timesteps, 4404]))
        env_seeds = rng.integers(0, 2**30, self.n_eval_episodes)
        policy_seeds = rng.integers(0, 2**30, self.n_eval_episodes)
        current = {}
        try:
            policy.evaluation_mu_only = True
            for mode in self.MODES:
                rewards, lengths = [], []
                for env_seed, policy_seed in zip(env_seeds, policy_seeds):
                    self.eval_env.seed(int(env_seed))
                    policy.key = jax.random.PRNGKey(int(policy_seed))
                    reward, length = evaluate_policy(
                        self.model, self.eval_env, n_eval_episodes=1,
                        deterministic=(mode == "zero_z"), return_episode_rewards=True,
                        warn=self.warn)
                    rewards.extend(reward)
                    lengths.extend(length)
                if not np.isfinite(rewards).all():
                    raise FloatingPointError(f"Nonfinite {mode} evaluation return")
                current[mode] = dict(results=rewards, ep_lengths=lengths,
                    env_seeds=env_seeds.tolist(), policy_seeds=policy_seeds.tolist())
        finally:
            policy.key, policy.noise_key = old_key, old_noise_key
            policy.evaluation_mu_only = old_mu_only

        self.evaluations_timesteps.append(self.num_timesteps)
        for mode, result in current.items():
            history = self.histories[mode]
            for key, value in result.items():
                history[key].append(value)
            prefix = f"eval/{mode}"
            self.logger.record(f"{prefix}/mean_reward", float(np.mean(result["results"])))
            self.logger.record(f"{prefix}/std_reward", float(np.std(result["results"])))
            self.logger.record(f"{prefix}/mean_ep_length", float(np.mean(result["ep_lengths"])))
            temporary = self.directory / f"evaluations_{mode}.tmp.npz"
            np.savez(temporary, timesteps=self.evaluations_timesteps, **history)
            temporary.replace(self.directory / f"evaluations_{mode}.npz")
            if self.verbose:
                print(f"Eval {mode}, step={self.num_timesteps}, "
                      f"reward={np.mean(result['results']):.2f}, "
                      f"length={np.mean(result['ep_lengths']):.2f}", flush=True)
        self.evaluations_results = self.histories["zero_z"]["results"]
        self.evaluations_length = self.histories["zero_z"]["ep_lengths"]
        self.last_mean_reward = float(np.mean(current["zero_z"]["results"]))
        self.best_mean_reward = max(self.best_mean_reward, self.last_mean_reward)
        self.logger.dump(self.num_timesteps)
        return True
