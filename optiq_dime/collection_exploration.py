"""DACER-inspired exploration confined to environment collection.

The base actor, TD target and OT-NLL are inherited without overrides. Entropy
is the DACER GMM proxy H(A,Z), not exact mixture entropy or an RL reward.
"""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from .algorithm import OptiQDIME
from .policy import OptiQPolicy


@dataclass(frozen=True)
class ExplorationSettings:
    mode: str = "adaptive"
    noise_scale: float = 0.1
    initial_alpha: float = 1.5
    target_entropy_per_dim: float = 0.0
    alpha_lr: float = 0.03
    update_interval: int = 10000
    recent_capacity: int = 10000
    entropy_states: int = 256
    entropy_samples: int = 200
    gmm_components: int = 3
    gmm_reg_covar: float = 1e-6

    def __post_init__(self):
        if self.mode not in {"none", "fixed", "adaptive"}:
            raise ValueError("collection exploration mode must be none, fixed or adaptive")
        for name in ("noise_scale", "initial_alpha", "alpha_lr", "gmm_reg_covar"):
            if not math.isfinite(float(getattr(self, name))) or getattr(self, name) <= 0:
                raise ValueError(f"collection_exploration.{name} must be positive and finite")
        if not math.isfinite(self.target_entropy_per_dim):
            raise ValueError("target_entropy_per_dim must be finite")
        for name in ("update_interval", "recent_capacity", "entropy_states", "entropy_samples", "gmm_components"):
            value = getattr(self, name)
            if not math.isfinite(float(value)) or int(value) != value or value < 1:
                raise ValueError(f"collection_exploration.{name} must be a positive integer")
        if self.entropy_states > self.recent_capacity:
            raise ValueError("entropy_states must not exceed recent_capacity")
        if self.entropy_samples <= self.gmm_components:
            raise ValueError("entropy_samples must exceed gmm_components")


def validate_exploration(cfg):
    settings = cfg.alg.get("collection_exploration")
    if settings is None:
        return None
    settings = ExplorationSettings(**dict(settings))
    if cfg.alg.critic.backup_mode != "td" or cfg.alg.ent_coef.init != 0:
        raise ValueError("Collection exploration requires ordinary TD with zero entropy bonus")
    if cfg.alg.behavior_uniform_probability != 0 or cfg.alg.actor.td_noise_std != 0:
        raise ValueError("Do not combine this experiment with other behavior or TD noise")
    if cfg.alg.actor.type != "semi_implicit":
        raise ValueError("This exploration experiment requires the v3 conditional actor")
    return settings


class AlphaController:
    """Scalar Adam on log(alpha) * stop_gradient(H_estimate - H_target)."""
    def __init__(self, initial_alpha, learning_rate):
        self.log_alpha = math.log(initial_alpha)
        self.learning_rate = learning_rate
        self.m = self.v = 0.0
        self.updates = 0

    @property
    def alpha(self):
        return math.exp(self.log_alpha)

    def update(self, entropy, target):
        gradient = float(entropy - target)
        if not math.isfinite(gradient):
            raise FloatingPointError("Nonfinite exploration entropy")
        self.updates += 1
        self.m = 0.9 * self.m + 0.1 * gradient
        self.v = 0.999 * self.v + 0.001 * gradient**2
        m_hat = self.m / (1 - 0.9**self.updates)
        v_hat = self.v / (1 - 0.999**self.updates)
        self.log_alpha -= self.learning_rate * m_hat / (math.sqrt(v_hat) + 1e-8)
        if not math.isfinite(self.alpha):
            raise FloatingPointError("Nonfinite exploration alpha")


def gmm_entropy_proxy(actions, components=3, reg_covar=1e-6):
    """Fit separately per state in normalized, clipped action coordinates."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.mixture import GaussianMixture
    from threadpoolctl import threadpool_limits

    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 3 or not np.isfinite(actions).all():
        raise ValueError("Expected finite actions [states, samples, action_dim]")
    values, converged = [], []
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for state_actions in actions:
            mixture = GaussianMixture(
                n_components=components, covariance_type="full", random_state=42,
                reg_covar=reg_covar, n_init=1, max_iter=100, tol=1e-3,
            ).fit(state_actions)
            signs, log_determinants = np.linalg.slogdet(mixture.covariances_)
            if np.any(signs <= 0):
                raise FloatingPointError("Nonpositive GMM covariance determinant")
            weights = mixture.weights_
            component_h = 0.5 * (actions.shape[-1] * math.log(2 * math.pi * math.e) + log_determinants)
            values.append(float(-np.sum(weights * np.log(weights)) + np.dot(weights, component_h)))
            converged.append(mixture.converged_)
    return np.asarray(values), float(np.mean(converged))


class CollectionExplorationOptiQDIME(OptiQDIME):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exploration = validate_exploration(self.cfg)
        if self.exploration is None:
            raise ValueError("Missing collection_exploration settings")
        if len(self.observation_space.shape) != 1 or len(self.action_space.shape) != 1:
            raise ValueError("Collection exploration currently supports flat Box spaces")
        e = self.exploration
        self.exploration_alpha = AlphaController(e.initial_alpha, e.alpha_lr)
        streams = np.random.SeedSequence([int(self.seed or 0), 310916]).spawn(3)
        self.exploration_noise_rng = np.random.default_rng(streams[0])
        self.exploration_state_rng = np.random.default_rng(streams[1])
        self.exploration_probe_rng = np.random.default_rng(streams[2])
        self.exploration_key = jax.random.fold_in(jax.random.PRNGKey(int(self.seed or 0)), 310916)
        self.recent_observations = np.empty((e.recent_capacity, *self.observation_space.shape), dtype=np.float32)
        self.recent_pos = self.recent_count = 0
        self.exploration_action_count = 0
        self.last_exploration_update = 0
        self.exploration_metrics = {}

    @property
    def exploration_noise_std(self):
        return self.exploration.noise_scale * self.exploration_alpha.alpha

    def _remember_observations(self):
        for observation in np.asarray(self._last_obs, dtype=np.float32):
            self.recent_observations[self.recent_pos] = observation
            self.recent_pos = (self.recent_pos + 1) % self.exploration.recent_capacity
            self.recent_count = min(self.recent_count + 1, self.exploration.recent_capacity)

    def _update_exploration(self):
        e = self.exploration
        if (e.mode != "adaptive" or self._n_updates <= 0
                or self._n_updates % e.update_interval != 0
                or self.last_exploration_update == self._n_updates):
            return
        if self.recent_count < e.entropy_states:
            return
        indices = self.exploration_state_rng.choice(self.recent_count, e.entropy_states, replace=False)
        observations = np.repeat(self.recent_observations[indices], e.entropy_samples, axis=0)
        self.exploration_key, key = jax.random.split(self.exploration_key)
        base = np.asarray(OptiQPolicy.sample_action(
            self.policy.actor_state, jnp.asarray(observations), key, deterministic=False,
        )).reshape(e.entropy_states, e.entropy_samples, -1)
        before_std = self.exploration_noise_std
        perturbations = self.exploration_probe_rng.normal(size=base.shape) * before_std
        noisy_unclipped = base + perturbations
        noisy = np.clip(noisy_unclipped, -1.0, 1.0)
        entropies, converged = gmm_entropy_proxy(noisy, e.gmm_components, e.gmm_reg_covar)
        target = e.target_entropy_per_dim * base.shape[-1]
        self.exploration_alpha.update(float(entropies.mean()), target)
        self.last_exploration_update = self._n_updates
        self.exploration_metrics = {
            "entropy_proxy": float(entropies.mean()),
            "entropy_proxy_per_dim": float(entropies.mean() / base.shape[-1]),
            "entropy_proxy_p10": float(np.quantile(entropies, 0.1)),
            "target_entropy": target,
            "entropy_error": float(entropies.mean() - target),
            "gmm_converged_fraction": converged,
            "base_action_std": float(base.std(axis=1).mean()),
            "noisy_action_std": float(noisy.std(axis=1).mean()),
            "probe_clipping_fraction": float(np.mean(np.abs(noisy_unclipped) > 1.0)),
            "entropy_measurement_noise_std": before_std,
            "entropy_measurement_env_step": self.num_timesteps,
            "entropy_measurement_train_updates": self._n_updates,
            "alpha_updates": self.exploration_alpha.updates,
        }

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        action, buffer_action = super()._sample_action(learning_starts, action_noise, n_envs)
        if self.exploration.mode == "none" or self.num_timesteps < learning_starts:
            return action, buffer_action
        self._remember_observations()
        self._update_exploration()
        noise = self.exploration_noise_rng.normal(size=buffer_action.shape) * self.exploration_noise_std
        unclipped = buffer_action + noise
        executed = np.clip(unclipped, -1.0, 1.0).astype(buffer_action.dtype)
        self.exploration_action_count += n_envs
        metrics = {
            **self.exploration_metrics,
            "alpha": self.exploration_alpha.alpha,
            "noise_std": self.exploration_noise_std,
            "action_count": self.exploration_action_count,
        }
        for name, value in metrics.items():
            self.logger.record(f"exploration/{name}", value)
        self.logger.record_mean("exploration/clipping_fraction", float(np.mean(np.abs(unclipped) > 1)))
        self.logger.record_mean("exploration/executed_perturbation_rms", float(np.sqrt(np.mean((executed - buffer_action)**2))))
        # Replay stores precisely the action used for the environment transition.
        return self.policy.unscale_action(executed), executed

    def _save_model(self):
        super()._save_model()
        if not hasattr(self, "exploration_alpha"):
            return
        data = {
            "settings": asdict(self.exploration),
            "alpha_controller": vars(self.exploration_alpha),
            "noise_std": self.exploration_noise_std,
            "last_update": self.last_exploration_update,
            "action_count": self.exploration_action_count,
            "metrics": self.exploration_metrics,
            "noise_rng": self.exploration_noise_rng.bit_generator.state,
            "state_rng": self.exploration_state_rng.bit_generator.state,
            "probe_rng": self.exploration_probe_rng.bit_generator.state,
            "jax_probe_key": np.asarray(self.exploration_key).tolist(),
            "exact_training_resume": False,
        }
        path = Path(self.model_save_path) / f"exploration_state_{self.num_timesteps}.json"
        path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
