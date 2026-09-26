"""One-step iBOLT policy with twin critics."""

from collections.abc import Sequence
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState
from gymnasium import spaces

from common.policies import BaseJaxPolicy
from common.type_aliases import RLTrainState
from models.critic import VectorCritic

from .optimizers import adam_with_grad_clip
from .box_gaussian import sample_box


def kernel_init(scale: float = 1.0):
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


class SemiImplicitActor(nn.Module):
    """Bounded Gaussian center and log scale for box-truncated conditionals."""

    action_dim: int
    hidden_dims: Sequence[int]
    log_std_min: float
    log_std_max: float
    initial_log_std: float
    mean_output_init_scale: float = 1.0e-4
    log_std_output_init_scale: float = 0.0
    mean_latent_skip_scale: float = 0.0

    @nn.compact
    def __call__(self, observations, latents):
        x = jnp.concatenate((observations, latents), axis=-1)
        for width in self.hidden_dims:
            x = nn.gelu(nn.Dense(width, kernel_init=kernel_init())(x))
        mu = nn.Dense(self.action_dim, kernel_init=kernel_init(self.mean_output_init_scale), name="mu")(x)
        if self.mean_latent_skip_scale != 0.0:
            mu = mu + self.mean_latent_skip_scale*latents
        raw_log_std = nn.Dense(
            self.action_dim, kernel_init=kernel_init(self.log_std_output_init_scale),
            bias_init=nn.initializers.constant(self.initial_log_std), name="log_std",
        )(x)
        return jnp.tanh(mu), jnp.clip(raw_log_std, self.log_std_min, self.log_std_max)


class IBOLTPolicy(BaseJaxPolicy):
    """SB3-compatible normalized policy with one network evaluation per action."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        cfg,
        squash_output: bool = True,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=None,
            features_extractor_kwargs=None,
            squash_output=squash_output,
        )
        self.cfg = cfg
        self.use_sde = False

    def build(self, key, lr_schedule, qf_learning_rate: float):
        key, actor_key, qf_key, dropout_key, bn_key = jax.random.split(key, 5)
        key, self.key = jax.random.split(key)
        self.reset_noise()

        if isinstance(self.observation_space, spaces.Dict):
            obs_dim = spaces.flatdim(self.observation_space)
        else:
            obs_dim = int(np.prod(self.observation_space.shape))
        action_dim = self.action_space.shape[0]
        obs = jnp.zeros((1, obs_dim), dtype=jnp.float32)
        action = jnp.zeros((1, action_dim), dtype=jnp.float32)
        latent = jnp.zeros_like(action)

        self.qf = VectorCritic(
            net_arch=self.cfg.alg.critic.hs,
            n_critics=self.cfg.alg.critic.n_critics,
        )
        qf_variables = self.qf.init(
            {"params": qf_key, "dropout": dropout_key, "batch_stats": bn_key},
            obs,
            action,
            train=False,
        )
        self.qf_state = RLTrainState.create(
            apply_fn=self.qf.apply,
            params=qf_variables["params"],
            batch_stats=qf_variables.get("batch_stats", {}),
            target_params=qf_variables["params"],
            target_batch_stats=qf_variables.get("batch_stats", {}),
            tx=adam_with_grad_clip(
                learning_rate=qf_learning_rate,
                b1=self.cfg.alg.optimizer.critic_b1,
                b2=self.cfg.alg.optimizer.critic_b2,
                max_grad_norm=self.cfg.alg.optimizer.get("ac_grad_norm", None),
            ),
        )
        self.qf.apply = jax.jit(self.qf.apply)

        actor_cfg = self.cfg.alg.actor
        self.actor_model = SemiImplicitActor(
            action_dim=action_dim, hidden_dims=tuple(actor_cfg.hidden_dims),
            log_std_min=actor_cfg.log_std_min, log_std_max=actor_cfg.log_std_max,
            initial_log_std=actor_cfg.initial_log_std,
            mean_output_init_scale=actor_cfg.mean_output_init_scale,
            log_std_output_init_scale=actor_cfg.log_std_output_init_scale)
        actor_params = self.actor_model.init(actor_key, obs, latent)["params"]
        actor_tx = adam_with_grad_clip(
            learning_rate=self.cfg.alg.optimizer.lr_actor,
            b1=self.cfg.alg.optimizer.actor_b1,
            b2=self.cfg.alg.optimizer.actor_b2,
            max_grad_norm=self.cfg.alg.optimizer.get("ac_grad_norm", None),
        )
        self.actor_state = TrainState.create(
            apply_fn=self.actor_model.apply,
            params=actor_params,
            tx=actor_tx,
        )
        return key

    @staticmethod
    @partial(jax.jit, static_argnames=["deterministic", "sample_conditional_noise"])
    def sample_action(
        actor_state,
        observations,
        key,
        deterministic=False,
        sample_conditional_noise=True,
    ):
        latent_key, noise_key = jax.random.split(key)
        shape = (observations.shape[0], actor_state.params['mu']['bias'].shape[0])
        z = (jnp.zeros(shape,dtype=observations.dtype) if deterministic else
             jax.random.normal(latent_key,shape,dtype=observations.dtype))
        mu, log_std = actor_state.apply_fn({'params':actor_state.params},observations,z)
        if deterministic or not sample_conditional_noise:
            return mu
        return sample_box(noise_key,mu,log_std)

    def _predict(self, observation: np.ndarray, deterministic: bool = False):
        self.reset_noise()
        return self.sample_action(
            self.actor_state,
            observation,
            self.noise_key,
            deterministic=deterministic,
            # Evaluation callbacks may disable only the conditional Gaussian
            # draw while retaining a fresh stochastic latent z. Collection and
            # TD calls never set this flag and therefore keep the full policy.
            sample_conditional_noise=not bool(
                getattr(self, "evaluation_mu_only", False)
            ),
        )[0]

    def reset_noise(self, batch_size: int = 1) -> None:
        self.key, self.noise_key = jax.random.split(self.key)

    def forward(self, obs: np.ndarray, deterministic: bool = False):
        return self._predict(obs, deterministic=deterministic)

    def predict_critic(self, observation: np.ndarray, action: np.ndarray):
        self.reset_noise()
        return self.qf_state.apply_fn(
            {"params": self.qf_state.params, "batch_stats": self.qf_state.batch_stats},
            observation,
            action,
            rngs={"dropout": self.noise_key},
            train=False,
        )
