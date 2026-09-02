"""One-step implicit OptiQ policy backed by DIME's distributional critic."""

from collections.abc import Sequence
from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from gymnasium import spaces

from common.policies import BaseJaxPolicy
from common.type_aliases import RLTrainState
from models.critic import VectorCritic
from models.utils import activation_fn

from .transport import clip_action


def kernel_init(scale: float = 1.0):
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


class ImplicitActor(nn.Module):
    action_dim: int
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations: jax.Array, latents: jax.Array) -> jax.Array:
        x = jnp.concatenate((observations, latents), axis=-1)
        for width in self.hidden_dims:
            x = nn.Dense(width, kernel_init=kernel_init())(x)
            x = nn.gelu(x)
        return nn.Dense(
            self.action_dim,
            kernel_init=kernel_init(1.0e-2),
        )(x)


class OptiQPolicy(BaseJaxPolicy):
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
            dropout_rate=self.cfg.alg.critic.dropout_rate,
            use_layer_norm=self.cfg.alg.critic.use_layer_norm,
            use_batch_norm=self.cfg.alg.optimizer.bn,
            bn_warmup=self.cfg.alg.optimizer.bn_warmup,
            batch_norm_momentum=self.cfg.alg.optimizer.bn_momentum,
            batch_norm_mode=self.cfg.alg.optimizer.bn_mode,
            net_arch=self.cfg.alg.critic.hs,
            activation_fn=activation_fn[self.cfg.alg.critic.activation],
            n_critics=self.cfg.alg.critic.n_critics,
            n_atoms=self.cfg.alg.critic.n_atoms,
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
            batch_stats=qf_variables["batch_stats"],
            target_params=qf_variables["params"],
            target_batch_stats=qf_variables["batch_stats"],
            tx=optax.adam(
                learning_rate=qf_learning_rate,
                b1=self.cfg.alg.optimizer.critic_b1,
                b2=self.cfg.alg.optimizer.critic_b2,
            ),
        )
        self.qf.apply = jax.jit(
            self.qf.apply,
            static_argnames=(
                "dropout_rate",
                "use_layer_norm",
                "use_batch_norm",
                "batch_norm_momentum",
                "bn_mode",
            ),
        )

        self.actor_model = ImplicitActor(
            action_dim=action_dim,
            hidden_dims=tuple(self.cfg.alg.actor.hidden_dims),
        )
        actor_params = self.actor_model.init(actor_key, obs, latent)["params"]
        actor_tx = optax.adam(
            learning_rate=self.cfg.alg.optimizer.lr_actor,
            b1=self.cfg.alg.optimizer.actor_b1,
            b2=self.cfg.alg.optimizer.actor_b2,
        )
        self.actor_state = TrainState.create(
            apply_fn=self.actor_model.apply,
            params=actor_params,
            tx=actor_tx,
        )
        self.target_actor_state = TrainState.create(
            apply_fn=self.actor_model.apply,
            params=actor_params,
            tx=actor_tx,
        )
        return key

    @staticmethod
    @partial(jax.jit, static_argnames=["deterministic"])
    def sample_action(actor_state, observations, key, deterministic=False):
        output_layer = f"Dense_{len(actor_state.params) - 1}"
        latent_shape = (observations.shape[0],) + (
            actor_state.params[output_layer]["bias"].shape[0],
        )
        latents = (
            jnp.zeros(latent_shape, dtype=observations.dtype)
            if deterministic
            else jax.random.normal(key, latent_shape, dtype=observations.dtype)
        )
        raw_actions = actor_state.apply_fn(
            {"params": actor_state.params}, observations, latents
        )
        return clip_action(raw_actions)

    def _predict(self, observation: np.ndarray, deterministic: bool = False):
        self.reset_noise()
        return self.sample_action(
            self.actor_state,
            observation,
            self.noise_key,
            deterministic=deterministic,
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
