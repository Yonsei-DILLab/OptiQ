"""JAX port of haarnoja/softqlearning at 6f51eac (amortized SVGD SQL).

See SQL_BASELINE.md for provenance and the explicit experiment-level changes.
The bounded-action score, particle split, sum-reduced policy gradient, TF1 Adam
epsilon placement and hard target copies follow the original implementation.
"""
from dataclasses import asdict, dataclass
import json
import math

import flax.linen as nn
import flax.serialization
from flax import struct
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax

UPSTREAM_COMMIT = "6f51eaca77d15b35c6443363c51a5a53ff4e9854"


@dataclass(frozen=True)
class SQLConfig:
    hidden_dims: tuple = (256, 256)
    learning_rate: float = 3e-4
    kernel_particles: int = 16
    kernel_update_ratio: float = .5
    value_particles: int = 16
    target_update_interval: int = 1000
    discount: float = .99
    reward_scale: float = 1.
    temperature: float = 1.

    def __post_init__(self):
        if not self.hidden_dims or any(int(x) != x or x <= 0 for x in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive integers")
        for name in ("kernel_particles", "value_particles", "target_update_interval"):
            x = getattr(self, name)
            if not isinstance(x, int) or x <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.kernel_update_ratio) or not 0 < self.kernel_update_ratio < 1:
            raise ValueError("kernel_update_ratio must be between 0 and 1")
        updated = int(self.kernel_particles * self.kernel_update_ratio)
        if updated < 1 or self.kernel_particles - updated < 2:
            raise ValueError("SVGD needs at least two fixed particles and one updated particle")
        for name in ("learning_rate", "reward_scale", "temperature"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.discount) or not 0 <= self.discount <= 1:
            raise ValueError("discount must lie in [0, 1]")


class Network(nn.Module):
    """Separate first-layer input projections, as in upstream feedforward_net."""
    hidden_dims: tuple
    output_dim: int
    squash: bool = False

    @nn.compact
    def __call__(self, observations, inputs):
        init = nn.initializers.glorot_uniform()
        first = self.hidden_dims[0]
        x = nn.Dense(first, use_bias=False, kernel_init=init, name="observation")(observations)
        x = x + nn.Dense(first, use_bias=False, kernel_init=init, name="input")(inputs)
        x = nn.relu(x + self.param("first_bias", nn.initializers.zeros, (first,)))
        for i, width in enumerate(self.hidden_dims[1:]):
            x = nn.relu(nn.Dense(width, kernel_init=init, name=f"hidden_{i}")(x))
        x = nn.Dense(self.output_dim, kernel_init=init, name="output")(x)
        return jnp.tanh(x) if self.squash else x[..., 0]


def tf1_adam(learning_rate, b1=.9, b2=.999, eps=1e-8):
    """TF1 Adam uses epsilon on uncorrected v, unlike optax.adam's default."""
    def init(params):
        zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
        return optax.ScaleByAdamState(jnp.zeros([], jnp.int32), zeros, zeros)

    def update(grads, state, params=None):
        del params
        count = optax.safe_int32_increment(state.count)
        mu = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, state.mu, grads)
        nu = jax.tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, state.nu, grads)
        rate = learning_rate * jnp.sqrt(1 - b2 ** count) / (1 - b1 ** count)
        updates = jax.tree_util.tree_map(lambda m, v: -rate * m / (jnp.sqrt(v) + eps), mu, nu)
        return updates, optax.ScaleByAdamState(count, mu, nu)
    return optax.GradientTransformation(init, update)


def adaptive_isotropic_gaussian_kernel(xs, ys, h_min=1e-3):
    diff = xs[..., :, None, :] - ys[..., None, :, :]
    dist_sq = jnp.sum(diff * diff, axis=-1)
    pairs = dist_sq.shape[-2] * dist_sq.shape[-1]
    # TF top_k(..., K//2+1)[-1] picks the LOWER middle for even K.
    # jnp.median would average the two middle entries and change the algorithm.
    median = jnp.sort(dist_sq.reshape(dist_sq.shape[:-2] + (pairs,)), axis=-1)[..., (pairs - 1) // 2]
    h = jax.lax.stop_gradient(jnp.maximum(median / math.log(xs.shape[-2]), h_min))
    output = jnp.exp(-dist_sq / h[..., None, None])
    gradient = -2 * diff * output[..., None] / h[..., None, None, None]
    return {"output": output, "gradient": gradient, "bandwidth": h}


def svgd_direction(q_fn, observations, fixed, updated, temperature=1.):
    fixed = jax.lax.stop_gradient(fixed)
    def log_density(actions):
        q = q_fn(observations[:, None, :], actions)
        correction = jnp.log(1 - actions ** 2 + 1e-6).sum(-1)
        return (q / temperature + correction).sum()
    score = jax.lax.stop_gradient(jax.grad(log_density)(fixed))
    kernel = adaptive_isotropic_gaussian_kernel(fixed, updated)
    phi = (kernel["output"][..., None] * score[:, :, None, :] + kernel["gradient"]).mean(axis=1)
    return jax.lax.stop_gradient(phi), kernel["bandwidth"]


def actor_gradients(actor, observations, latents, q_fn, config):
    updated_count = int(config.kernel_particles * config.kernel_update_ratio)
    fixed_count = config.kernel_particles - updated_count
    def loss(params):
        actions = actor.apply_fn({"params": params}, observations[:, None, :], latents)
        fixed, updated = actions[:, :fixed_count], actions[:, fixed_count:]
        phi, bandwidth = svgd_direction(q_fn, observations, fixed, updated, config.temperature)
        # TF gradients(updated_actions, params, grad_ys=phi) SUMS B and K.
        surrogate = -jnp.sum(updated * phi)
        return surrogate, dict(svgd_norm=jnp.sqrt(jnp.mean(phi ** 2)),
                               kernel_bandwidth=bandwidth.mean(),
                               action_spread=jnp.var(actions, axis=1).mean())
    (_, info), grads = jax.value_and_grad(loss, has_aux=True)(actor.params)
    return grads, {**info, "actor_grad_norm": optax.global_norm(grads)}


def soft_value(q_values, temperature=1.):
    return temperature * (jax.scipy.special.logsumexp(q_values / temperature, axis=-1)
                          - math.log(q_values.shape[-1]))


def td_targets(next_q, rewards, terminals, action_dim, config):
    value = soft_value(next_q, config.temperature) + config.temperature * action_dim * math.log(2)
    return jax.lax.stop_gradient(config.reward_scale * rewards +
                                (1 - terminals) * config.discount * value)


@struct.dataclass
class SQLState:
    actor: TrainState
    critic: object
    target_critic: object
    key: object
    updates: object


class SQLLearner:
    """Dimension-independent SQL learner; fixed_q optionally removes critic learning."""
    def __init__(self, observation_dim, action_dim, seed=0, config=SQLConfig(), fixed_q=None, checkpoint_context=None):
        if observation_dim <= 0 or action_dim <= 0:
            raise ValueError("Observation and action dimensions must be positive")
        self.config, self.observation_dim, self.action_dim = config, observation_dim, action_dim
        self.fixed_q = fixed_q
        self.checkpoint_context = checkpoint_context
        self.actor = Network(config.hidden_dims, action_dim, squash=True)
        self.critic = Network(config.hidden_dims, 1)
        key, actor_key, critic_key = jax.random.split(jax.random.PRNGKey(seed), 3)
        obs, act = jnp.zeros((1, observation_dim)), jnp.zeros((1, action_dim))
        def create(net, rng):
            return TrainState.create(apply_fn=net.apply, params=net.init(rng, obs, act)["params"],
                                     tx=tf1_adam(config.learning_rate))
        actor = create(self.actor, actor_key)
        critic = create(self.critic, critic_key) if fixed_q is None else None
        self.state = SQLState(actor, critic, critic.params if critic else None, key, jnp.array(0, jnp.int32))
        self.update_fn = jax.jit(self._update)
        self.advance_fn = jax.jit(self._advance, static_argnums=(1, 2))
        self.sample_fn = jax.jit(lambda params, obs, rng: self.actor.apply(
            {"params": params}, obs, jax.random.normal(rng, obs.shape[:-1] + (action_dim,))))
        self.q_fn = jax.jit(lambda params, obs, a: self.critic.apply({"params": params}, obs, a))

    @property
    def signature(self):
        return json.dumps(dict(config=asdict(self.config), observation_dim=self.observation_dim,
                               action_dim=self.action_dim, fixed_q=self.fixed_q is not None,
                               checkpoint_context=self.checkpoint_context,
                               upstream_commit=UPSTREAM_COMMIT), sort_keys=True)

    def _policy_update(self, state, observations, rng):
        latents = jax.random.normal(rng, (len(observations), self.config.kernel_particles, self.action_dim))
        q_fn = self.fixed_q if self.fixed_q is not None else lambda s, a: self.critic.apply(
            {"params": state.critic.params}, s, a)
        grads, info = actor_gradients(state.actor, observations, latents, q_fn, self.config)
        return state.actor.apply_gradients(grads=grads), info

    def _update(self, state, batch, iteration):
        key, policy_key, value_key = jax.random.split(state.key, 3)
        # Both gradients read the pre-update live Q. The target copy follows both.
        actor, info = self._policy_update(state, batch["observations"], policy_key)
        actions = jax.random.uniform(value_key, (1, self.config.value_particles, self.action_dim), minval=-1, maxval=1)
        next_q = self.critic.apply({"params": state.target_critic}, batch["next_observations"][:, None, :], actions)
        ys = td_targets(next_q, batch["rewards"], batch["terminals"], self.action_dim, self.config)
        def loss(params):
            q = self.critic.apply({"params": params}, batch["observations"], batch["actions"])
            return .5 * jnp.mean((ys - q) ** 2)
        value, grad = jax.value_and_grad(loss)(state.critic.params)
        critic = state.critic.apply_gradients(grads=grad)
        copy = iteration % self.config.target_update_interval == 0
        target = jax.lax.cond(copy, lambda: critic.params, lambda: state.target_critic)
        state = state.replace(actor=actor, critic=critic, target_critic=target, key=key, updates=state.updates + 1)
        return state, {**info, "critic_loss": value, "target_mean": ys.mean(), "target_copied": copy.astype(jnp.float32)}

    def _advance(self, state, count, batch_size):
        def step(state, _):
            key, rng = jax.random.split(state.key)
            actor, info = self._policy_update(state, jnp.zeros((batch_size, self.observation_dim)), rng)
            return state.replace(actor=actor, key=key, updates=state.updates + 1), info
        state, infos = jax.lax.scan(step, state, None, length=count)
        return state, jax.tree_util.tree_map(jnp.mean, infos)

    def update(self, batch, iteration=None):
        if self.fixed_q is not None:
            raise ValueError("Use advance for fixed-Q reconstruction")
        required = {"observations", "actions", "next_observations", "rewards", "terminals"}
        if set(batch) != required:
            raise ValueError(f"Batch must contain exactly {sorted(required)}")
        batch = {k: jnp.asarray(v, jnp.float32) for k, v in batch.items()}
        size = batch["observations"].shape[0]
        for key, shape in (("observations", (size, self.observation_dim)),
                           ("next_observations", (size, self.observation_dim)),
                           ("actions", (size, self.action_dim)), ("rewards", (size,)), ("terminals", (size,))):
            if batch[key].shape != shape or size == 0:
                raise ValueError(f"Unexpected {key} shape: {batch[key].shape}; expected {shape}")
        self.state, info = self.update_fn(self.state, batch, self.state.updates if iteration is None else iteration)
        return info

    def advance(self, count, batch_size):
        if self.fixed_q is None or count <= 0 or batch_size <= 0:
            raise ValueError("advance requires fixed Q and positive count/batch size")
        self.state, info = self.advance_fn(self.state, count, batch_size)
        return info

    def save(self, path):
        path.write_bytes(flax.serialization.to_bytes(dict(state=self.state, signature=self.signature)))

    def restore(self, path):
        raw = flax.serialization.msgpack_restore(path.read_bytes())
        if raw["signature"] != self.signature:
            raise ValueError("SQL checkpoint configuration does not match this learner")
        self.state = flax.serialization.from_state_dict(self.state, raw["state"])
