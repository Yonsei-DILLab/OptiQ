"""Unconditional adapter using the common OptiQ update and optional unbounded KDE."""
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import optax
import torch
from flax.training.train_state import TrainState

from common.type_aliases import RLTrainState
from optiq_dime.algorithm import OptiQDIME
from optiq_dime.policy import ImplicitActor
from optiq_dime.transport import clip_action
from .target_torch import GMM
from .latent_sampling import gaussian_grid


def make_target():
    # Preserve caller RNG; all network seeds use the same official target seed 0.
    with torch.random.fork_rng():
        return GMM(dim=2, n_mixes=40, loc_scaling=40, log_var_scaling=1, seed=0, device='cpu')


def log_prob(points, locs, scales):
    differences = (points[..., None, :] - locs) / scales
    components = -0.5 * jnp.sum(differences ** 2, axis=-1)
    components -= jnp.sum(jnp.log(scales), axis=-1) + jnp.log(2 * jnp.pi)
    return jsp.special.logsumexp(components, axis=-1) - jnp.log(locs.shape[0])


def oracle_apply(variables, observations, actions, rngs=None, train=False):
    del observations, rngs, train
    p = variables['params']
    q = log_prob(actions * p['coordinate_scale'], p['locs'], p['scales'])
    # Existing actor interface expects two heads; each supplies the same exact Q.
    return jnp.stack((q, q), axis=0)[..., None]


def initialize(seed, hidden_dims=(256, 256, 256), lr=0.0003, coordinate_scale=50.0,
               initial_output_std=0., initialization='random', initialization_noise=.01):
    target = make_target()
    model = ImplicitActor(action_dim=2, hidden_dims=tuple(hidden_dims))
    key, init_key = jax.random.split(jax.random.PRNGKey(seed))
    params = model.init(init_key, jnp.zeros((1, 0)), jnp.zeros((1, 2)))['params']
    if initialization not in ('random', 'identity', 'identity_wide'):
        raise ValueError('Unknown initialization: '+str(initialization))
    if initialization == 'identity_wide':
        if (initial_output_std <= 0 or initialization_noise < 0 or not hidden_dims
                or any(w < 4 or w % 2 for w in hidden_dims)):
            raise ValueError('Wide identity initialization requires positive std, nonnegative noise and even widths>=4')
        # Spread the same initial linear map across every existing hidden unit.
        # Paired GELU features obey GELU(u)-GELU(-u)=u. The equally spaced
        # directions form a tight frame, so decoding all pairs recovers z.
        # These are initial trainable weights only: no runtime frame, extra
        # layer, target information, skip connection or output transform.
        decoder = jnp.eye(2, dtype=jnp.float32)
        for i, width in enumerate(hidden_dims):
            half = width // 2
            rotation = jax.random.uniform(jax.random.fold_in(init_key, 3900+i), (),
                                          minval=0., maxval=jnp.pi)
            angles = rotation + jnp.arange(half, dtype=jnp.float32) * (jnp.pi/half)
            frame = jnp.sqrt(2.) * jnp.stack((jnp.cos(angles), jnp.sin(angles)))
            encoder = jnp.concatenate((frame, -frame), axis=1)
            layer = params[f'Dense_{i}']
            layer['kernel'] = decoder @ encoder + initialization_noise * layer['kernel']
            decoder = jnp.concatenate((frame.T, -frame.T), axis=0) / half
        output = params[f'Dense_{len(hidden_dims)}']
        output['kernel'] = decoder * initial_output_std + initialization_noise * output['kernel']
    elif initialization == 'identity':
        if initial_output_std <= 0 or initialization_noise < 0 or any(w < 4 for w in hidden_dims):
            raise ValueError('Identity initialization requires positive initial_output_std, nonnegative noise and widths>=4')
        # GELU(x)-GELU(-x)=x. Initialize four channels in each existing Dense
        # layer to carry signed coordinate pairs. This only sets trainable
        # weights: no skip connection, frozen map, output transform or target
        # information is added. Small seeded weights keep other units active.
        for layer in params.values():
            layer['kernel'] = layer['kernel'] * initialization_noise
        signed = jnp.array([[1., 0., -1., 0.], [0., 1., 0., -1.]], dtype=jnp.float32)
        params['Dense_0']['kernel'] = params['Dense_0']['kernel'].at[:, :4].set(signed)
        for i in range(1, len(hidden_dims)):
            pair_map = jnp.concatenate((signed, -signed), axis=0)
            name = f'Dense_{i}'
            params[name]['kernel'] = params[name]['kernel'].at[:4, :4].set(pair_map)
        output = params[f'Dense_{len(hidden_dims)}']
        output['kernel'] = output['kernel'].at[:4, :].set(signed.T * initial_output_std)
    elif initial_output_std > 0:
        # Target-free weight initialization: calibrate only the final affine
        # layer on independent standard Gaussian latents. No output coordinate
        # conversion or fixed multiplier remains in the trained generator.
        calibration_z = jax.random.normal(jax.random.fold_in(init_key, 271828), (4096, 2))
        raw = np.asarray(model.apply({'params': params}, jnp.zeros((4096, 0)), calibration_z))
        mean = raw.mean(axis=0)
        covariance = np.cov(raw, rowvar=False, bias=True)
        values, vectors = np.linalg.eigh(covariance)
        if values.min() <= 1e-14 or not np.isfinite(values).all():
            raise ValueError('Degenerate initial output covariance')
        transform = jnp.asarray((vectors * (initial_output_std / np.sqrt(values))) @ vectors.T,
                                dtype=jnp.float32)
        output = params[f'Dense_{len(hidden_dims)}']
        output['kernel'] = output['kernel'] @ transform
        output['bias'] = (output['bias'] - jnp.asarray(mean)) @ transform
    actor = TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(lr, b1=.9, b2=.999))
    oracle_params = {
        'locs': jnp.asarray(target.locs.numpy()),
        'scales': jnp.asarray(torch.diagonal(target.scale_trils, dim1=-2, dim2=-1).numpy()),
        'coordinate_scale': jnp.asarray(coordinate_scale),
    }
    oracle = RLTrainState.create(apply_fn=oracle_apply, params=oracle_params,
        tx=optax.identity(), target_params=oracle_params, batch_stats={}, target_batch_stats={})
    return actor, oracle, key


def update(actor, oracle, key, config):
    latent_sampling = config.get('latent_sampling', 'iid')
    policy_latents = None
    if latent_sampling == 'grid':
        latent_key = jax.random.split(key,4)[1]
        if config['batch_size'] == 1:
            policy_latents = gaussian_grid(latent_key,config['num_policy_samples'])[None]
        else:
            keys = jax.random.split(latent_key,config['batch_size'])
            policy_latents = jax.vmap(lambda k:gaussian_grid(k,config['num_policy_samples']))(keys)
    elif latent_sampling != 'iid':
        raise ValueError('Unknown latent_sampling: '+str(latent_sampling))
    return OptiQDIME.update_actor(
        actor_state=actor, qf_state=oracle,
        observations=jnp.zeros((config['batch_size'], 0), dtype=jnp.float32),
        key=key, z_atoms=jnp.ones((1,)),
        num_policy_samples=config['num_policy_samples'],
        proposals_per_policy_sample=config['proposals_per_policy_sample'],
        proposal_sampling_mode='stratified',
        proposal_std=config['proposal_std'], proposal_clip=config['proposal_clip'],
        include_anchor=config.get('include_anchor', False), density_correction=True,
        density_beta=config['density_beta'], adaptive_density_beta=False,
        minimum_source_ess=16., density_beta_grid_size=257,
        temperature=config['temperature'],
        sinkhorn_epsilon=config['sinkhorn_epsilon'], sinkhorn_iterations=config['sinkhorn_iterations'],
        source_q_eval='mean', transport_target_mode='argmax',
        unbounded_actions=config.get('unbounded_actions', False),
        policy_latents=policy_latents,
    )


def training_candidates_per_update(config):
    return config['batch_size'] * config['num_policy_samples'] * config['proposals_per_policy_sample']


def scheduled_config(config, update_index):
    """Optional exponential hyperparameter schedules; the OptiQ update is unchanged."""
    if config.get('anneal_updates', 0) <= 0:
        return config
    fraction = min(max(update_index / config['anneal_updates'], 0.), 1.)
    effective = dict(config)
    for field in ('temperature', 'proposal_std', 'sinkhorn_epsilon'):
        start = config.get(field + '_start', -1.)
        if start > 0:
            effective[field] = start * (config[field] / start) ** fraction
    return effective


@partial(jax.jit, static_argnames=['count', 'unbounded_actions'])
def draw(actor, key, count, coordinate_scale=50., unbounded_actions=False):
    z = jax.random.normal(key, (count, 2))
    raw = actor.apply_fn({'params': actor.params}, jnp.zeros((count, 0)), z)
    return (raw if unbounded_actions else clip_action(raw)) * coordinate_scale


def default_config():
    return dict(updates=30000, batch_size=256, hidden_dims=[256, 256, 256],
        learning_rate=.0003, initial_output_std=0., initialization='random', initialization_noise=.01,
        latent_sampling='iid', num_policy_samples=16, proposals_per_policy_sample=4,
        proposal_std=.2, proposal_clip=.5, include_anchor=False, unbounded_actions=False,
        density_beta=.1, temperature=.25,
        sinkhorn_epsilon=.05, sinkhorn_iterations=30, coordinate_scale=50.,
        temperature_start=-1., proposal_std_start=-1., sinkhorn_epsilon_start=-1., anneal_updates=0,
        eval_interval=1000, eval_samples=10000, checkpoint_interval=5000,
        log_interval=100, reference_seed=20260821)
