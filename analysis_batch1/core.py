"""Use the audited OptiQ actor update directly, without cached Hydra overrides."""
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import optax
from flax.training.train_state import TrainState

from analysis_boltzmann.problems import make_problem
from common.type_aliases import RLTrainState
from optiq_dime.algorithm import OptiQDIME
from optiq_dime.policy import ImplicitActor
from benchmarks.gmm40.latent_sampling import gaussian_grid


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def state_digest(state):
    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves((state.params, state.opt_state, state.step)):
        array = np.asarray(leaf)
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def config_for(case, version, seed):
    gmm = case == 'gmm40'
    assert version in ('ver1', 'ver2')
    n, repeats, anchor = (16, 5, True) if version == 'ver1' else (2048, 1, False)
    return dict(case=case, version=version, seed=seed, batch_size=1,
        num_policy_samples=n, proposals_per_policy_sample=repeats,
        candidate_count=n*repeats, random_candidate_count=n*(repeats-int(anchor)),
        include_anchor=anchor, temperature=1.0, proposal_std=1.0, proposal_clip=.5,
        proposal_sampling_mode='stratified', unbounded_actions=gmm,
        proposal_distribution='unbounded_gaussian_kde' if gmm else 'truncated_gaussian_kde',
        action_domain='R^2' if gmm else '[-1,1]^d', coordinate_scale=1.0,
        density_beta=1.0, adaptive_density_beta=False,
        sinkhorn_epsilon=.0001 if gmm else .05,
        sinkhorn_epsilon_start=.01 if gmm else .05,
        sinkhorn_anneal_updates=15000 if gmm else 0,
        sinkhorn_iterations=300 if gmm else 30,
        hidden_dims=[512]*5 if gmm else [256]*3,
        latent_sampling='grid' if gmm else 'iid',
        learning_rate=.0003, learning_rate_after_50k=.0001 if gmm else .0003,
        updates=85000 if gmm else 20000, source_q_eval='mean',
        transport_target_mode='argmax', distillation_loss='pointwise_mse',
        initialization='random', gradient_clipping=None,
        reference_temperature=1.0,
        q_definition='log p_GMM40' if gmm else 'original frozen Q unchanged (0.25 log mixture + constant)',
        checkpoints=([0,1,10,50,100,1000,5000,10000,20000,50000,75000,85000]
                     if gmm else [0,1,10,50,100,1000,5000,10000,20000]))


def initialize(cfg):
    if cfg['case'] == 'gmm40':
        from benchmarks.gmm40.sampler import initialize as gmm_initialize
        return gmm_initialize(cfg['seed'], cfg['hidden_dims'], cfg['learning_rate'], 1.0)
    problem = make_problem(cfg['case'])
    model = ImplicitActor(problem.dim, tuple(cfg['hidden_dims']))
    params = model.init(jax.random.PRNGKey(cfg['seed']), jnp.zeros((1,1)),
                        jnp.zeros((1,problem.dim)))['params']
    actor = TrainState.create(apply_fn=model.apply, params=params,
                             tx=optax.adam(cfg['learning_rate'], b1=.9, b2=.999))
    def oracle_apply(variables, observations, actions, **kwargs):
        del variables, observations, kwargs
        q = problem.q(actions, jnp, jsp.special.logsumexp)
        return jnp.stack((q,q), axis=0)[...,None]
    oracle = RLTrainState.create(apply_fn=oracle_apply, params={}, target_params={},
        batch_stats={}, target_batch_stats={}, tx=optax.identity())
    return actor, oracle, jax.random.PRNGKey(cfg['seed']+1000)


def call_arguments(cfg, key, step):
    gmm = cfg['case'] == 'gmm40'
    latents = None
    if cfg['latent_sampling'] == 'grid':
        latent_key = jax.random.split(key,4)[1]
        latents = gaussian_grid(latent_key,cfg['num_policy_samples'])[None]
    fraction = min(step / max(cfg['sinkhorn_anneal_updates'],1),1.)
    epsilon = (cfg['sinkhorn_epsilon_start'] *
               (cfg['sinkhorn_epsilon']/cfg['sinkhorn_epsilon_start'])**fraction)
    return dict(observations=jnp.zeros((1,0 if gmm else 1)), key=key,
        z_atoms=jnp.ones((1,)), num_policy_samples=cfg['num_policy_samples'],
        proposals_per_policy_sample=cfg['proposals_per_policy_sample'],
        proposal_sampling_mode='stratified', proposal_std=1.0, proposal_clip=.5,
        include_anchor=cfg['include_anchor'], density_correction=True, density_beta=1.0,
        adaptive_density_beta=False, minimum_source_ess=16., density_beta_grid_size=257,
        temperature=1.0, sinkhorn_epsilon=epsilon,
        sinkhorn_iterations=cfg['sinkhorn_iterations'], source_q_eval='mean',
        transport_target_mode='argmax', unbounded_actions=cfg['unbounded_actions'],
        policy_latents=latents)


def update(actor, oracle, key, cfg, step):
    return OptiQDIME.update_actor(actor_state=actor, qf_state=oracle,
                                 **call_arguments(cfg,key,step))


def sampler(actor, cfg):
    gmm = cfg['case']=='gmm40'
    dim = 2 if gmm else make_problem(cfg['case']).dim
    @jax.jit
    def draw(params, key):
        z = jax.random.normal(key,(4096,dim))
        actions = actor.apply_fn({'params':params},jnp.zeros((4096,0 if gmm else 1)),z)
        return actions if cfg['unbounded_actions'] else jnp.clip(actions,-1,1)
    def samples(params, key, count):
        chunks=[]
        for offset in range(0,count,4096):
            key,sub=jax.random.split(key)
            chunks.append(np.asarray(draw(params,sub))[:min(4096,count-offset)])
        return np.concatenate(chunks)
    return samples
