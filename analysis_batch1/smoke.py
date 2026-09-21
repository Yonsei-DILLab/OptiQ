"""Validate actual sample clouds and reconstruct the OT regression loss on GPU."""
import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from analysis_boltzmann.problems import make_problem
from optiq_dime.transport import GaussianKDE, TruncatedGaussianKDE, sinkhorn
from .core import config_for, initialize, state_digest, update, write_json
from benchmarks.gmm40.latent_sampling import gaussian_grid


def check(case,version):
    cfg=config_for(case,version,0)
    actor,oracle,key=initialize(cfg);digest=state_digest(actor)
    n,m=cfg['num_policy_samples'],cfg['candidate_count']
    gmm=case=='gmm40';dim=2 if gmm else make_problem(case).dim
    returned,latent_key,proposal_key,_=jax.random.split(key,4)
    latents=(gaussian_grid(latent_key,n)[None] if gmm else
             jax.random.normal(latent_key,(1,n,dim)))
    obs=jnp.zeros((n,0 if gmm else 1))
    raw=actor.apply_fn({'params':actor.params},obs,latents.reshape(n,dim))[None]
    centers=raw if gmm else jnp.clip(raw,-1,1)
    kde=GaussianKDE(centers,1.) if gmm else TruncatedGaussianKDE.from_centers(centers,1.,.5)
    candidates=kde.sample_stratified(proposal_key,cfg['proposals_per_policy_sample'],cfg['include_anchor'])
    assert candidates.shape==(1,n,cfg['proposals_per_policy_sample'],dim)
    if version=='ver1': np.testing.assert_array_equal(candidates[:,:,0,:],centers)
    if not gmm:
        assert float(jnp.max(abs(candidates)))<=1.000001
        assert float(jnp.max(abs(candidates-centers[:,:,None,:])))<=.500001
    proposals=candidates.reshape(1,m,dim)
    q=oracle.apply_fn({'params':oracle.params},jnp.zeros((m,0 if gmm else 1)),
                       proposals.reshape(m,dim),train=False).mean(0).reshape(1,m)
    weights=jax.nn.softmax(q-kde.log_prob(proposals),axis=-1)  # T=1, beta=1
    costs=((centers[:,:,None,:]-proposals[:,None,:,:])**2).sum(-1)
    costs=costs/(costs.mean(axis=(-2,-1),keepdims=True)+1e-8)
    coupling=sinkhorn(costs,weights,cfg['sinkhorn_epsilon_start'],cfg['sinkhorn_iterations'])
    assert coupling.shape==(1,n,m)
    normalized=coupling/jnp.maximum(coupling.sum(-1,keepdims=True),1e-20)
    targets=proposals[0,jnp.argmax(normalized[0],axis=-1)]
    expected_loss=float(((raw[0]-targets)**2).sum(-1).mean())
    begin=time.monotonic();updated,loss,next_key,metrics=update(actor,oracle,key,cfg,0)
    actual=float(loss);seconds=time.monotonic()-begin
    np.testing.assert_allclose(actual,expected_loss,rtol=2e-4,atol=2e-5)
    np.testing.assert_array_equal(next_key,returned)
    assert state_digest(updated)!=digest
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(updated.params))
    assert np.isfinite(actual)
    row=dict(case=case,version=version,initial_state_sha256=digest,ot_shape=list(coupling.shape),
             anchor_count=n if cfg['include_anchor'] else 0,temperature=1.,proposal_std=1.,
             unbounded=gmm,actual_loss=actual,independent_loss=expected_loss,loss_verified=True,
             first_update_including_compile_seconds=seconds)
    print(json.dumps(row),flush=True)
    return row


def run(campaign):
    assert jax.default_backend()=='gpu'
    rows=[]
    for case in ('modes2d_8','gmm40','separable_8'):
        pair=[check(case,v) for v in ('ver1','ver2')]
        assert pair[0]['initial_state_sha256']==pair[1]['initial_state_sha256']
        rows+=pair
        jax.clear_caches()
    write_json(Path(campaign)/'smoke_passed.json',dict(passed=True,rows=rows,
        devices=[str(d) for d in jax.devices()],paired_initialization_verified=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True)
    run(parser.parse_args().campaign)
