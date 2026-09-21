"""Read-only OT marginal and argmax diagnostics for a fixed OptiQ checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import wandb

from optiq_dime.runtime import load_environment
from optiq_dime.transport import GaussianKDE, sinkhorn
from .sampler import initialize, log_prob
from .train import write_json
from .latent_sampling import gaussian_grid


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=20261005)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--temperature', type=float, help='Optional diagnostic temperature; checkpoint stays unchanged')
    p.add_argument('--latent-sampling', choices=['iid','grid'], default='iid')
    p.add_argument('--num-policy-samples', type=int)
    p.add_argument('--proposal-std', type=float)
    p.add_argument('--candidates-per-center', type=int, nargs='+', default=[4, 16])
    p.add_argument('--epsilons', type=float, nargs='+', default=[.001, .0001])
    p.add_argument('--iterations', type=int, nargs='+', default=[30, 300, 3000])
    a = p.parse_args()
    cfg = json.loads((a.checkpoint.parent/'config.json').read_text())
    if cfg['coordinate_scale'] != 1. or not cfg['unbounded_actions'] or cfg['include_anchor']:
        p.error('This diagnostic currently supports native, unbounded, no-anchor runs')
    temperature = cfg['temperature'] if a.temperature is None else a.temperature
    proposal_std = cfg['proposal_std'] if a.proposal_std is None else a.proposal_std
    if temperature <= 0:
        p.error('Temperature must be positive')
    load_environment()
    a.output.mkdir(parents=True, exist_ok=False)
    record = {k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    record.update(parent_config=cfg, diagnostic_only=True, actor_updated=False,
                  checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    run = wandb.init(project=os.environ.get('WANDB_PROJECT','optiq_dime_no_anchor'),
        entity=os.environ.get('WANDB_ENTITY'), mode='online', dir=str(a.output),
        group='gmm40-ot-diagnostics', job_type='diagnostic', config=record,
        name='OT-marginals-'+a.checkpoint.parent.parent.name)
    try:
        actor, oracle, _ = initialize(cfg['seed'],cfg['hidden_dims'],cfg['learning_rate'],1.)
        actor = flax.serialization.from_bytes(actor,a.checkpoint.read_bytes())
        locs, scales = oracle.params['locs'], oracle.params['scales']
        n = cfg['num_policy_samples'] if a.num_policy_samples is None else a.num_policy_samples
        solve = jax.jit(sinkhorn, static_argnames=['iterations'])
        results = []
        for repeat in range(a.repeats):
            _, latent_key, proposal_key, _ = jax.random.split(jax.random.PRNGKey(a.seed+repeat),4)
            z = (jax.random.normal(latent_key,(n,2)) if a.latent_sampling == 'iid'
                 else gaussian_grid(latent_key,n))
            centers = actor.apply_fn({'params':actor.params},jnp.zeros((n,0)),z)[None]
            center_logp = log_prob(centers,locs,scales)
            center_modes = jnp.square(centers[0,:,None,:]-locs[None,:,:]).sum(-1).argmin(-1)
            occupancy = jnp.bincount(center_modes,length=40)/n
            for r in a.candidates_per_center:
                kde = GaussianKDE(centers,proposal_std)
                candidates = kde.sample_stratified(proposal_key,r,False).reshape(1,n*r,2)
                candidate_logp = log_prob(candidates,locs,scales)
                weights = jax.nn.softmax(candidate_logp/temperature-cfg['density_beta']*kde.log_prob(candidates),axis=-1)
                squared = jnp.square(centers[:,:,None,:]-candidates[:,None,:,:]).sum(-1)
                costs = squared/(squared.mean(axis=(-2,-1),keepdims=True)+1e-8)
                for eps in a.epsilons:
                    for iterations in a.iterations:
                        transport = solve(costs,weights,eps,iterations)
                        rows, columns = transport.sum(-1), transport.sum(-2)
                        distribution = transport/jnp.maximum(rows[...,None],1e-20)
                        indices = jnp.argmax(distribution,axis=-1)
                        selected = candidates[0,indices[0]]
                        row = dict(repeat=repeat,n=n,r=r,epsilon=eps,iterations=iterations,temperature=temperature,proposal_std=proposal_std,
                            row_marginal_tv=float(.5*jnp.abs(rows-1/n).sum()),
                            column_marginal_tv=float(.5*jnp.abs(columns-weights).sum()),
                            max_relative_row_error=float(jnp.abs(rows*n-1).max()),
                            source_ess=float(1/jnp.square(weights).sum()),
                            current_mean_logp=float(center_logp.mean()),
                            current_occupancy_tv=float(.5*jnp.abs(occupancy-1/40).sum()),
                            candidate_weighted_logp=float((weights*candidate_logp).sum()),
                            selected_mean_logp=float(log_prob(selected,locs,scales).mean()),
                            selected_delta_l2=float(jnp.linalg.norm(selected-centers[0],axis=-1).mean()),
                            selected_unique_fraction=float(len(np.unique(np.asarray(indices)))/n))
                        results.append(row)
                        run.log(row)
                        with (a.output/'history.jsonl').open('a') as f:
                            f.write(json.dumps(row,allow_nan=False)+'\n')
                        print(json.dumps(row),flush=True)
        record.update(wandb_url=run.url,actor_update=int(actor.step),results=results)
        write_json(a.output/'summary.json',record)
        artifact = wandb.Artifact('gmm40-ot-diagnostic-'+run.id,type='diagnostic')
        artifact.add_file(str(a.output/'summary.json'));run.log_artifact(artifact);run.finish()
    except BaseException:
        run.finish(exit_code=1)
        raise


if __name__ == '__main__':
    main()
