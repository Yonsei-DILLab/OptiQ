"""Read-only local Jacobian diagnostics; no training or density substitution."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import flax.serialization
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import wandb

from optiq_dime.runtime import load_environment
from .sampler import initialize


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    p.add_argument('--labels', nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--count', type=int, default=4096)
    p.add_argument('--seed', type=int, default=20261014)
    p.add_argument('--matmul-precision', choices=['default','highest'], default='highest')
    args = p.parse_args()
    if len(args.checkpoints) != len(args.labels):
        p.error('One label per checkpoint is required')
    args.output.mkdir(parents=True, exist_ok=False)
    load_environment()
    jax.config.update('jax_default_matmul_precision', args.matmul_precision)
    provenance = dict(diagnostic_only=True, actor_updated=False,
        not_paper_nll=True, seed=args.seed, count=args.count,
        matmul_precision=args.matmul_precision, jax_version=jax.__version__,
        precision_note='Diagnostic arithmetic only; identical saved weights. '
            'Existing training processes and exported sample files remain untouched. '
            'Highest precision probes the smooth float32-weight network, not bitwise fast GPU arithmetic.',
        interpretation='Jacobian signs and conditioning describe local geometry. '
            'Sign changes demonstrate an orientation change, not a numerical density estimate '
            'or proof that this alone causes the remaining metric gap.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    run = wandb.init(project=os.environ.get('WANDB_PROJECT','optiq_dime_no_anchor'),
        entity=os.environ.get('WANDB_ENTITY'), mode='online', dir=str(args.output),
        group='gmm40-idem-protocol', job_type='diagnostic', name='GMM40-generator-map-geometry',
        config=provenance)
    z = jax.random.normal(jax.random.PRNGKey(args.seed),(args.count,2))
    axis = np.linspace(-3,3,192,dtype=np.float32)
    zx,zy = np.meshgrid(axis,axis)
    grid = np.stack([zx.ravel(),zy.ravel()],axis=-1)
    result = dict(config=provenance, wandb_url=run.url, candidates=[])
    fig, axes = plt.subplots(2,len(args.labels),figsize=(5*len(args.labels),9),squeeze=False)
    try:
        for col,(path,label) in enumerate(zip(args.checkpoints,args.labels)):
            cfg = json.loads((path.parent/'config.json').read_text())
            if cfg['coordinate_scale'] != 1 or not cfg['unbounded_actions']:
                raise ValueError('This diagnostic requires direct native unbounded outputs')
            actor,oracle,_ = initialize(cfg['seed'],cfg['hidden_dims'],cfg['learning_rate'],1.)
            actor = flax.serialization.from_bytes(actor,path.read_bytes())
            @jax.jit
            def forward(points):
                return actor.apply_fn({'params':actor.params},jnp.zeros((points.shape[0],0)),points)
            jacobian = jax.jit(jax.vmap(jax.jacfwd(lambda point:forward(point[None])[0])))
            jac = np.concatenate([np.asarray(jacobian(z[i:i+256])) for i in range(0,len(z),256)])
            det = np.linalg.det(jac.astype(np.float64))
            singular = np.linalg.svd(jac.astype(np.float64),compute_uv=False)
            condition = singular[:,0]/np.maximum(singular[:,1],1e-15)
            # Independent finite differences on the actual map validate derivative scale.
            probe = z[:64]
            derivative_scale = max(np.sqrt(np.mean(jac[:64]**2)),1e-12)
            fd_errors = {}
            for delta in [.003,.001,.0003,.0001]:
                fd = np.stack([np.asarray((forward(probe+jnp.eye(2)[i]*delta)
                    -forward(probe-jnp.eye(2)[i]*delta))/(2*delta)) for i in range(2)],axis=-1)
                fd_errors[str(delta)] = float(np.sqrt(np.mean((fd-jac[:64])**2))/derivative_scale)
            reverse_jacobian = jax.jit(jax.vmap(jax.jacrev(lambda point:forward(point[None])[0])))
            reverse = np.asarray(reverse_jacobian(probe))
            reverse_error = float(np.sqrt(np.mean((reverse-jac[:64])**2))/derivative_scale)
            points = np.concatenate([np.asarray(forward(jnp.asarray(grid[i:i+4096])))
                                     for i in range(0,len(grid),4096)])
            locs = np.asarray(oracle.params['locs'])
            modes = ((points[:,None]-locs[None])**2).sum(-1).argmin(-1)
            eps = 1e-6 * np.maximum(singular[:,0]**2,1e-15)
            positive,negative = det > eps,det < -eps
            row = dict(label=label,checkpoint=str(path),
                checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                actor_update=int(actor.step), parent_config=cfg,
                positive_fraction=float(positive.mean()),negative_fraction=float(negative.mean()),
                near_singular_fraction=float((~(positive|negative)).mean()),
                abs_determinant_quantiles=np.quantile(np.abs(det),[.01,.1,.5,.9,.99]).tolist(),
                condition_quantiles=np.quantile(condition,[.01,.1,.5,.9,.99]).tolist(),
                finite_difference_relative_rms=fd_errors['0.0001'],
                finite_difference_step=.0001, finite_difference_errors_by_step=fd_errors,
                reverse_autodiff_relative_rms=reverse_error)
            # Retain every numerical probe even if a validation condition fails.
            (args.output/f'candidate_{col}_numerics.json').write_text(json.dumps(row,indent=2)+'\n')
            print(json.dumps(dict(label=label,finite_difference_errors=fd_errors,
                                  reverse_autodiff_relative_rms=reverse_error)),flush=True)
            if not np.isfinite(jac).all() or fd_errors['0.0001'] > .03 or reverse_error > 1e-4:
                raise ValueError('Nonfinite Jacobian or finite-difference disagreement')
            result['candidates'].append(row)
            np.savez(args.output/f'candidate_{col}.npz',z=np.asarray(z),jacobian=jac,
                     determinant=det,condition=condition,grid=grid,grid_outputs=points,grid_modes=modes)
            axes[0,col].imshow(modes.reshape(zx.shape),origin='lower',extent=(-3,3,-3,3),
                               cmap='turbo',vmin=0,vmax=39,interpolation='nearest')
            axes[0,col].set_title(label+'\nNearest output component in latent space')
            scatter=axes[1,col].scatter(np.asarray(z)[:,0],np.asarray(z)[:,1],
                c=np.log10(condition),s=3,cmap='viridis',vmin=0,vmax=4)
            fig.colorbar(scatter,ax=axes[1,col],label='log10 Jacobian condition number')
            axes[1,col].set_title(f'Opposite orientation: {100*negative.mean():.1f}% negative')
            for ax in axes[:,col]:
                ax.set_xlim(-3,3);ax.set_ylim(-3,3);ax.set_aspect('equal');ax.set_xlabel('z₁');ax.set_ylabel('z₂')
            metrics={k:v for k,v in row.items() if isinstance(v,(int,float))}
            run.log({label+'/'+key:value for key,value in metrics.items()})
            print(json.dumps(dict(label=label,**metrics)),flush=True)
        fig.suptitle('Fixed generator geometry — diagnostic only; no actor updates')
        fig.tight_layout();fig.savefig(args.output/'map_geometry.png',dpi=160);plt.close(fig)
        (args.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        run.log({'map_geometry':wandb.Image(str(args.output/'map_geometry.png'))})
        artifact=wandb.Artifact('gmm40-map-geometry-'+run.id,type='diagnostic')
        for file in args.output.iterdir():
            if file.is_file():artifact.add_file(str(file))
        run.log_artifact(artifact);run.finish()
    except BaseException as exc:
        (args.output/'failed.json').write_text(json.dumps(dict(error=repr(exc),wandb_url=run.url))+'\n')
        run.finish(exit_code=1)
        raise


if __name__ == '__main__':
    main()
