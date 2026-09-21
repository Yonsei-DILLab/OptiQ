"""Read-only mode support diagnostics for direct generator samples; not paper metrics."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from scipy.special import logsumexp
import torch
import wandb

from .target_torch import GMM

ROOT = Path(__file__).resolve().parents[2]


def mode_statistics(samples, locs, scales):
    """Posterior-responsibility moments; these do not alter the input samples."""
    mass = np.zeros(len(locs))
    first = np.zeros_like(locs, dtype=np.float64)
    second = np.zeros((len(locs), 2, 2))
    tails = np.zeros(len(locs))
    for x in np.array_split(samples.astype(np.float64), max(1, len(samples) // 4096)):
        residual = (x[:, None, :] - locs) / scales
        distance2 = (residual**2).sum(-1)
        logits = -.5 * distance2 - np.log(scales).sum(-1)
        weights = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
        mass += weights.sum(0)
        first += weights.T @ x
        second += np.einsum('nk,ni,nj->kij', weights, x, x)
        tails += (weights * (distance2 > -2 * np.log(.05))).sum(0)
    means = first / mass[:, None]
    covariance = second / mass[:, None, None] - means[:, :, None] * means[:, None, :]
    relative_covariance = covariance / (scales[:, :, None] * scales[:, None, :])
    eigenvalues = np.linalg.eigvalsh(relative_covariance)
    return dict(mode_mass=(mass / len(samples)).tolist(),
        mean_error_in_sigma=np.linalg.norm((means-locs)/scales, axis=-1).tolist(),
        covariance_eigenvalues_relative_to_target=eigenvalues.tolist(),
        outside_95pct_ellipse=(tails/mass).tolist(),
        mean_absolute_mode_mass_error=float(np.abs(mass/len(samples)-1/len(locs)).mean()))


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--samples', type=Path, nargs='+', required=True)
    p.add_argument('--labels', nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if len(a.samples) != len(a.labels):
        p.error('One label per sample file is required')
    for file in [os.environ.get('OPTIQ_ENV_FILE'), ROOT/'.env', ROOT.parent/'.env']:
        if file:
            load_dotenv(file, override=False)
    a.output.mkdir(parents=True, exist_ok=False)
    config = dict(diagnostic_only=True, not_paper_metrics=True,
        reference_seed=20261007, samples_are_direct_generator_outputs=True,
        inputs={label:dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                for label, path in zip(a.labels, a.samples)},
        local_mode_selection='Largest posterior covariance anisotropy in the first supplied sampler; all40 mode statistics also retained',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    run = wandb.init(project=os.environ.get('WANDB_PROJECT', 'optiq_dime_no_anchor'),
        entity=os.environ.get('WANDB_ENTITY'), mode='online', dir=str(a.output),
        group='gmm40-idem-protocol', job_type='diagnostic', name='GMM40-mode-support', config=config)
    try:
        torch.set_num_threads(1)
        target = GMM(2, 40, 40, log_var_scaling=1., seed=0, device='cpu')
        torch.random.default_generator.manual_seed(config['reference_seed'])
        sets = {'GT independent':target.sample((100000,)).numpy()}
        sets.update({label:np.load(path) for label,path in zip(a.labels,a.samples)})
        locs = target.locs.numpy().astype(np.float64)
        scales = torch.diagonal(target.scale_trils, dim1=-2, dim2=-1).numpy().astype(np.float64)
        for points in sets.values():
            if points.shape != (100000,2) or not np.isfinite(points).all():
                raise ValueError('Expected100k finite samples in original coordinates')
        statistics = {label:mode_statistics(points, locs, scales) for label,points in sets.items()}
        eigen = np.asarray(statistics[a.labels[0]]['covariance_eigenvalues_relative_to_target'])
        local_mode = int(np.argmax(eigen[:,1] / np.maximum(eigen[:,0],1e-12)))
        fig, axes = plt.subplots(2,len(sets),figsize=(5*len(sets),9),squeeze=False)
        center = locs[local_mode]; radius = 4*scales[local_mode,0]
        for col,(label,points) in enumerate(sets.items()):
            for row in range(2):
                ax = axes[row,col]
                shown = points[:10000] if row == 0 else points[np.linalg.norm(points-center,axis=1)<radius]
                ax.scatter(shown[:,0],shown[:,1],s=1,alpha=.3,rasterized=True)
                for loc,scale in zip(locs,scales):
                    ax.add_patch(Circle(loc,scale[0],fill=False,color='tab:red',lw=.5,alpha=.5))
                ax.set_aspect('equal')
                if row == 0:
                    ax.set_xlim(-46,46);ax.set_ylim(-46,46);ax.set_title(label)
                else:
                    ax.set_xlim(center[0]-radius,center[0]+radius)
                    ax.set_ylim(center[1]-radius,center[1]+radius)
                    ax.set_title(f'Local support: component {local_mode}')
                ax.set_xlabel('x₁');ax.set_ylabel('x₂')
        fig.suptitle('Direct g(z) in original coordinates; circles show target component 1σ',fontsize=14)
        fig.tight_layout();fig.savefig(a.output/'mode_support.png',dpi=160);plt.close(fig)
        result = dict(config=config,statistics=statistics,local_mode=local_mode,wandb_url=run.url)
        (a.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        run.log({'mode_support':wandb.Image(str(a.output/'mode_support.png'))})
        for label,stats in statistics.items():
            eig=np.asarray(stats['covariance_eigenvalues_relative_to_target'])
            row=dict(mean_min_variance_ratio=float(eig[:,0].mean()),mean_max_variance_ratio=float(eig[:,1].mean()),
                mean_normalized_center_error=float(np.mean(stats['mean_error_in_sigma'])),
                mean_tail_fraction=float(np.mean(stats['outside_95pct_ellipse'])))
            run.log({label+'/'+key:value for key,value in row.items()})
            print(json.dumps(dict(method=label,**row)),flush=True)
        artifact=wandb.Artifact('gmm40-mode-diagnostic-'+run.id,type='diagnostic')
        for name in ['summary.json','mode_support.png']:
            artifact.add_file(str(a.output/name))
        run.log_artifact(artifact);run.finish()
    except BaseException:
        run.finish(exit_code=1)
        raise


if __name__ == '__main__':
    main()
