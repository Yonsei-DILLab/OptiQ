"""One evaluator for final g(z) samples and the authors' released baselines."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import torch
import wandb

from optiq_dime.runtime import load_environment, provenance, ROOT
from .metrics import evaluate_sample_tensor
from .sampler import make_target
from .train import write_json

PAPER = {'True': -6.85, 'FAB': -10.74, 'iDEM': -8.33, 'DiKL': -7.21}


def energy_tvd(generated, reference, bins=200):
    # Same reference-defined bins/renormalization as OptiC/DiKL metric.py.
    ref_hist, edges = np.histogram(reference, bins=bins)
    gen_hist, _ = np.histogram(generated, bins=edges)
    tvd = 1. if gen_hist.sum() == 0 else .5 * np.abs(ref_hist/ref_hist.sum() - gen_hist/gen_hist.sum()).sum()
    # The released convention drops out-of-range mass. Also expose that mass
    # and the TVD with two explicit tail bins so failures cannot be hidden.
    gen_counts = np.r_[np.sum(generated < edges[0]), gen_hist, np.sum(generated > edges[-1])]
    ref_counts = np.r_[0, ref_hist, 0]
    with_tails = .5 * np.abs(gen_counts/len(generated) - ref_counts/len(reference)).sum()
    return float(tvd), float(with_tails), float(1 - gen_hist.sum()/len(generated))


def exact_w2(samples, reference):
    if samples.shape != reference.shape or samples.ndim != 2 or len(samples) < 2:
        raise ValueError('Exact assignment requires equal-sized sample matrices')
    # Equal uniform mass reduces exact empirical OT to a linear assignment.
    # Preserve the source evaluator's float32 torch.cdist cost convention.
    costs = torch.cdist(samples.float(), reference.float()).square().numpy()
    rows, cols = linear_sum_assignment(costs)
    return float(np.sqrt(costs[rows, cols].astype(np.float64).mean()))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--runs-root', type=Path, default=ROOT/'outputs/gmm40/runs')
    parser.add_argument('--baselines-root', type=Path, default=ROOT/'outputs/gmm40/reference')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--baselines-only', action='store_true')
    parser.add_argument('--group', default='gmm40-common-evaluation')
    parser.add_argument('--baseline-report', type=Path, help='Reuse a verified prior common baseline evaluation')
    parser.add_argument('--sample-count', type=int, default=10000)
    parser.add_argument('--reference-seed', type=int, default=20260821)
    parser.add_argument('--transport-count', type=int, default=2000)
    parser.add_argument('--transport-repeats', type=int, default=10)
    args = parser.parse_args()
    if args.sample_count < args.transport_count or args.transport_count < 2 or args.transport_repeats < 1:
        parser.error('Invalid evaluation sample counts')
    load_environment()
    torch.set_num_threads(1)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    run = wandb.init(project=os.environ.get('WANDB_PROJECT', 'optiq_dime_no_anchor'),
        entity=os.environ.get('WANDB_ENTITY'), mode='online', job_type='evaluation',
        name='gmm40-common-eval-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'),
        group=args.group, dir=str(output),
        config={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()})
    try:
        runtime = provenance()
        run.config.update({'runtime': runtime})
        target = make_target()
        with torch.random.fork_rng():
            torch.manual_seed(args.reference_seed)
            reference = target.sample((args.sample_count,))
            torch.manual_seed(args.reference_seed+2)
            independent_reference = target.sample((args.sample_count,))
        sample_sets = {'GT independent': independent_reference}
        sources = {'GT independent': {'seed': args.reference_seed+2, 'type': 'independent target draw for metric floor'}}
        for method, name in [('DiKL', 'dikl_samples.pt'), ('iDEM', 'idem_samples.pt'), ('FAB', 'fab_samples.pt'), ('R-KL', 'rkl_samples.pt')]:
            path = args.baselines_root/name
            samples = torch.load(path, map_location='cpu', weights_only=True)
            samples = torch.as_tensor(samples).detach().float()
            if samples.ndim != 2 or samples.shape[1] != 2 or not torch.isfinite(samples).all():
                raise ValueError('Invalid baseline sample matrix: '+str(path))
            full_mean = float(target.log_prob(samples).mean())
            if len(samples) < args.sample_count:
                raise ValueError('Insufficient baseline samples: '+str(path))
            if len(samples) > args.sample_count:
                selection = np.random.default_rng(args.reference_seed+3).choice(len(samples), args.sample_count, replace=False)
                selected = samples[selection]
            else:
                selected = samples
            sample_sets[method] = selected
            sources[method] = {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'released_count': len(samples), 'released_full_mean_log_prob': full_mean,
                'paper_table_1_mean_log_prob': PAPER.get(method), 'common_evaluation_count': len(selected)}
        if not args.baselines_only:
            completed = sorted(args.runs_root.glob('*/completed.json'))
            seen = set()
            for file in completed:
                config = json.loads((file.parent/'config.json').read_text())
                done = json.loads(file.read_text())
                seed = config['seed']
                if seed in seen:
                    raise ValueError('Multiple completed runs for seed '+str(seed)+'; use an explicit run root')
                if not done['finished'] or done['updates'] != 30000:
                    raise ValueError('Expected the prespecified final 30k checkpoint')
                seen.add(seed)
                path = file.parent/'samples_final.npy'
                samples = torch.from_numpy(np.load(path)).float()
                if len(samples) != args.sample_count or not torch.isfinite(samples).all():
                    raise ValueError('Generator sample count/finiteness mismatch')
                method = 'OptiQ seed '+str(seed)
                sample_sets[method] = samples
                sources[method] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'config': config, 'completion': done}
            if seen != {0, 1, 2}:
                raise ValueError('Expected completed seeds 0, 1, 2; found '+str(seen))
        records, bootstrap_rows = [], []
        cached = None
        if args.baseline_report:
            cached = json.loads(args.baseline_report.read_text())
            expected = {'sample_count': args.sample_count, 'reference_seed': args.reference_seed,
                'exact_w2_subsample': args.transport_count, 'exact_w2_repeats': args.transport_repeats}
            if any(cached['protocol'][k] != v for k,v in expected.items()):
                raise ValueError('Cached baseline evaluation protocol differs')
            cache_dir = args.baseline_report.parent
            hashes = json.loads((cache_dir/'analysis_metadata.json').read_text())['evaluation_source_sha256']
            for source in ['benchmarks/gmm40/metrics.py', 'benchmarks/gmm40/target_torch.py']:
                if hashes[source] != hashlib.sha256((ROOT/source).read_bytes()).hexdigest():
                    raise ValueError('Cached target/evaluator code differs: '+source)
            old_tree = ast.parse((cache_dir/'evaluation_source/benchmarks/gmm40/evaluate.py').read_text())
            new_tree = ast.parse(Path(__file__).read_text())
            for function in ['energy_tvd', 'exact_w2']:
                old = next(n for n in old_tree.body if isinstance(n,ast.FunctionDef) and n.name==function)
                new = next(n for n in new_tree.body if isinstance(n,ast.FunctionDef) and n.name==function)
                if ast.dump(old) != ast.dump(new):
                    raise ValueError('Cached transport metric implementation differs')
            cached_rows = {r['method']:r for r in cached['records']}
            cached_repeats = pd.read_csv(cache_dir/'transport_repeats.csv')
        reference_energy = -target.log_prob(reference).numpy()
        for method, samples in sample_sets.items():
            if cached is not None and not method.startswith('OptiQ'):
                if method != 'GT independent' and cached['sources'][method]['sha256'] != sources[method]['sha256']:
                    raise ValueError('Cached released-sample identity differs: '+method)
                record = cached_rows[method].copy()
                records.append(record)
                bootstrap_rows.extend(cached_repeats[cached_repeats.method==method].to_dict('records'))
                run.log({method+'/'+k:v for k,v in record.items() if k != 'method'})
                print('Verified cached baseline: '+method, flush=True)
                continue
            metrics = evaluate_sample_tensor(samples, target, args.reference_seed)
            record = {'method': method, **{k.removeprefix('eval/'): v for k,v in metrics.items()}}
            energies = -target.log_prob(samples).numpy()
            tvd, tails, overflow = energy_tvd(energies, reference_energy)
            record.update(energy_tvd=tvd, energy_tvd_with_tails=tails, energy_outside_reference_range=overflow)
            rng = np.random.default_rng(args.reference_seed+4)
            w2_values = []
            for repeat in range(args.transport_repeats):
                si = rng.choice(len(samples), args.transport_count, replace=False)
                ri = rng.choice(len(reference), args.transport_count, replace=False)
                w2 = exact_w2(samples[si], reference[ri])
                w2_values.append(w2)
                bootstrap_rows.append({'method': method, 'repeat': repeat, 'sample_w2': w2})
            record.update(sample_w2=float(np.mean(w2_values)), sample_w2_resampling_sd=float(np.std(w2_values)))
            records.append(record)
            print(json.dumps(record), flush=True)
            run.log({method+'/'+k: v for k,v in record.items() if k != 'method'})
        frame = pd.DataFrame(records)
        frame.to_csv(output/'comparison.csv', index=False)
        pd.DataFrame(bootstrap_rows).to_csv(output/'transport_repeats.csv', index=False)
        summary = {'wandb_url': run.url, 'runtime': runtime, 'records': records, 'sources': sources,
            'baseline_evaluation_reused_from': str(args.baseline_report) if cached is not None else None,
            'baseline_evaluation_wandb_url': cached['wandb_url'] if cached is not None else None,
            'paper_table_1': PAPER, 'paper_url': 'https://arxiv.org/html/2410.12456v2#S5.T1',
            'protocol': {'sample_count': args.sample_count, 'reference_seed': args.reference_seed,
                'exact_w2_subsample': args.transport_count, 'exact_w2_repeats': args.transport_repeats,
                'note': 'GMM Table 1 reports log p only. Transport/coverage metrics are our common re-evaluation, not published DiKL GMM table entries. W2 repeat SD is resampling variation, not network seed SD.'}}
        ours = frame[frame.method.str.startswith('OptiQ')]
        if len(ours):
            columns = [c for c in frame.columns if c != 'method']
            summary['optiq_three_seed_mean'] = ours[columns].mean().to_dict()
            summary['optiq_three_seed_sd'] = ours[columns].std(ddof=1).to_dict()
        write_json(output/'summary.json', summary)
        panels = [('Ground truth', reference)] + [(k,v) for k,v in sample_sets.items() if k != 'GT independent']
        fig, axes = plt.subplots(2, 4, figsize=(15, 7.8), sharex=True, sharey=True)
        extent = max(50., max(float(samples.abs().max()) for _,samples in panels)+2.)
        for ax, (label, samples) in zip(axes.flat, panels):
            points = samples.numpy()
            ax.scatter(points[:3000,0], points[:3000,1], s=2, alpha=.35, rasterized=True)
            ax.scatter(target.locs[:,0], target.locs[:,1], marker='x', s=16, color='#B34747', linewidths=.7)
            ax.set_title(label)
            ax.set_xlim(-extent,extent); ax.set_ylim(-extent,extent); ax.set_aspect('equal')
            ax.set_xticks([-40,0,40]); ax.set_yticks([-40,0,40])
        for ax in list(axes.flat)[len(panels):]:
            ax.set_visible(False)
        fig.suptitle('GMM40: direct generator samples vs released DiKL baselines')
        if len(ours):
            c = sources['OptiQ seed 0']['config']
            geometry = f"{c['num_policy_samples']}x{c['num_policy_samples']*c['proposals_per_policy_sample']}"
            coordinates = 'native coordinates' if c['coordinate_scale'] == 1. else f"coordinate scale={c['coordinate_scale']:g}"
            caption = f"OptiQ: T={c['temperature']} fixed, beta={c['density_beta']}, {geometry} OT, sigma={c['proposal_std']*c['coordinate_scale']:g}, anchor={c.get('include_anchor', False)}, unbounded={c.get('unbounded_actions', False)}, {c['updates']:,} updates/seed, {coordinates}."
        else:
            caption = 'Released GMM40 baselines evaluated in original coordinates.'
        fig.text(.06,.015,caption+f' Red crosses: true component centers.\nPanels show up to 3,000 of the {args.sample_count:,} evaluated samples; metrics use the original GMM target.',fontsize=9)
        fig.tight_layout(rect=(0,.065,1,.96))
        for extension in ['png','pdf']:
            fig.savefig(output/('samples.'+extension),dpi=170)
        plt.close(fig)
        run.log({'sample_comparison': wandb.Image(str(output/'samples.png'))})
        run.summary.update({'completed': True, 'report': str(output)})
        artifact = wandb.Artifact('gmm40-comparison-'+run.id, type='evaluation')
        for file in sorted(output.iterdir()):
            if file.is_file():
                artifact.add_file(str(file), name='report/'+file.name)
        run.log_artifact(artifact)
        run.finish()
        print('Report: '+str(output), flush=True)
    except BaseException:
        run.finish(exit_code=1)
        raise


if __name__ == '__main__':
    main()
