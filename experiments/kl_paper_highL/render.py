"""Paper figures and auditable tables from completed high-L data; no training."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from ..kl_diverse_targets_1d.target import reference as gm_reference
from ..kl_nongmm_targets_1d.target import reference as ng_reference

CASES = [('t00_reference', 'Three Gaussian modes'),
         ('n00_spike_ramp', 'Spike + ramp')]
BLUE, ORANGE = '#1f77b4', '#e67e22'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig, out, name):
    fig.canvas.draw()
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(out / f'{name}.{ext}', dpi=300, bbox_inches='tight', pad_inches=.08)
    plt.close(fig)


def style():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
        'mathtext.fontset': 'dejavusans', 'font.size': 13,
        'axes.titlesize': 19, 'axes.labelsize': 17,
        'xtick.labelsize': 12, 'ytick.labelsize': 12,
        'axes.linewidth': .8, 'axes.edgecolor': '#666666',
        'axes.spines.top': True, 'axes.spines.right': True,
        'pdf.fonttype': 42, 'svg.fonttype': 'none',
    })


def density_figure(root, out, provenance):
    fig, axes = plt.subplots(1, 4, figsize=(18.8, 4.0))
    fig.subplots_adjust(left=.048, right=.995, bottom=.19, top=.84, wspace=.30)
    x = np.linspace(-10, 10, 16385)
    table = []
    for pair, (case, title) in enumerate(CASES):
        for col, method in enumerate(['forward', 'reverse']):
            ax = axes[2 * pair + col]
            values, metrics = [], []
            for seed in range(4):
                folder = root / 'input' / case / f'{method}_s{seed}'
                complete = read(folder / 'COMPLETE.json')
                run = read(folder / 'RUN.json')
                metric = read(folder / 'metrics_100000.json')
                assert complete['step'] == metric['step'] == 100000
                assert metric['sample_count'] == 2**20
                assert run['config']['n'] == run['config']['m'] == 128
                assert run['config']['batch'] == 32
                assert run['L'] == (2**20 if method == 'reverse' else 0)
                cp = folder / 'checkpoint.msgpack'
                sample = folder / 'samples_100000.npz'
                with np.load(sample) as data:
                    edges = data['edges'].copy()
                    mass = data['histogram_mass'].copy()
                    assert len(data['actions']) == 2**20 and len(mass) == 512
                    assert np.allclose(np.histogram(data['actions'], edges)[0] / 2**20, mass)
                    assert abs(mass.sum() - 1) < 1e-8
                    assert abs(.5 * np.abs(mass - data['target_mass']).sum() - metric['histogram_TV']) < 1e-8
                    values.append(mass / np.diff(edges))
                other = read(root / 'input' / case / f'{"reverse" if method == "forward" else "forward"}_s{seed}' / 'RUN.json')
                assert run['initial_parameter_sha256'] == other['initial_parameter_sha256']
                entry = dict(case=case, method=method, seed=seed,
                             TV=metric['histogram_TV'], W1=metric['wasserstein_1'],
                             missing_modes=metric['missing_modes'],
                             mode_mass=metric['mode_mass'], sigma_mean=metric['sigma_mean'])
                table.append(entry)
                metrics.append(entry)
                provenance['density_runs'].append(dict(
                    case=case, method=method, seed=seed, training_run=run,
                    checkpoint_sha256=sha(cp), samples_sha256=sha(sample),
                    sample_path=str(sample), metrics=metric))
            target = (gm_reference if case == 't00_reference' else ng_reference)(run['config'], x)[0]
            color = [BLUE, ORANGE][col]
            for density in values:
                ax.stairs(density, edges, color=color, alpha=.18, lw=.65)
            mean = np.mean(values, axis=0)
            ax.stairs(mean, edges, fill=True, color=color, alpha=.20, linewidth=0)
            ax.stairs(mean, edges, color=color, lw=1.55)
            ax.plot(x, target, color='black', ls='--', lw=1.45)
            ax.set(xlim=(-10, 10), ylim=(0, None), xlabel=r'$a$')
            ax.set_xticks([-10, -5, 0, 5, 10])
            ax.tick_params(direction='out', length=4, width=.8)
            if pair == col == 0:
                ax.set_ylabel('Density')
            ax.set_title(f'({"abcd"[2 * pair + col]}) {method.capitalize()} KL', pad=12)
            provenance['metrics'][f'{case}/{method}'] = {
                k: dict(mean=float(np.mean([m[k] for m in metrics])),
                        sd=float(np.std([m[k] for m in metrics], ddof=1)),
                        seeds=[m[k] for m in metrics]) for k in ['TV', 'W1', 'missing_modes']}
    save(fig, out, 'forward_vs_reverse_1x4_L20')
    with (out / 'density_metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)


def load_scores(root, provenance):
    scores = {}
    for case, title in CASES:
        for seed in range(4):
            folder = root / 'runtime/results' / case / f'reverse_s{seed}'
            complete, run, summary = [read(folder / name) for name in ['COMPLETE.json', 'RUN.json', 'SUMMARY.json']]
            assert complete['checkpoint_sha256'] == sha(root / 'input' / case / f'reverse_s{seed}' / 'checkpoint.msgpack')
            with np.load(folder / 'scores.npz') as d:
                data = {k: d[k].copy() for k in d.files}
            assert data['scores'].shape == (16, 18, 140)
            assert data['reference_scores'].shape == (4, 140)
            assert np.isfinite(data['scores']).all()
            assert np.array_equal(data['Ls'], 2**np.arange(7, 25))
            data['summary'] = summary
            scores[case, seed] = data
            provenance['score_runs'].append(dict(
                case=case, seed=seed, run=run, complete=complete, summary=summary,
                raw_sha256=sha(folder / 'scores.npz'), validation=read(folder / 'VALIDATION.json')))
    return scores


def limits(data, indices):
    values = data['scores'][:, :, indices]
    lower, upper = np.quantile(values, [.1, .9], axis=0)
    mean = values.mean(0)
    ref = data['reference_scores'][:, indices].mean(0)
    lo = min(float(lower.min()), float(mean.min()), float(ref.min()))
    hi = max(float(upper.max()), float(mean.max()), float(ref.max()))
    pad = max((hi - lo) * .12, .002)
    return lo - pad, hi + pad


def score_axis(ax, data, j, title, ylim, ylabel=False):
    k = 128 + j
    s = data['scores'][:, :, k]
    mean, ref = s.mean(0), data['reference_scores'][:, k].mean()
    lo, hi = np.quantile(s, [.1, .9], axis=0)
    ax.fill_between(data['Ls'], lo, hi, color=BLUE, alpha=.25, lw=0)
    ax.plot(data['Ls'], mean, '-o', color=BLUE, lw=2, ms=3.8, zorder=3)
    ax.axhline(ref, color='#333333', ls=':', lw=1.6)
    ax.axvline(2**20, color='#555555', ls='--', lw=1.6)
    ax.set_xscale('log', base=2)
    powers = [7, 10, 14, 17, 20, 24]
    ax.set_xticks([2**p for p in powers], [rf'$2^{{{p}}}$' for p in powers])
    ax.set(xlim=(2**7, 2**24), ylim=ylim, xlabel=r'Density-bank size $L$')
    ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    ax.tick_params(direction='out', length=4, width=.8)
    ax.set_title(title + '\n' + rf'$a={data["actions"][k]:.3f}$', fontsize=17, pad=10)
    if ylabel:
        ax.set_ylabel(r'Actor score $\hat{s}_L(a)$')


def score_legend(fig):
    # No gray reference band and no MC-range legend entry, per user feedback.
    handles = [Line2D([], [], color=BLUE, marker='o', ms=4, lw=2),
               Line2D([], [], color='#333333', ls=':', lw=1.6),
               Line2D([], [], color='#555555', ls='--', lw=1.6)]
    labels = ['MC mean (16 repetitions)', r'Independent $2^{24}$ reference', r'Training $L=2^{20}$']
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.52, .01),
               ncol=3, frameon=False, fontsize=12)


def score_figures(scores, out):
    for zoom in [True, False]:
        suffix = 'zoom' if zoom else 'common_y'
        for case, title in CASES:
            data = scores[case, 0]
            fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
            fig.subplots_adjust(left=.075, right=.99, bottom=.24, top=.80, wspace=.32)
            for j, ax in enumerate(axes):
                lim = limits(data, [128 + j] if zoom else [128, 129, 130])
                score_axis(ax, data, j, f'({"abc"[j]}) {[10,50,90][j]}% policy quantile', lim, j == 0)
            score_legend(fig)
            slug = 'gaussian' if case == 't00_reference' else 'spike_ramp'
            save(fig, out, f'score_convergence_{slug}_{suffix}')
        fig, axes = plt.subplots(2, 3, figsize=(14.3, 8.1))
        fig.subplots_adjust(left=.075, right=.99, bottom=.13, top=.91, hspace=.72, wspace=.33)
        for row, (case, title) in enumerate(CASES):
            data = scores[case, 0]
            for j, ax in enumerate(axes[row]):
                lim = limits(data, [128 + j] if zoom else [128, 129, 130])
                score_axis(ax, data, j, f'({"abcdef"[3*row+j]}) {[10,50,90][j]}% policy quantile', lim, j == 0)
        score_legend(fig)
        save(fig, out, f'score_convergence_{suffix}')
    for case, title in CASES:
        fig, axes = plt.subplots(4, 3, figsize=(14.3, 15))
        fig.subplots_adjust(left=.08, right=.985, top=.955, bottom=.065, hspace=.70, wspace=.34)
        for seed in range(4):
            for j, ax in enumerate(axes[seed]):
                data = scores[case, seed]
                score_axis(ax, data, j, f'Seed {seed} | {[10,50,90][j]}% quantile', limits(data, [128+j]), j == 0)
        score_legend(fig)
        slug = 'gaussian' if case == 't00_reference' else 'spike_ramp'
        save(fig, out, f'score_all_seeds_{slug}')


def write_tables(scores, out, provenance):
    rows, quantiles = [], []
    for case, title in CASES:
        for seed in range(4):
            data = scores[case, seed]
            for rec in data['summary']['results']:
                rows.append(dict(case=case, seed=seed, **rec))
            idx = int(np.where(data['Ls'] == 2**20)[0][0])
            for j in range(3):
                k = 128+j
                quantiles.append(dict(case=case, seed=seed, quantile=[.1,.5,.9][j],
                    action=float(data['actions'][k]), score_L20_mean=float(data['scores'][:,idx,k].mean()),
                    score_L20_sd=float(data['scores'][:,idx,k].std(ddof=1)),
                    reference=float(data['reference_scores'][:,k].mean()),
                    reference_se=float(data['reference_scores'][:,k].std(ddof=1)/2)))
    for name, table in [('score_metrics', rows), ('score_quantiles', quantiles)]:
        with (out / f'{name}.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    provenance['score_quantiles'] = quantiles


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', required=True, type=Path)
    p.add_argument('--render-commit', required=True)
    args = p.parse_args()
    root = args.workspace / 'studies/20260926_kl_paper_highL'
    out = args.workspace / 'reports/20260926_kl_paper_highL'
    out.mkdir(parents=True, exist_ok=True)
    style()
    provenance = dict(render_commit=args.render_commit, measurement_commit=read(root/'SOURCE_MANIFEST.json')['commit'],
        training_L=2**20, training_updates=100000, N=128, M=128, batch=32,
        sample_count=2**20, bins=512, seeds=[0,1,2,3], score_main_seed=0,
        score_shading='10th–90th percentiles of 16 MC repetitions; not a confidence interval',
        density_runs=[], score_runs=[], metrics={})
    density_figure(root, out, provenance)
    scores = load_scores(root, provenance)
    score_figures(scores, out)
    write_tables(scores, out, provenance)
    (out / 'PROVENANCE.json').write_text(json.dumps(provenance, indent=2) + '\n')
    from .report import write_report
    write_report(out, provenance)
    print(out)


if __name__ == '__main__':
    main()
