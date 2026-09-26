"""One-row paper layout for the two requested L1024 target comparisons."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .figure_three import CASES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', required=True, type=Path)
    parser.add_argument('--tight', action='store_true')
    parser.add_argument('--render-commit')
    args = parser.parse_args()
    out = args.workspace / 'reports/20260925_kl_paper_row'
    if args.tight:
        out = out / 'tight'
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
        'mathtext.fontset': 'dejavusans', 'font.size': 13,
        'axes.titlesize': 19, 'axes.labelsize': 17,
        'xtick.labelsize': 12, 'ytick.labelsize': 12,
        'axes.linewidth': .8, 'axes.edgecolor': '#666666',
        'axes.spines.top': True, 'axes.spines.right': True,
        'pdf.fonttype': 42, 'svg.fonttype': 'none',
    })
    fig, axes = plt.subplots(1, 4, figsize=(18.8, 4.0))
    fig.subplots_adjust(left=.048, right=.995, bottom=.19, top=.84, wspace=.30)
    if args.tight:
        panel_width = 18.8 * (.995 - .048) / (4 + 3 * .30)
        gap, left, right = .50, 18.8 * .048, 18.8 * .005
        width = left + 4 * panel_width + 3 * gap + right
        fig.set_size_inches(width, 4.0)
        for i, ax in enumerate(axes):
            ax.set_position([(left + i * (panel_width + gap)) / width,
                             .19, panel_width / width, .84 - .19])
    colors = ['#1f77b4', '#e67e22']
    x = np.linspace(-10, 10, 16385)
    provenance = {'reverse_L': 1024, 'updates': 100000, 'N': 128, 'M': 128,
                  'batch': 32, 'seeds': [0,1,2,3], 'bins': 512,
                  'samples_per_seed': 262144, 'runs': [], 'metrics': {}}
    if args.render_commit:
        provenance['figure_source_commit'] = args.render_commit
    provenance['tight_spacing'] = args.tight
    if args.tight:
        provenance['horizontal_gap_inches'] = .50
        provenance['original_png_sha256'] = hashlib.sha256((out.parent/'forward_vs_reverse_1x4.png').read_bytes()).hexdigest()
    for pair, (ident, _, dirname, reference) in enumerate(CASES[:2]):
        root = args.workspace / 'studies' / dirname
        cfg = json.loads((root/'config.json').read_text())
        case = next(c for c in cfg['cases'] if c['id'] == ident)
        target = reference(dict(cfg, **case), x)[0]
        for method_index, method in enumerate(['forward', 'reverse']):
            ax = axes[2*pair+method_index]
            values, tvs = [], []
            for seed in range(4):
                stage = 'screen' if seed < 2 else 'validate_seeds'
                folder = root/'runtime'/stage/ident/f'{method}_s{seed}'
                assert json.loads((folder/'COMPLETE.json').read_text())['step'] == 100000
                metric = json.loads((folder/'metrics_100000.json').read_text())
                assert metric['sample_count'] == 262144
                sample = folder/'samples_100000.npz'
                with np.load(sample) as z:
                    edges = z['edges'].copy()
                    mass = z['histogram_mass'].copy()
                    assert len(mass) == 512 and abs(mass.sum()-1) < 1e-7
                    assert abs(.5*np.abs(mass-z['target_mass']).sum()-metric['histogram_TV']) < 1e-8
                    values.append(mass/np.diff(edges))
                tvs.append(metric['histogram_TV'])
                provenance['runs'].append({'case': ident, 'method': method, 'seed': seed,
                    'file': str(sample), 'sha256': hashlib.sha256(sample.read_bytes()).hexdigest()})
            mean = np.mean(values, axis=0)
            color = colors[method_index]
            for density in values:
                ax.stairs(density, edges, color=color, alpha=.18, lw=.65)
            ax.stairs(mean, edges, fill=True, color=color, alpha=.20, linewidth=0)
            ax.stairs(mean, edges, color=color, lw=1.55)
            ax.plot(x, target, color='black', ls='--', lw=1.45)
            ax.set(xlim=(-10,10), ylim=(0,None), xlabel=r'$a$')
            ax.set_xticks([-10,-5,0,5,10])
            ax.tick_params(direction='out', length=4, width=.8)
            if pair == 0 and method_index == 0:
                ax.set_ylabel('Density')
            ax.set_title(f'({"abcd"[2*pair+method_index]}) {method.capitalize()} KL', pad=12)
            provenance['metrics'][f'{ident}/{method}'] = {'TV_seeds': tvs, 'mean_TV': float(np.mean(tvs))}
    fig.canvas.draw()
    if args.tight:
        original = json.loads((out.parent/'PROVENANCE.json').read_text())
        assert provenance['runs'] == original['runs']
        assert provenance['metrics'] == original['metrics']
    renderer = fig.canvas.get_renderer()
    titles = [ax.title.get_window_extent(renderer) for ax in axes]
    assert all(left.x1 < right.x0 for left, right in zip(titles, titles[1:]))
    for ext in ['png','pdf','svg']:
        fig.savefig(out/f'forward_vs_reverse_1x4.{ext}', dpi=300, bbox_inches='tight', pad_inches=.08)
    plt.close(fig)
    (out/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (out/'caption.md').write_text('''# Forward KL vs. Reverse KL — one row, four panels

Three Gaussian modes: (a) Forward KL, (b) Reverse KL. Spike + ramp: (c) Forward KL, (d) Reverse KL.

All panels use the completed L=2^10 screening experiment: 100K updates, N=M=128, batch 32, seeds 0–3. Forward has no auxiliary density bank; L applies to Reverse. These are not L=2^20 confirmation results.

Black dashed curves are the exact target. The learned densities are 512-bin sample histograms from 262,144 actions per seed, averaged over four seeds, without smoothing. Light fills show the learned density; thin curves show individual seeds. Vertical scales are independent, as in the preceding figure. Font: DejaVu Sans (sans-serif); all four axes use a boxed frame. Environment headings, legend, and TV labels are omitted; only the first panel has a Density label. Numerical data and target definitions are unchanged.

Mean per-seed TV: three Gaussian modes, Forward 0.0387 / Reverse 0.6666; spike + ramp, Forward 0.0432 / Reverse 0.1977. Exact values and source hashes are recorded in PROVENANCE.json. The examples were selected during screening and do not establish universal performance ordering.
''')
    if args.tight:
        with (out/'caption.md').open('a') as f:
            f.write('\nPanel gaps reduced to 0.50 inches; original plotting-area sizes, fonts, axis limits and data are retained.\n')
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(out.iterdir()) if p.is_file() and p.name != 'SHA256.json'}
    (out/'SHA256.json').write_text(json.dumps(hashes, indent=2)+'\n')
    print(out)


if __name__ == '__main__':
    main()
