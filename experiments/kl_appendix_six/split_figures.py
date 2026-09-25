"""Separate the original six-row density/score figure without changing panels."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from .render import CASES, SEEDS, density_axis, limits, load, save, score_axis, sha, style


def render_split(density, scores, out, rows, kind):
    # Use the original combined figure's physical panel sizes and positions.
    # bbox_inches='tight' crops the unused rows/columns when exporting.
    fig, axes = plt.subplots(6, 5, figsize=(24, 4.05 * 6), squeeze=False)
    fig.subplots_adjust(left=.04, right=.995, bottom=.035, top=.975,
                        hspace=.72, wspace=.40)
    columns = [0, 1] if kind == 'density' else [2, 3, 4]
    for row in range(6):
        for col in range(5):
            ax = axes[row, col]
            if row >= rows or col not in columns:
                ax.remove()
                continue
            case, _ = CASES[row]
            if kind == 'density':
                method = ['forward', 'reverse'][col]
                density_axis(ax, density[case, method], method,
                             f'({chr(97 + col)}) {method.capitalize()} KL', col == 0)
            else:
                j = col - 2
                data = scores[case, 0]
                score_axis(ax, data, j,
                           f'({chr(99 + j)}) {[10, 50, 90][j]}% policy quantile',
                           limits(data, [128 + j]), j == 0)
    save(fig, out, f'{kind}_{rows}x{len(columns)}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--render-commit', required=True)
    args = parser.parse_args()
    report = args.workspace / 'reports/20260926_kl_appendix_six'
    out = report / 'split'
    out.mkdir(exist_ok=True)
    style()
    density, scores, _, _, _ = load(args.workspace / 'studies/20260926_kl_appendix_six')
    for rows in [5, 6]:
        for kind in ['density', 'score']:
            render_split(density, scores, out, rows, kind)
    provenance = {
        'render_commit': args.render_commit,
        'parent_provenance_sha256': sha(report / 'PROVENANCE.json'),
        'parent_combined_png_sha256': sha(report / 'density_and_score_six.png'),
        'five_row_cases': [case for case, _ in CASES[:5]],
        'six_row_cases': [case for case, _ in CASES],
        'paired_density_seeds': SEEDS,
        'score_seed': 0,
        'change': 'Layout only; original panel data, styling, axis limits and labels retained.',
        'five_row_selection': 'First five original rows; six-row versions preserve all original environments.',
    }
    (out / 'PROVENANCE.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (out / 'README.md').write_text('''# Density and score: separate figures

The requested 5 × 2 density and 5 × 3 score figures contain the first five rows
of the original 6 × 5 figure, in unchanged order. The sixth environment is not
included in these five-row layouts; the additional 6 × 2 and 6 × 3 exports
preserve all six environments. PNG, vector PDF and SVG versions are provided.

Only layout changes. Data, seeds, histogram bins, score estimates, physical
panel sizes, fonts, colors, shading, axis limits and panel labels are unchanged.
Density panels retain labels (a, b); score panels retain (c, d, e).

See the parent report and PROVENANCE.json for the full experimental details.
''')
    hashes = {p.name: sha(p) for p in sorted(out.iterdir())
              if p.is_file() and p.name != 'SHA256.json'}
    (out / 'SHA256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    print(out)


if __name__ == '__main__':
    main()
