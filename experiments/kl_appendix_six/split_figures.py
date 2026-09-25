"""Separate density/score figures with shared headers and row-panel labels."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from .render import CASES, SEEDS, density_axis, limits, load, save, score_axis, sha, style


def render_split(density, scores, out, rows, kind, compact_labels=False, tight=False):
    # Use the original combined figure's physical panel sizes and positions.
    # bbox_inches='tight' crops the unused rows/columns when exporting.
    fig, axes = plt.subplots(6, 5, figsize=(24, 4.05 * 6), squeeze=False)
    fig.subplots_adjust(left=.04, right=.995, bottom=.035, top=.975,
                        hspace=.72, wspace=.40)
    columns = [0, 1] if kind == 'density' else [2, 3, 4]
    if tight:
        # Keep each plotting area exactly as large as in the original figure;
        # shrink the canvas by reducing the gaps, rather than shrinking text.
        panel_width = 24 * (.995 - .04) / (5 + 4 * .40)
        panel_height = (4.05 * 6) * (.975 - .035) / (6 + 5 * .72)
        gap_x, gap_y = .70, .68
        left, right, bottom, top = .90, .18, .70, .98
        width = left + len(columns) * panel_width + (len(columns) - 1) * gap_x + right
        height = bottom + rows * panel_height + (rows - 1) * gap_y + top
        fig.set_size_inches(width, height)
    for row in range(6):
        for col in range(5):
            ax = axes[row, col]
            if row >= rows or col not in columns:
                ax.remove()
                continue
            if tight:
                x = left + columns.index(col) * (panel_width + gap_x)
                y = bottom + (rows - 1 - row) * (panel_height + gap_y)
                ax.set_position([x / width, y / height, panel_width / width, panel_height / height])
            case, _ = CASES[row]
            separator = '' if compact_labels else '-'
            panel = f'({row + 1}{separator}{chr(97 + col)})'
            if kind == 'density':
                method = ['forward', 'reverse'][col]
                density_axis(ax, density[case, method], method,
                             panel, col == 0)
                header = f'{method.capitalize()} KL'
            else:
                j = col - 2
                data = scores[case, 0]
                score_axis(ax, data, j, panel,
                           limits(data, [128 + j]), j == 0)
                ax.set_title(panel + '  ' + rf'$a={data["actions"][128 + j]:.3f}$',
                             fontsize=17, pad=12)
                header = f'{[10, 50, 90][j]}% policy quantile'
            if row == 0:
                ax.annotate(header, xy=(.5, 1), xycoords='axes fraction',
                            xytext=(0, 44 if tight else 52), textcoords='offset points',
                            ha='center', va='bottom', fontsize=19,
                            annotation_clip=False)
            if row != rows - 1:
                ax.set_xlabel('')
    save(fig, out, f'{kind}_{rows}x{len(columns)}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--render-commit', required=True)
    parser.add_argument('--compact-labels', action='store_true')
    parser.add_argument('--tight', action='store_true')
    args = parser.parse_args()
    report = args.workspace / 'reports/20260926_kl_appendix_six'
    out = report / (('split_compact' if args.compact_labels else 'split') + ('_tight' if args.tight else ''))
    out.mkdir(exist_ok=True)
    style()
    density, scores, _, _, _ = load(args.workspace / 'studies/20260926_kl_appendix_six')
    for rows in [5, 6]:
        for kind in ['density', 'score']:
            render_split(density, scores, out, rows, kind, args.compact_labels, args.tight)
    provenance = {
        'render_commit': args.render_commit,
        'parent_provenance_sha256': sha(report / 'PROVENANCE.json'),
        'parent_combined_png_sha256': sha(report / 'density_and_score_six.png'),
        'five_row_cases': [case for case, _ in CASES[:5]],
        'six_row_cases': [case for case, _ in CASES],
        'paired_density_seeds': SEEDS,
        'score_seed': 0,
        'panel_label_format': '(1a)' if args.compact_labels else '(1-a)',
        'tight_spacing': args.tight,
        'gap_inches': {'horizontal': .70, 'vertical': .68} if args.tight else None,
        'change': 'Presentation only: shared column headers, bottom-row x labels, numbered row-panel labels; original data and axis limits retained.',
        'five_row_selection': 'First five original rows; six-row versions preserve all original environments.',
    }
    (out / 'PROVENANCE.json').write_text(json.dumps(provenance, indent=2) + '\n')
    readme = '''# Density and score: separate figures

The requested 5 × 2 density and 5 × 3 score figures contain the first five rows
of the original 6 × 5 figure, in unchanged order. The sixth environment is not
included in these five-row layouts; the additional 6 × 2 and 6 × 3 exports
preserve all six environments. PNG, vector PDF and SVG versions are provided.

Only presentation changes. Data, seeds, histogram bins, score estimates, physical
panel sizes, fonts, colors, shading and axis limits are unchanged. Column headings
appear only above the first row, and x-axis labels only below the last row.
Panel labels are (1-a), (1-b), (2-a), (2-b), etc. for density and (1-c),
(1-d), (1-e), (2-c), etc. for score. Each score panel retains its fixed action.

See the parent report and PROVENANCE.json for the full experimental details.
'''
    if args.compact_labels:
        import re
        readme = re.sub(r'\((\d+)-([a-e])\)', r'(\1\2)', readme)
    if args.tight:
        readme += '\nTight layout: horizontal gaps are 0.70 inches and vertical gaps are 0.68 inches. Panel sizes and text sizes are preserved; the overall canvas is smaller.\n'
    (out / 'README.md').write_text(readme)
    hashes = {p.name: sha(p) for p in sorted(out.iterdir())
              if p.is_file() and p.name != 'SHA256.json'}
    (out / 'SHA256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    print(out)


if __name__ == '__main__':
    main()
