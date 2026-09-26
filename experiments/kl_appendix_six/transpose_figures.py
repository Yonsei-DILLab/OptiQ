"""Transpose the five-target density/score layouts without changing the data."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from .render import CASES, SEEDS, density_axis, limits, load, save, score_axis, sha, style


def draw(density, scores, out, kind):
    rows, cols = (2 if kind == 'density' else 3), 5
    panel_w = 24 * (.995 - .04) / (5 + 4 * .40)
    panel_h = (4.05 * 6) * (.975 - .035) / (6 + 5 * .72)
    gap_x, gap_y = .70, .68
    left, right, bottom, top = 1.15, .18, .70, .50
    width = left + cols*panel_w + (cols-1)*gap_x + right
    height = bottom + rows*panel_h + (rows-1)*gap_y + top
    fig = plt.figure(figsize=(width, height))
    for row in range(rows):
        for col, (case, _) in enumerate(CASES[:5]):
            x = left + col*(panel_w+gap_x)
            y = bottom + (rows-1-row)*(panel_h+gap_y)
            ax = fig.add_axes([x/width, y/height, panel_w/width, panel_h/height])
            letter = chr(97+row) if kind == 'density' else chr(99+row)
            panel = f'({col+1}{letter})'
            if kind == 'density':
                method = ['forward', 'reverse'][row]
                density_axis(ax, density[case,method], method, panel, False)
                if col == 0:
                    ax.set_ylabel(f'{method.capitalize()} KL\nDensity')
            else:
                data = scores[case,0]
                score_axis(ax, data, row, panel, limits(data,[128+row]), False)
                ax.set_title(panel+'  '+rf'$a={data["actions"][128+row]:.3f}$',
                             fontsize=17, pad=12)
                if col == 0:
                    ax.set_ylabel(f'{[10,50,90][row]}% policy quantile\n'+r'Actor score $\hat{s}_L(a)$')
            if row != rows-1:
                ax.set_xlabel('')
    save(fig,out,f'{kind}_{rows}x{cols}')


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--render-commit',required=True);a=p.parse_args()
    report=a.workspace/'reports/20260926_kl_appendix_six'
    out=report/'transposed';out.mkdir(exist_ok=True)
    style();density,scores,_,_,_=load(a.workspace/'studies/20260926_kl_appendix_six')
    for kind in ['density','score']:draw(density,scores,out,kind)
    provenance=dict(render_commit=a.render_commit,
        parent_provenance_sha256=sha(report/'PROVENANCE.json'),
        preceding_layout_provenance_sha256=sha(report/'split_compact_tight/PROVENANCE.json'),
        column_cases=[case for case,_ in CASES[:5]],
        paired_density_seeds={case:SEEDS[case] for case,_ in CASES[:5]},
        score_seed=0,density_rows=['Forward KL','Reverse KL'],score_rows=['10%','50%','90%'],
        note='Layout transpose only. Original data, axis limits, panel sizes, fonts, colors and shading retained.')
    (out/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (out/'README.md').write_text('''# Five-target figures, transposed

Columns retain the previous five-row target order: three Gaussian modes;
spike + ramp; spike + plateau + ramp; two offset Gaussian modes; unequal Gaussian
masses. Only layout changes; numerical inputs and per-panel axes are unchanged.

- Density 2 × 5: Forward KL in row 1, Reverse KL in row 2.
- Score 3 × 5: 10%, 50%, 90% policy quantile actions in rows 1, 2, 3.

Panel identifiers retain their original meanings: (1a), (1b) belong to target 1's
Forward and Reverse densities; (1c), (1d), (1e) are its score probes. The method
and quantile headings now appear once per row on the left. X-axis names appear
only on the bottom row. Panel sizes and tight gaps match the preceding layout.
All figures are exported as PNG, vector PDF and SVG.
''')
    (out/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in sorted(out.iterdir())
        if p.is_file() and p.name!='SHA256.json'},indent=2)+'\n')
    print(out)


if __name__=='__main__':main()
