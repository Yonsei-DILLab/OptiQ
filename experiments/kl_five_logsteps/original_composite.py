"""Combine new 0/10/100 observations with existing 1K/10K/100K runs.

This is a disclosed composite of replayed and original segments, not a single
continuous reconstructed trajectory. No training, smoothing or interpolation.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from ..kl_five_progress.figure import BLUE, ORANGE, gm_ref, ng_ref
from ..kl_five_progress.prepare import CASES, original_folder

STEPS = [0, 10, 100, 1000, 10000, 100000]
NAMES = {
    't00_reference': 'Three Gaussian modes',
    'n00_spike_ramp': 'Spike + ramp',
    'n07_spike_flat_ramp': 'Spike + plateau + ramp',
    't01_two_offset': 'Two offset Gaussian modes',
    't05_unequal_mass': 'Unequal-mass Gaussian modes',
}


def read(file):
    return json.loads(file.read_text())


def sha(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def old_record(workspace, task, step):
    folder = original_folder(workspace, task['case'], task['method'], task['seed'], step)
    file = folder / f'samples_{step:06d}.npz'
    metric = read(folder / f'metrics_{step:06d}.json')
    with np.load(file) as z:
        mass, edges = z['histogram_mass'].copy(), z['edges'].copy()
        count = len(z['actions'])
        assert count == metric['sample_count']
        assert len(mass) == 512 and np.isclose(mass.sum(), 1)
        assert np.allclose(np.histogram(z['actions'], edges)[0] / count, mass)
        assert np.isclose(.5 * np.abs(mass - z['target_mass']).sum(), metric['histogram_TV'])
        probe = {k: z[k].ravel()[:128].copy() for k in ['actions', 'mu', 'sigma']}
    return dict(histogram_mass=mass, edges=edges, count=count, metric=metric, probe=probe,
                file=str(file), sha256=sha(file), source_commit=task['original_commit'])


def draw(case, data, out, pdf):
    fig, axes = plt.subplots(2, 6, figsize=(19.0, 6.0))
    fig.subplots_adjust(left=.072, right=.994, bottom=.14, top=.90, wspace=.20, hspace=.16)
    for row, method in enumerate(['forward', 'reverse']):
        color = BLUE if method == 'forward' else ORANGE
        ymax = 1.06 * max(max(d['values'].max(), d['target'].max())
                          for key, d in data.items() if key[0] == case and key[1] == method)
        for col, step in enumerate(STEPS):
            ax = axes[row, col]
            d = data[case, method, step]
            for values in d['values']:
                ax.stairs(values, d['edges'], color=color, alpha=.18, lw=.5)
            mean = d['values'].mean(0)
            ax.stairs(mean, d['edges'], color=color, fill=True, alpha=.20, lw=0)
            ax.stairs(mean, d['edges'], color=color, lw=1.4)
            ax.plot(d['x'], d['target'], 'k--', lw=1.3)
            ax.set(xlim=(-10, 10), ylim=(0, ymax), xticks=[-10, -5, 0, 5, 10])
            ax.tick_params(length=3, width=.7, labelbottom=row == 1)
            if row == 1:
                ax.set_xlabel(r'$a$')
            if row == 0:
                label = 'Initialization' if step == 0 else (f'{step // 1000}K' if step >= 1000 else str(step))
                label = label if step == 0 else f'{label} step'
                ax.set_title(f'({chr(97 + col)}) {label}', pad=9)
            if col == 0:
                ax.set_ylabel('Density')
    for row, name in enumerate(['Forward KL', 'Reverse KL']):
        pos = axes[row, 0].get_position()
        fig.text(.018, (pos.y0 + pos.y1) / 2, name, rotation=90, ha='center', va='center', fontsize=17)
    # Visually mark the provenance boundary, but do not alter any density.
    boundary = (axes[0, 2].get_position().x1 + axes[0, 3].get_position().x0) / 2
    fig.add_artist(plt.Line2D([boundary, boundary], [.14, .90], transform=fig.transFigure,
                             color='.62', lw=.8, ls=':'))
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(out / f'{case}_2x6.{ext}', dpi=250, bbox_inches='tight', pad_inches=.06)
    pdf.savefig(fig, bbox_inches='tight', pad_inches=.06)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    study = args.workspace / 'studies/20260926_kl_five_progress'
    tasks = read(study / 'campaign/data/TASKS.json')
    records = read(args.bundle)['records']
    ready, missing = [], []
    for case in CASES:
        absent = [f'{case}/{m}/{seed}/{step}' for m in ['forward', 'reverse']
                  for seed in range(4) for step in [0, 10, 100]
                  if f'{case}/{m}/{seed}/{step}' not in records]
        if absent:
            missing.extend(absent)
        else:
            ready.append(case)
    args.out.mkdir(parents=True, exist_ok=True)
    data, inputs, metrics, overlap = {}, [], [], []
    for task in tasks:
        case, method, seed = task['case'], task['method'], task['seed']
        if case not in ready:
            continue
        for step in STEPS:
            if step <= 100:
                item = records[f'{case}/{method}/{seed}/{step}']
                assert item['L'] == task['L'] == (2**20 if method == 'reverse' else 0)
                assert item['initial_parameter_sha256'] == task['initial_parameter_sha256']
            else:
                item = old_record(args.workspace, task, step)
            mass, edges = np.asarray(item['histogram_mass']), np.asarray(item['edges'])
            key = case, method, step
            if key not in data:
                x = np.linspace(-10, 10, 16385)
                ref = ng_ref if task['config']['target_kind'] == 'nongmm' else gm_ref
                data[key] = dict(values=[], edges=edges, x=x, target=ref(task['config'], x)[0])
            assert np.array_equal(edges, data[key]['edges'])
            data[key]['values'].append(mass / np.diff(edges))
            inputs.append(dict(case=case, method=method, seed=seed, step=step,
                               segment='early_replay' if step <= 100 else 'original',
                               **{k: item[k] for k in ['file', 'sha256', 'count', 'source_commit']}))
            metrics.append(dict(case=case, method=method, seed=seed, step=step, samples=item['count'],
                                TV=item['metric']['histogram_TV']))
        # Overlap is diagnostic only: no exclusions based on agreement/outcomes.
        overlap_key = f'{case}/{method}/{seed}/1000'
        if overlap_key in records:
            early = records[overlap_key]
            old = old_record(args.workspace, task, 1000)
            overlap.append(dict(case=case, method=method, seed=seed,
                                histogram_TV_between_runs=float(.5 * np.abs(
                                    np.asarray(early['histogram_mass']) - old['histogram_mass']).sum())))
    for d in data.values():
        d['values'] = np.asarray(d['values'])
        assert d['values'].shape == (4, 512)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
                         'mathtext.fontset': 'dejavusans', 'font.size': 11, 'axes.titlesize': 17,
                         'axes.labelsize': 15, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
                         'axes.edgecolor': '#666666', 'axes.linewidth': .8,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    with PdfPages(args.out / 'five_environments_progress.pdf') as pdf:
        for case in ready:
            draw(case, data, args.out, pdf)
    for name, rows in [('metrics.csv', metrics), ('overlap_1k.csv', overlap)]:
        if rows:
            with (args.out / name).open('w') as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
    note = ('초기화·10·100 updates는 동일 설정·초기화에서 별도로 재현한 결과이고, '
            '1K·10K·100K는 원래 저장된 결과입니다. 점선 구분선은 이 출처의 경계를 표시합니다. '
            '단일 실행의 연속 checkpoint를 복구한 그림은 아닙니다.')
    sample_note = ('각 패널은 seed 0–3의 512-bin histogram 평균입니다. 초기화·10·100·100K는 '
                   'seed당 2^20개, 원래 1K·10K는 seed당 32,768개 action을 사용합니다. '
                   'KDE·보간 없이 실제 저장 샘플을 사용했습니다. 파랑: Forward KL, 주황: Reverse KL, '
                   '검정 점선: true target. 옅은 선: 개별 seed. N=M=128, batch=32, Reverse L=2^20.')
    prov = dict(plot_commit=args.commit, steps=STEPS, ready_cases=ready, missing=missing,
                reverse_L=2**20, seeds=list(range(4)), bins=512, inputs=inputs,
                overlap_1k=overlap, composite_segments=True, note=note)
    (args.out / 'PROVENANCE.json').write_text(json.dumps(prov, ensure_ascii=False, indent=2) + '\n')
    text = '# Initialization → 10 → 100 → 1K → 10K → 100K\n\n' + note + '\n\n' + sample_note + '\n\n'
    text += 'All four seeds are required for each environment; availability is the only inclusion criterion.\n'
    (args.out / 'README.md').write_text(text)
    html = ('<!doctype html><html lang="ko"><meta charset="utf-8"><title>Six-timepoint density progression</title>'
            '<style>body{font-family:Arial,sans-serif;margin:28px;color:#222}img{width:100%;max-width:2000px}'
            'p{line-height:1.65}section{margin:30px 0 50px}a{color:#176399}</style>'
            '<h1>Initialization → 10 → 100 → 1K → 10K → 100K</h1>'
            f'<p>{note}</p><p>{sample_note}</p><p><a href="five_environments_progress.pdf">전체 PDF</a></p>')
    if missing:
        html += f'<p>완성된 환경 {len(ready)}/5. 아직 초반 기록이 없는 환경은 포함하지 않았습니다.</p>'
    for i, case in enumerate(CASES, 1):
        if case in ready:
            stem = case + '_2x6'
            html += (f'<section><h2>{i}. {NAMES[case]}</h2><p><a href="{stem}.png">PNG</a> · '
                     f'<a href="{stem}.pdf">PDF</a> · <a href="{stem}.svg">SVG</a></p>'
                     f'<img src="{stem}.png" alt="{NAMES[case]} training progression"></section>')
    (args.out / 'report.html').write_text(html + '</html>')
    (args.out / 'SHA256.json').write_text(json.dumps({p.name: sha(p) for p in args.out.iterdir()
        if p.is_file() and p.name != 'SHA256.json'}, indent=2) + '\n')
    print(json.dumps(dict(ready=ready, missing=missing, out=str(args.out))))


if __name__ == '__main__':
    main()
