"""Collect and verify the frozen SQL particle ablation, then plot all seeds."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
CAMPAIGN = 'gmm40-sql-particles-100k-4seed-20260925'
SOURCE = '8a73d69112d73fd88867bfe75d42ba9a4348a379'
BASELINE = ROOT.parent/'gmm40_5090_queue/results/results'
HOSTS = [('vast-heechan-180', 0), ('vast-heechan-199', 1)]
STEPS = [0, 100, 500, 1000, 2500, 5000] + list(range(10000, 100001, 10000))
KS = [16, 32, 64, 128, 256]


def read(path):
    return json.loads(path.read_text())


def sync():
    for host, shard in HOSTS:
        dest = ROOT/f'shard{shard}'
        dest.mkdir(parents=True, exist_ok=True)
        source = f'{host}:/home/heechan/optiq-experiments/{CAMPAIGN}/'
        subprocess.run(['rsync', '-a', '--prune-empty-dirs', '--include=*/', '--include=*.json',
                        '--include=**/step_0100000/samples.npy', '--exclude=*', source, str(dest)+'/'],
                       check=True)
    dest = ROOT/'baseline_history'
    dest.mkdir(exist_ok=True)
    source = 'vast-heechan-180:/home/heechan/optiq-experiments/gmm40-5090-100k-4seed-20260921/results/'
    subprocess.run(['rsync', '-a', '--prune-empty-dirs', '--include=*/', '--include=*.json',
                    '--exclude=*', source, str(dest)+'/'], check=True)


def collect():
    reference = read(ROOT/'shard0/results/target/definition.json')
    values = {key: reference[key] for key in ('means', 'std', 'weights')}
    assert hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest() == (
        'a22a67f6f37d782fd75961e75f5df13c424fda9624245c99efe93862c929e87c')
    baseline_target = read(BASELINE/'target/definition.json')
    assert {key: baseline_target[key] for key in values} == values
    rows, histories = [], {}
    for k in KS:
        histories[k] = {}
        for seed in range(4):
            if k == 16:
                folder = BASELINE/f'sql_s{seed}_100k'
                hist = ROOT/'baseline_history'/f'sql_s{seed}_100k'
            else:
                folder = ROOT/f'shard{seed//2}/results'/f'sql_k{k}_s{seed}_100k'
                hist = folder
            cfg = read(folder/'config.json')
            assert cfg['method'] == 'sql' and cfg['seed'] == seed
            assert cfg['steps'] == 100000 and cfg['batch'] == 256
            assert cfg['width'] == 256 and cfg['depth'] == 2 and cfg['temperature'] == 1
            assert cfg['eval_samples'] == 10000
            assert cfg['sql_kernel_particles'] == k and cfg['sql_kernel_update_ratio'] == .5
            assert cfg['sql_value_particles'] == 16 and cfg['sql_target_update_interval'] == 1000
            if k != 16:
                assert cfg['source_git_commit'] == SOURCE
            assert read(folder/'status.json')['status'] == 'completed'
            audit = read(folder/'update_count_audit.json')
            assert audit['status'] == 'passed' and audit['actor_updates'] == 100000
            final = folder/'evaluations/step_0100000'
            metrics = read(final/'metrics.json')
            samples = np.load(final/'samples.npy')
            assert samples.shape == (10000, 2) and np.isfinite(samples).all()
            assert metrics['n_samples'] == 10000 and 0 <= metrics['mode_coverage'] <= 40
            histories[k][seed] = {}
            for step in STEPS:
                p = hist/'evaluations'/f'step_{step:07d}'/'metrics.json'
                if p.exists():
                    histories[k][seed][step] = read(p)
            assert 100000 in histories[k][seed]
            rows.append(dict(k=k, seed=seed, coverage=metrics['mode_coverage'],
                             mmd2=metrics['mmd2'], near=metrics['high_density_fraction'],
                             train_seconds=read(folder/'status.json').get('train_seconds'),
                             q_queries=100000*256*(k//2), source=cfg['source_git_commit']))
    return rows, histories, reference


def draw(rows, histories, reference):
    means = np.asarray(reference['means']); std = np.asarray(reference['std'])
    grid = np.linspace(-40, 40, 181)
    xx, yy = np.meshgrid(grid, grid)
    xy = np.stack((xx, yy), axis=-1)
    delta = (xy[..., None, :] - means) / std[None, None, :, None]
    components = -.5*np.sum(delta**2, axis=-1)-2*np.log(std)-np.log(2*np.pi)
    peak = components.max(axis=-1)
    density = peak+np.log(np.exp(components-peak[..., None]).sum(axis=-1))-np.log(40)
    fig, axes = plt.subplots(5, 4, figsize=(15.5, 18), constrained_layout=True)
    for row, k in enumerate(KS):
        for seed in range(4):
            ax = axes[row, seed]
            folder = BASELINE/f'sql_s{seed}_100k' if k == 16 else ROOT/f'shard{seed//2}/results'/f'sql_k{k}_s{seed}_100k'
            samples = np.load(folder/'evaluations/step_0100000/samples.npy')
            item = next(r for r in rows if r['k'] == k and r['seed'] == seed)
            ax.contour(xx, yy, density, levels=[-12, -9, -7, -6, -5], colors='#777777', linewidths=.4)
            ax.scatter(samples[:, 0], samples[:, 1], s=1, alpha=.2, c='#0000cc', linewidths=0, rasterized=True)
            ax.scatter(means[:, 0], means[:, 1], s=11, c='black', marker='+', linewidths=.6)
            ax.set(xlim=(-40, 40), ylim=(-40, 40), aspect='equal',
                   title=f'K={k}, seed {seed} | {item["coverage"]}/40, MMD²={item["mmd2"]:.3f}')
    fig.suptitle('GMM40 SQL SVGD | 100k updates | 10,000 native policy samples')
    fig.savefig(ROOT/'sql_particle_distributions.png', dpi=180)
    fig.savefig(ROOT/'sql_particle_distributions.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    colors = ['#4c78a8', '#f58518', '#54a24b', '#e45756', '#b279a2']
    for k, color in zip(KS, colors):
        steps = sorted(set.intersection(*(set(histories[k][s]) for s in range(4))))
        for ax, key in zip(axes[:2], ('mode_coverage', 'mmd2')):
            values = np.array([[histories[k][s][step][key] for step in steps] for s in range(4)])
            avg = values.mean(0); sd = values.std(0, ddof=1)
            ax.plot(steps, avg, color=color, label=f'K={k}')
            ax.fill_between(steps, avg-sd, avg+sd, color=color, alpha=.12)
    axes[0].set(ylabel='Covered GT components /40', xlabel='Actor updates')
    axes[1].set(ylabel='MMD² ↓', xlabel='Actor updates')
    for k, color in zip(KS, colors):
        vals = np.array([next(r for r in rows if r['k']==k and r['seed']==s)['train_seconds'] for s in range(4)])
        axes[2].bar(str(k), vals.mean(), yerr=vals.std(ddof=1), color=color)
    axes[2].set(xlabel='SVGD particles', ylabel='Training seconds (reported)')
    axes[0].legend(frameon=False)
    fig.savefig(ROOT/'sql_particle_curves.png', dpi=180)
    fig.savefig(ROOT/'sql_particle_curves.pdf')
    plt.close(fig)


def report():
    rows, histories, reference = collect()
    draw(rows, histories, reference)
    with (ROOT/'per_seed.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = []
    for k in KS:
        selected = [r for r in rows if r['k']==k]
        summary.append(dict(k=k, seeds=4, q_queries=selected[0]['q_queries'],
                            **{field: dict(mean=float(np.mean([r[field] for r in selected])),
                                           sample_sd=float(np.std([r[field] for r in selected], ddof=1)))
                               for field in ('coverage', 'mmd2', 'near', 'train_seconds')}))
    output = dict(status='completed', rows=rows, summary=summary, new_source_commit=SOURCE,
                  baseline_source_commit='87d5d8ff210569ace7e59bd8a53ad02141b67f0a')
    (ROOT/'results.json').write_text(json.dumps(output, indent=2)+'\n')
    lines = ['# GMM40 SQL SVGD 입자 수 실험', '', '100k actor updates, 4 seeds, fixed Q, batch 256, T=1, 256×2. K=16은 기존 baseline.', '',
             '| K | Mode coverage /40 ↑ | MMD² ↓ | Near 3σ ↑ | 학습 시간 (s) | Q queries |',
             '|---:|---:|---:|---:|---:|---:|']
    for item in summary:
        def f(key, precision):
            val = item[key]; return f"{val['mean']:.{precision}f} ± {val['sample_sd']:.{precision}f}"
        lines.append(f"| {item['k']} | {f('coverage', 2)} | {f('mmd2', 4)} | {f('near', 3)} | {f('train_seconds', 1)} | {item['q_queries']:,} |")
    lines += ['', '평균 ± seed 간 sample SD. 입자를 늘리면 update 수는 같지만 Q query와 SVGD pairwise 계산량이 증가한다.',
              '모드 커버율은 GT 중심 3σ 안의 샘플 수가 기준 이상인 component 수다.',
              '새 16개 run은 서버의 W&B 인증 키 부재로 업로드되지 않았다. 로컬 결과와 SHA256 검증 기록을 보존했다.',
              '', '![분포](sql_particle_distributions.png)', '', '![학습곡선](sql_particle_curves.png)']
    (ROOT/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:10]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sync', action='store_true')
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()
    if args.sync: sync()
    if args.report: report()
