"""Render existing 1M-action metrics using 512 bins and filled policy densities.

Presentation-only revision: reuse histograms and Wasserstein metrics from rebin.py.
No sampling, retraining, KDE, or numerical-metric changes.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--analysis', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--commit', required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = np.load(args.analysis / 'histograms_512.npz')
    summary = json.loads((args.analysis / 'SUMMARY.json').read_text())
    edges = data['edges']; width = np.diff(edges)
    assert len(width) == 512
    metrics = summary['metrics']['512']; w1 = summary['wasserstein1']
    centers = np.array([-4.25, 0., 4.25]); std = .5; bound = 10.
    normalizer = np.mean(ndtr((bound-centers)/std)-ndtr((-bound-centers)/std))
    cdf = lambda x: np.mean(ndtr((x[:, None]-centers)/std)-ndtr((-bound-centers)/std),axis=1)/normalizer
    assert np.allclose(np.diff(cdf(edges)), data['target_mass'], atol=1e-14)
    x = np.linspace(-bound, bound, 8193)
    target = np.exp(-.5*((x[:, None]-centers)/std)**2).mean(1)/(std*np.sqrt(2*np.pi)*normalizer)
    plt.style.use('default')
    plt.rcParams.update({'font.size':12,'axes.titlesize':16,'axes.labelsize':13,
                         'xtick.labelsize':11,'ytick.labelsize':11,'legend.fontsize':10,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True, sharey=True)
    for k, (ax, method) in enumerate(zip(axes, ['forward', 'reverse'])):
        mass = np.stack([data[f'{method}_s{s}'] for s in range(4)])
        tv = .5 * np.abs(mass-data['target_mass']).sum(1)
        assert abs(tv.mean()-metrics[method]['TV_mean']) < 1e-12
        density = mass / width
        mean = density.mean(0); sd = density.std(0, ddof=1)
        # Pale baseline-to-mean fill is separate from the seed-uncertainty band.
        ax.stairs(mean, edges, baseline=0, fill=True, color=f'C{k}',
                  alpha=.18, linewidth=0, zorder=1)
        ax.fill_between(edges, np.r_[np.maximum(mean-sd, 0), max(mean[-1]-sd[-1], 0)],
                        np.r_[mean+sd, mean[-1]+sd[-1]], step='post', color=f'C{k}',
                        alpha=.25, linewidth=0, zorder=2, label=r'$\pm$1 SD across seeds')
        ax.stairs(mean, edges, color=f'C{k}', lw=1.3, label='Policy (mean)', zorder=3)
        ax.plot(x, target, 'k--', lw=1.8, label='Target', zorder=4)
        ax.set_title(f'{method.capitalize()} KL')
        ax.text(.035, .93, f'Mean TV = {metrics[method]["TV_mean"]:.4f}\nMean W1 = {w1[method]["mean"]:.4f}',
                transform=ax.transAxes, ha='left', va='top')
        ax.set(xlim=(-10, 10), ylim=(0, .9), xlabel='Action')
        ax.set_xticks([-10, -5, 0, 5, 10])
        handles, labels = ax.get_legend_handles_labels()
        order = [labels.index('Policy (mean)'), labels.index(r'$\pm$1 SD across seeds'), labels.index('Target')]
        ax.legend([handles[i] for i in order], [labels[i] for i in order], loc='upper right')
    axes[0].set_ylabel('Probability density')
    fig.tight_layout(pad=1.4, w_pad=2.)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(args.output/f'final_density_512.{ext}', dpi=300)
    plt.close(fig)
    caption = r'''**Forward versus reverse KL on a three-mode target.** Each policy is shown as a 512-bin histogram averaged across four seeds after 100,000 updates, using $2^{20}$ saved action samples per seed on $[-10,10]$. Pale fill denotes the area under the mean policy density; the darker band indicates $\pm1$ sample standard deviation across seeds. The unfilled dashed curve is the exact target density. No KDE smoothing is applied. TV is evaluated against exact target bin probabilities, and Wasserstein-1 (W1) uses sorted actions and exact target quantiles independently of the histogram. Annotations report averages of per-seed metrics. Both methods use $N=M=128$; reverse KL uses $L=2^{20}$. The equally weighted target modes have centers $(-4.25,0,4.25)$ and standard deviation $0.5$. This is the previously selected illustrative toy setting.'''
    (args.output/'caption.md').write_text(caption+'\n')
    lines = ['# Forward / Reverse KL 최종 비교 — 512 bins', '',
             '![4-seed 평균 histogram](final_density_512.png)', '',
             '각 actor에서 저장한 2²⁰개 action을 그대로 사용했다. 100K 학습 완료, seed 0–3, N=M=128이며 Reverse 학습은 L=2²⁰이다. 목표는 중심 (-4.25, 0, 4.25), 폭 0.5, 동일 질량의 세 Gaussian을 [-10,10]에서 정규화한 분포다.', '',
             '512-bin histogram을 seed별로 만든 뒤 평균했다. 옅은 색은 평균 policy density 아래의 면적이고, 곡선 주변의 더 진한 띠는 seed 간 ±1 SD다. Target은 채우지 않은 검정 점선이다. KDE smoothing은 사용하지 않았다.', '',
             '|방법|TV 평균 ± SD (512 bins)|Wasserstein-1 평균 ± SD|',
             '|---|---:|---:|']
    for method in ['forward','reverse']:
        m=metrics[method];w=w1[method]
        lines.append(f'|{method.capitalize()} KL|{m["TV_mean"]:.8f} ± {m["TV_SD"]:.8f}|{w["mean"]:.8f} ± {w["SD"]:.8f}|')
    lines += ['', 'TV는 각 seed의 bin 질량과 exact target bin 질량의 ½ L1 거리를 먼저 계산한 뒤 평균한 값이다. 평균 그림 자체의 TV가 아니다. W1은 정렬한 action과 exact target 분위수로 계산한 action 단위의 거리이므로 bin 수 변경의 영향을 받지 않는다.', '',
              '이전 4096-bin 분석은 별도 보존했다. 이번 수정은 512 bins로 돌아가고 policy 곡선 아래에 색을 채운 표시 방식 변경이다. 새 학습·샘플링은 없다.', '',
              '이 조건은 탐색으로 선정한 설명용 toy이다. 모든 target이나 초기화에서 Forward가 우월하다는 뜻은 아니다.', '',
              '[논문용 PDF](final_density_512.pdf) · [SVG](final_density_512.svg)', '',
              '## 논문용 caption', '', caption, '', '## 재현', '',
              f'- Plot source commit: `{args.commit}`.',
              f'- Metric source commit: `{summary["rebin_commit"]}`.',
              '- 원본 1M action 평가 source: `8b35c3ecae257846ee453e099af2432064e894f0`.',
              '- 보관: `dildata:/data1/heejoonorm/OptiQ/studies/20260924_selected_tv1m/final_512/`.', '']
    (args.output/'report.md').write_text('\n'.join(lines))
    summary.update({'selected_plot_bins':512,'plot_commit':args.commit,'policy_area_fill_alpha':.18,
                    'seed_sd_band_alpha':.25,'metric_source_summary_sha256':hashlib.sha256((args.analysis/'SUMMARY.json').read_bytes()).hexdigest()})
    (args.output/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'TV_512':metrics,'W1':{m:w1[m] for m in ['forward','reverse']}},indent=2))


if __name__ == '__main__':
    main()
