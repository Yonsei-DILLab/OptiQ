"""Create fixed-action score-versus-log-L figures and a self-contained report."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

COLORS = {'forward': '#087e8b', 'reverse': '#d85832'}
MARK = 2**20


def axis_style(ax):
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_locator(FixedLocator(2.**np.array([7, 10, 14, 17, 20, 24])))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: r'$2^{%d}$' % round(np.log2(x))))
    ax.axvline(MARK, color='#b82455', linestyle='--', lw=1.6, label=r'Training $L=2^{20}$')
    ax.grid(alpha=.2)
    ax.set_xlabel('Density-bank size L (log scale)')


def draw(ax, z, action_index, method):
    Ls, scores = z['Ls'], z['scores'][:, :, action_index]
    refs = z['reference_scores'][:, action_index]
    color = COLORS[method]
    for trace in scores:
        ax.plot(Ls, trace, color=color, alpha=.10, lw=.7)
    mean = scores.mean(0)
    lower, upper = np.quantile(scores, [.1, .9], axis=0)
    ax.fill_between(Ls, lower, upper, color=color, alpha=.2, label='10-90% MC range')
    ax.plot(Ls, mean, color=color, marker='o', ms=3, lw=1.6, label='Mean of 16 IID banks')
    ref, se = refs.mean(), refs.std(ddof=1)/np.sqrt(len(refs))
    ax.axhspan(ref-2*se, ref+2*se, color='#555555', alpha=.15)
    ax.axhline(ref, color='#333333', ls=':', lw=1.2, label=r'Independent $2^{24}$ MC reference')
    idx = np.flatnonzero(Ls == MARK)[0]
    ax.scatter([MARK], [mean[idx]], color='#b82455', s=44, zorder=5)
    ax.text(.98, .97, r'$s_{2^{20}}=%.4f$' % mean[idx]+'\n'+r'$mathrm{SD}=%.4f$' % scores[:, idx].std(ddof=1),
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(facecolor='white', alpha=.8, edgecolor='none'))
    axis_style(ax)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--results', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args=ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    data={}; summary={}
    for method in ['forward', 'reverse']:
        for seed in range(4):
            name=f'{method}_s{seed}'; path=args.results/name
            if not (path/'COMPLETE.json').exists():
                continue
            data[name]=np.load(path/'scores.npz')
            summary[name]=json.loads((path/'SUMMARY.json').read_text())
    assert len(data)==8, f'Expected all8 finished actors, got{len(data)}'

    # Main figure: seed0 prespecified, same fixed policy-quantile actions at every L.
    fig, axes=plt.subplots(2,3,figsize=(16,8), constrained_layout=True)
    for row,method in enumerate(['forward','reverse']):
        z=data[f'{method}_s0']
        for col,q in enumerate([10,50,90]):
            ai=128+col; ax=axes[row,col]; draw(ax,z,ai,method)
            ax.set_title(f'{method.capitalize()} | fixed {q}% policy quantile\na = {z["actions"][ai]:.3f}')
            if col==0: ax.set_ylabel(r'Actor score $s_L(a)=\partial_a\log q_L(a)$')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=4,fontsize=9)
    fig.suptitle('Frozen 100K actors: score convergence as L increases\nSeed 0 | N=M=128 | fixed actions and parameters | 16 independent repetitions', fontsize=15)
    fig.savefig(args.output/'score_vs_L_main.png',dpi=180); plt.close(fig)

    # External probes: show every prespecified action, including low-density tails.
    fig,axes=plt.subplots(2,9,figsize=(30,7.2),constrained_layout=True)
    for row,method in enumerate(['forward','reverse']):
        z=data[f'{method}_s0']
        for col in range(9):
            ai=131+col; ax=axes[row,col]; draw(ax,z,ai,method)
            ax.set_title(f'{method.capitalize()} | a={z["actions"][ai]:g}',fontsize=11)
            ax.tick_params(labelsize=8)
            if col==0: ax.set_ylabel('Actor score')
    fig.suptitle('Prespecified target-region probes | Seed0 | same fixed action for both actors\nRare/off-policy actions are shown separately; stability here can require a much larger bank',fontsize=16)
    fig.savefig(args.output/'score_vs_L_target_probes.png',dpi=160);plt.close(fig)

    fig,axes=plt.subplots(4,6,figsize=(24,13),constrained_layout=True)
    for seed in range(4):
        for mi,method in enumerate(['forward','reverse']):
            z=data[f'{method}_s{seed}']
            for j,q in enumerate([10,50,90]):
                ai=128+j;ax=axes[seed,mi*3+j];draw(ax,z,ai,method)
                ax.set_title(f'{method.capitalize()} s{seed}, q{q}, a={z["actions"][ai]:.3f}',fontsize=10)
                ax.tick_params(labelsize=8)
                if mi==0 and j==0:ax.set_ylabel('Actor score')
    fig.suptitle('All eight frozen 100K actors | no seed exclusion',fontsize=17)
    fig.savefig(args.output/'score_vs_L_all_seeds.png',dpi=160);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(13,4.8),constrained_layout=True)
    for ax,method in zip(axes,['forward','reverse']):
        for seed in range(4):
            s=summary[f'{method}_s{seed}']; rows=s['results']
            x=[r['L'] for r in rows];y=[100*r['policy_relative_RMSE_mean'] for r in rows]
            ax.plot(x,y,marker='.',label=f'Seed {seed}')
        axis_style(ax);ax.set_yscale('log');ax.axhline(1,color='#555',ls=':',lw=1)
        ax.set_title(method.capitalize());ax.set_ylabel('Score RMSE / reference score RMS (%)')
        ax.legend(fontsize=9)
    fig.suptitle(r'128 fixed policy actions | deviation from independent $2^{24}$ MC reference (not exact)',fontsize=13)
    fig.savefig(args.output/'score_relative_error.png',dpi=180);plt.close(fig)

    lines=['# 고정 actor에서 L에 따른 score의 Monte Carlo 수렴', '',
           '## 먼저 읽을 그림','', '![고정 action에서 score와 L](score_vs_L_main.png)', '',
           '최종100K checkpoint를 고정했다. 위는 사전에 지정한 seed0이고, 각 열은 actor가 실제로 생성한 행동의10%,50%,90% 분위수다. **같은 패널에서는 모든 L·반복에 완전히 동일한 action과 actor parameter를 사용한다.** Forward와 reverse는 분포가 다르므로 각자의 분위수 action도 다르다. 동일 action의 두 actor 비교는 아래 target-region 그림을 사용한다.', '',
           '- 실선: 독립16회 MC 추정의 평균. 얇은 선: 개별 반복.',
           '- 색 음영:16회 추정값의10–90% 구간. 평균의 신뢰구간이 아니다.',
           '- 분홍 점선: 실제 reverse 학습에 쓴 L=2²⁰=1,048,576.',
           '- 검정 점선: 별도로 뽑은 독립4개 L=2²⁴ bank의 평균. 회색 띠: 그 평균 ±2 standard errors. **정확한 적분값은 아니다.**','',
           '## 무엇을 추정했는가','',
           r'$$q_\theta(a)=\mathbb E_{z\sim\mathcal N(0,1)}[k_\theta(a\mid z)],\qquad s_\theta(a)=\partial_a\log q_\theta(a).$$', '',
           r'$$\widehat s_L(a)=\frac{\sum_{\ell=1}^L k_\theta(a\mid z_\ell)\,\frac{\mu_\theta(z_\ell)-a}{\sigma_\theta(z_\ell)^2}}{\sum_{\ell=1}^L k_\theta(a\mid z_\ell)},\quad z_\ell\overset{\mathrm{iid}}\sim\mathcal N(0,1).$$', '',
           'y축은 **학습된 actor의 action score**다. target GMM의 score나 파라미터 gradient가 아니다. 각 Gaussian은[-10,10]에 대해 정규화된 truncated Gaussian이다. 분모·분자를 따로 MC 추정한 비율이므로, 유한L에서 일반적으로 unbiased하지 않다.', '',
           '## 설정','',
           '|항목|값|','|---|---|',
           '|학습 조건|f05: 중심(-4.25,0,4.25), 폭0.5, mean-head scale3|',
           '|선택한 actor|Forward/reverse × seeds0–3, 전부100K checkpoint|',
           '|훈련 N,M,batch|128,128,32; 이번 분석에서는 업데이트 없음|',
           '|MC L|2⁷부터2²⁴까지 모든2의 거듭제곱|',
           '|반복|독립16회; 한 반복 안에서는 작은L bank가 큰L bank의 prefix|',
           '|고정 action|실제 평가표본에서128개, 분위수3개, 사전 target-region9개|',
           '|추가 reference|다른 random seed의 독립4개 L=2²⁴ bank|',
           '|수치 계산|actor float32/highest matmul, density·score 합산 float64 및 max scaling|','',
           '## L=2²⁰과 더 큰 L 비교','',
           '아래는 실제 actor action128개에서 score RMSE를 reference score RMS로 나눈 값이다.0에 가까운 점의 상대오차가 폭발하지 않도록 점별 나눗셈 대신 전체 RMS로 정규화했다. 평균은 독립16회에 대한 평균이며, reference 자체에도 MC 오차가 있다.', '',
           '|Actor|L=2²⁰ 상대 RMSE|L=2²²|L=2²⁴|Reference SE / score RMS|','|---|---:|---:|---:|---:|']
    for name,s in summary.items():
        values={r['L']:r for r in s['results']}
        row=[100*values[2**p]['policy_relative_RMSE_mean'] for p in [20,22,24]]
        se=100*s['reference_policy_SE_RMS']/s['reference_policy_score_RMS']
        lines.append(f'|{name}|{row[0]:.4f}%|{row[1]:.4f}%|{row[2]:.4f}%|{se:.4f}%|')
    lines+=['','![고정 action 전체의 score 오차](score_relative_error.png)','',
            '## 모든 seed','', '![모든 seed의 score](score_vs_L_all_seeds.png)','',
            '## 목표 mode 주변의 고정 위치','',
            '![target-region 고정 action](score_vs_L_target_probes.png)','',
            'Reverse는 외곽 mode를 거의 방문하지 않는다. 이런 낮은 policy density 영역에서의 추정 난이도와 실제 학습이 주로 받는 action에서의 난이도를 구분해야 한다. 이 그림의 점선도 해당 **actor**의 큰L score이며, 목표분포 score가 아니다.', '',
            '## 해석 범위','',
            '이 검사는 선택한 고정 actor와 action에서 유한L 추정이 얼마나 안정되는지 보여준다. 전체 학습 시점·모든 action·모든 초기화에서 L=2²⁰이 충분하다는 보장은 아니다. 큰L reference와 일치해도 미관측 극저확률 latent 기여가 없다는 수학적 증명은 아니다. 중간 checkpoint는 저장되어 있지 않아 최종100K 상태만 검사했다.', '',
            '## 수치 검증과 재현','',
            '동일 유한 bank에 대해 streaming score를 dense autodiff와 비교했다. actor와 checkpoint는 읽기만 했고 분석 전후 hash가 동일함을 확인했다. 각 run의 RUN.json, VALIDATION.json, SUMMARY.json, scores.npz에 provenance와 전체 반복값을 보관한다.','']
    meta=json.loads((args.results/'reverse_s0/RUN.json').read_text())
    lines += [f"- 학습 source commit: `{meta['parent_source_commit']}`", f"- 분석 source commit: `{meta['analysis_commit']}`", '']
    (args.output/'report.md').write_text('\n'.join(lines))
    (args.output/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
