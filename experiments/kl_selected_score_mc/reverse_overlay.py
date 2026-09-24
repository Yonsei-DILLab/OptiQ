"""Presentation-only: three fixed reverse-policy action quantiles on one axis."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--commit',required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    raw=args.results/'reverse_s0'/'scores.npz';z=np.load(raw)
    Ls=z['Ls'];ix=np.arange(128,131);actions=z['actions'][ix]
    scores=z['scores'][:,:,ix];refs=z['reference_scores'][:,ix]
    assert scores.shape==(16,18,3) and refs.shape==(4,3)
    summary=json.loads((args.results/'reverse_s0/SUMMARY.json').read_text())
    assert np.allclose(actions,summary['fixed_quantile_actions'])
    plt.style.use('default')
    plt.rcParams.update({'font.size':11,'axes.titlesize':15,'axes.labelsize':12,
                         'legend.fontsize':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(8,5.4))
    records=[]
    for j,q in enumerate([10,50,90]):
        v=scores[:,:,j];mu=v.mean(0);lo,hi=np.quantile(v,[.1,.9],axis=0)
        color=f'C{j}';ref=refs[:,j].mean();se=refs[:,j].std(ddof=1)/2
        ax.axhspan(ref-2*se,ref+2*se,color=color,alpha=.08,linewidth=0)
        ax.axhline(ref,color=color,ls=':',lw=1.2,alpha=.75)
        ax.fill_between(Ls,lo,hi,color=color,alpha=.18,linewidth=0)
        ax.plot(Ls,mu,color=color,marker='o',ms=3,lw=1.6,
                label=f'{q}% quantile\n'+r'$a=%.3f$'%actions[j])
        at=int(np.flatnonzero(Ls==2**20)[0])
        records.append({'quantile':q,'action':float(actions[j]),'score_L2p20_mean':float(mu[at]),
                        'score_L2p20_SD':float(v[:,at].std(ddof=1)),
                        'reference_mean':float(ref),'reference_SE':float(se)})
    ax.set_xscale('log',base=2)
    ax.xaxis.set_major_locator(FixedLocator(2.**np.array([7,10,14,17,20,24])))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:r'$2^{%d}$'%round(np.log2(x))))
    ax.axvline(2**20,color='.35',ls='--',lw=1.3)
    ax.text(2**20*1.13,.98,r'Training $L=2^{20}$',transform=ax.get_xaxis_transform(),
            ha='left',va='top',fontsize=10,color='.25')
    ax.set_xlim(2**7,2**24);ax.margins(y=.15)
    ax.set_xlabel(r'Density-bank size $L$ (log scale)')
    ax.set_ylabel(r'Actor score $\widehat{s}_L(a)$')
    ax.set_title('Reverse KL: score convergence')
    ax.grid(alpha=.15)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.19),ncol=3,frameon=False,
              title='Fixed policy-action quantile')
    fig.subplots_adjust(left=.12,right=.97,top=.9,bottom=.28)
    for ext in ['png','pdf','svg']:
        fig.savefig(args.output/f'score_vs_L_reverse.{ext}',dpi=300)
    plt.close(fig)
    caption=r'''**Monte Carlo score convergence for a frozen reverse-KL actor.** The three curves use fixed actions at the 10th, 50th, and 90th percentiles of the seed-0 policy after 100,000 updates. Each action and the actor parameters remain identical across all density-bank sizes $L$. Solid lines show means over 16 independent MC repetitions; shaded bands show the 10–90% range of those estimates, not confidence intervals for the mean. Colored horizontal dotted lines are action-specific reference means from four separate independent banks of size $2^{24}$, not exact scores. The gray vertical dashed line marks the training bank size $L=2^{20}$. Percentiles in the legend refer to action locations; the shaded percentile range refers to MC score estimates. This diagnoses the final actor at typical policy actions, not all training stages or low-density external regions.'''
    (args.output/'caption_reverse.md').write_text(caption+'\n')
    meta={'plot_commit':args.commit,'source_scores_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),
          'actor':'reverse_s0','checkpoint_updates':100000,'records':records,
          'numerical_data_changed':False,'seed_selection':'prespecified seed 0, unchanged from original main figure'}
    (args.output/'REVERSE_PLOT_PROVENANCE.json').write_text(json.dumps(meta,indent=2)+'\n')
    lines=['# Reverse KL: L에 따른 score 추정의 수렴','',
           '![Reverse 세 action의 score](score_vs_L_reverse.png)','',
           '**Reverse만, 하나의 좌표축에 세 action을 겹쳐 그렸다.** 기존 대표 그림과 동일한 seed 0의 100K checkpoint와 저장된 MC 결과를 사용했다. 학습이나 MC 추정을 다시 실행하지 않았다.','',
           '- 범례의 10%·50%·90%: 학습된 policy action의 분위수. 각 곡선은 그 위치의 action 하나를 고정해 사용한다.',
           '- 실선: 독립 16회 MC 추정 평균. 색 음영: 추정값의 10–90% 구간이며 평균의 신뢰구간이 아니다.',
           '- 색 점선: 각 action에서 별도의 독립 L=2²⁴ bank 4개로 계산한 MC 기준 평균. 정확한 적분값이 아니다.',
           '- 회색 세로 점선: 실제 Reverse 학습에서 사용한 L=2²⁰.', '',
           '범례의 분위수는 **action 위치**, 음영의 분위수는 **MC score의 반복 간 변동**이다. 서로 다른 의미다.','',
           '|Policy 분위수|고정 action|L=2²⁰ score 평균|반복 간 SD|큰-L 기준 평균|',
           '|---:|---:|---:|---:|---:|']
    for rec in records:
        lines.append(f'|{rec["quantile"]}%|{rec["action"]:.6f}|{rec["score_L2p20_mean"]:.6f}|{rec["score_L2p20_SD"]:.6f}|{rec["reference_mean"]:.6f}|')
    lines+=['','## 측정량','',
            r'$$\widehat{s}_L(a)=\partial_a\log\widehat q_L(a)=\frac{\sum_{\ell=1}^L k_\theta(a\mid z_\ell)(\mu_\theta(z_\ell)-a)/\sigma_\theta(z_\ell)^2}{\sum_{\ell=1}^L k_\theta(a\mid z_\ell)},\qquad z_\ell\sim\mathcal N(0,1).$$','',
            '학습된 actor의 action score를 측정한다. 각 conditional Gaussian은 [-10,10]에서 정규화된 truncated Gaussian이며, target GMM의 score나 parameter gradient가 아니다. L은 2⁷부터 2²⁴까지이다. 16개 반복은 서로 독립이고, 같은 반복 안에서는 작은 bank가 큰 bank의 prefix다.','',
            '## 해석 범위','',
            '최종 Reverse actor가 실제로 방문하는 128개 고정 action에서 L=2²⁰의 큰-L 기준 대비 평균 상대 score RMSE는 네 seed에 걸쳐 0.10–0.14%였다. 그림은 그중 사전 지정한 seed 0의 세 action이다. 서로 다른 action들의 score를 한 y축에 표시해 작은 추정 차이는 기존 개별 패널보다 덜 두드러질 수 있다.','',
            'Reverse가 거의 방문하지 않는 외곽 mode(a=±4.25)에서는 L=2²⁴까지도 drift가 남았다. 모든 action·모든 학습 시점에서 L=2²⁰이 충분하다는 주장은 하지 않는다. Forward를 포함한 기존 진단은 archive_before_reverse_overlay/에 보존했다.','',
            '[PDF](score_vs_L_reverse.pdf) · [SVG](score_vs_L_reverse.svg)','',
            '## 논문용 caption','',caption,'',
            '## 재현','',f'- Figure source commit: `{args.commit}`.',
            '- 원래 MC 분석 source: `d8610dd0e45320956c604c02d97cb44b7ac65992`.',
            '- 보관: `dildata:/data1/heejoonorm/OptiQ/studies/20260924_selected_score_mc/report/`.','']
    (args.output/'report.md').write_text('\n'.join(lines))
    print(json.dumps(meta,indent=2))


if __name__=='__main__':main()
