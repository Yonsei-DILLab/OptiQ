"""Publication-size paired histogram figure; TV is averaged after per-seed scoring."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import ndtr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator


def sd_math(x):
    if x>=1e-4:return f'{x:.4f}'
    if x==0:return '0'
    exponent=int(np.floor(np.log10(x)));return rf'{x/10**exponent:.1f}\times10^{{{exponent}}}'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    methods=['forward','reverse'];metrics={};hist={};runs={}
    for method in methods:
        metrics[method]=[];hist[method]=[]
        for seed in range(4):
            path=args.results/f'{method}_s{seed}';assert (path/'COMPLETE.json').exists()
            runs[f'{method}_s{seed}']=json.loads((path/'RUN.json').read_text())
            metric=json.loads((path/'METRICS.json').read_text());assert metric['samples']==2**20
            metrics[method].append(metric);d=np.load(path/'histogram.npz');hist[method].append(d['mass'])
            if method=='forward' and seed==0:edges=d['edges'];target_mass=d['target_mass']
            else:assert np.array_equal(edges,d['edges']) and np.array_equal(target_mass,d['target_mass'])
        hist[method]=np.stack(hist[method])
    cfg=runs['forward_s0']['parent_config'];centers=np.array(cfg['target_centers']);h=cfg['target_width'];bound=cfg['action_bound']
    x=np.linspace(-bound,bound,4097);Z=np.mean(ndtr((bound-centers)/h)-ndtr((-bound-centers)/h))
    target=np.exp(-.5*((x[:,None]-centers)/h)**2).mean(1)/(h*np.sqrt(2*np.pi)*Z)
    stats={}
    for method in methods:
        tv=np.array([m['histogram_TV'] for m in metrics[method]])
        stats[method]=dict(TV_mean=float(tv.mean()),TV_SD=float(tv.std(ddof=1)),
                           TV_of_seed_mean_histogram=float(.5*np.abs(hist[method].mean(0)-target_mass).sum()),
                           TV_old_mean=float(np.mean([m['old_32768_TV'] for m in metrics[method]])),
                           per_seed=metrics[method])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
                         'axes.titlesize':11,'axes.titleweight':'medium','xtick.labelsize':8,'ytick.labelsize':8,
                         'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
                         'xtick.major.width':.7,'ytick.major.width':.7,'pdf.fonttype':42,'ps.fonttype':42,
                         'svg.fonttype':'none','savefig.facecolor':'white'})
    fig,axs=plt.subplots(1,2,figsize=(7.2,2.95),sharex=True,sharey=True)
    fig.subplots_adjust(left=.09,right=.985,bottom=.25,top=.86,wspace=.14)
    colours={'forward':'#087E8B','reverse':'#D55E00'}
    for ax,method,panel in zip(axs,methods,['a','b']):
        density=hist[method]/np.diff(edges)[None]
        mean=density.mean(0);sd=density.std(0,ddof=1);color=colours[method]
        ax.stairs(mean,edges,fill=True,color=color,alpha=.08,lw=0)
        ax.fill_between(edges,np.r_[np.maximum(mean-sd,0),max(mean[-1]-sd[-1],0)],
                        np.r_[mean+sd,mean[-1]+sd[-1]],step='post',color=color,alpha=.24,lw=0)
        ax.stairs(mean,edges,color=color,lw=1.15,zorder=3)
        ax.plot(x,target,color='#222222',lw=1.3,dashes=(4,2.7),zorder=4)
        ax.set_title(f'({panel}) {method.capitalize()} KL',pad=8)
        s=stats[method];txt=rf'$\mathrm{{TV}}={s["TV_mean"]:.4f}\,\pm\,{sd_math(s["TV_SD"])}$'
        ax.text(.965,.965,txt,transform=ax.transAxes,ha='right',va='top',fontsize=8,
                bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=2))
        ax.set(xlim=(-bound,bound),ylim=(0,.9),xlabel='Action')
        ax.set_xticks([-10,-5,0,5,10]);ax.yaxis.set_major_locator(MultipleLocator(.2))
        ax.grid(axis='y',color='#E1E4E8',lw=.55,zorder=0);ax.set_axisbelow(True)
    axs[0].set_ylabel('Probability density')
    legend=[Line2D([0],[0],color='#222222',lw=1.3,dashes=(4,2.7),label='Target'),
            Line2D([0],[0],color='#666666',lw=1.3,label='Policy: seed mean'),
            Patch(facecolor='#999999',alpha=.28,label=r'$\pm$1 SD across seeds')]
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.54,.005),ncol=3,frameon=False,fontsize=8,
               handlelength=2,columnspacing=1.8)
    for ext in ['png','pdf','svg']:
        fig.savefig(args.output/f'final_density_1m.{ext}',dpi=600)
    plt.close(fig)
    np.savez_compressed(args.output/'figure_histograms.npz',edges=edges,target_mass=target_mass,
                        forward_seed_mass=hist['forward'],reverse_seed_mass=hist['reverse'])
    aggregate=dict(sample_count_per_seed=2**20,seeds=[0,1,2,3],checkpoint_step=100000,bins=512,
                   parent_commit=runs['forward_s0']['parent_commit'],analysis_commit=runs['forward_s0']['analysis_commit'],methods=stats)
    (args.output/'SUMMARY.json').write_text(json.dumps(aggregate,indent=2)+'\n')
    caption=r'''**Figure. Forward and reverse KL on a three-mode target.** Left: forward KL; right: reverse KL. Each learned density is the mean of four independently trained policies (seeds 0–3) after 100,000 updates, estimated using $2^{20}$ independently sampled actions per policy and 512 equal-width bins on $[-10,10]$. Shading denotes one standard deviation across seeds; no kernel smoothing is applied. The dashed curve is the exact target density, an equally weighted mixture with means $(-4.25,0,4.25)$ and standard deviation $0.5$, normalized over the action domain. TV annotations report the mean and sample standard deviation of per-seed histogram TV, computed against exact target bin probabilities, rather than TV of the averaged density. Both methods use $N=M=128$ and training batch size 32; reverse KL uses an independent density bank of size $L=2^{20}$. This target was selected in an exploratory search and illustrates the behavior of these estimators in the selected setting.'''
    (args.output/'caption.md').write_text(caption+'\n')
    lines=['# 최종 KL 비교: seed당 2²⁰ action으로 재평가','',
           '![최종 평균 histogram](final_density_1m.png)','',
           '**왼쪽 Forward KL, 오른쪽 Reverse KL.** 각 seed의 최종 100K checkpoint에서 새로 1,048,576개 action을 뽑았다. 네 seed를 동일 가중치로 평균한 histogram을 표시하며, 색 음영은 seed 간 ±1 표준편차다. 선을 부드럽게 만드는 KDE는 사용하지 않았다. PDF·SVG는 벡터 형식, PNG는 600 dpi다.','',
           '## 최종 TV','', '|방법|평균 TV|Seed 간 SD|기존 32,768 표본의 평균 TV|','|---|---:|---:|---:|']
    for method in methods:
        s=stats[method];lines.append(f'|{method.capitalize()} KL|{s["TV_mean"]:.8f}|{s["TV_SD"]:.8f}|{s["TV_old_mean"]:.8f}|')
    lines+=['','|Seed|Forward TV|Reverse TV|','|---:|---:|---:|']
    for seed in range(4):lines.append(f'|{seed}|{metrics["forward"][seed]["histogram_TV"]:.8f}|{metrics["reverse"][seed]["histogram_TV"]:.8f}|')
    lines+=['','## 계산 방식과 평균의 의미','',
            r'$$\widehat p_{s,b}=\frac{\#\{a_i\in I_b\}}{2^{20}},\qquad p_b^\star=\int_{I_b}p^\star(a)\,da,\qquad \widehat{\mathrm{TV}}_s=\frac12\sum_{b=1}^{512}|\widehat p_{s,b}-p_b^\star|.$$', '',
            r'$$\overline{\mathrm{TV}}=\frac14\sum_{s=0}^3\widehat{\mathrm{TV}}_s,\qquad \bar f_b=\frac14\sum_{s=0}^3\frac{\widehat p_{s,b}}{|I_b|}.$$', '',
            '그림은 평균 density이며, 숫자는 **각 seed의 TV를 먼저 계산한 뒤 평균**한 결과다. 서로 다른 seed의 오차가 상쇄될 수 있으므로 평균 density의 TV를 알고리즘 성능으로 대체하지 않았다. 참고로 평균 density 자체의 TV는 다음과 같다.','',
            f'- Forward 평균 density의 TV: {stats["forward"]["TV_of_seed_mean_histogram"]:.8f}.',
            f'- Reverse 평균 density의 TV: {stats["reverse"]["TV_of_seed_mean_histogram"]:.8f}.','',
            '기존과 같은 action 범위·512개 bin을 유지했다. 표본 수 증가로 sampling noise는 줄지만, 유한 bin에 의한 discretization은 남으므로 여기서 TV는 연속 밀도의 정확한 TV가 아닌 histogram TV다. 그림의 음영은 seed 간 변동이며 Monte Carlo 오차의 신뢰구간이 아니다.','',
            '## 논문용 caption','',caption,'',
            '## 재현 및 보관','',
            f'- 학습 source: `{aggregate["parent_commit"]}`.',f'- 이번 평가 source: `{aggregate["analysis_commit"]}`.',
            '- Seed별 checkpoint hash를 평가 전후 확인했으며, 학습 parameter와 optimizer는 변경하지 않았다.',
            '- 새 평가 RNG를 사용했고 action마다 latent와 conditional noise를 독립적으로 다시 뽑았다.',
            '- 원본 action·bin counts·seed별 지표·실행 기록: `dildata:/data1/heejoonorm/OptiQ/studies/20260924_selected_tv1m/`.',
            '- 이번 2²⁰은 **평가 action 수**다. Reverse 학습의 density bank 크기 L=2²⁰과는 별개의 설정이다.','']
    (args.output/'report.md').write_text('\n'.join(lines))
    print(json.dumps(stats,indent=2))


if __name__=='__main__':main()
