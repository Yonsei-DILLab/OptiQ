"""Reverse-only score convergence with independent, tightly zoomed y axes."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator


def plot(z, output, stem, min_power):
    Ls=z['Ls'];keep=Ls>=2**min_power;Ls=Ls[keep]
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.5),sharex=True)
    limits=[]
    for j,(ax,q) in enumerate(zip(axes,[10,50,90])):
        i=128+j;v=z['scores'][:,keep,i];mu=v.mean(0)
        lo,hi=np.quantile(v,[.1,.9],axis=0)
        refs=z['reference_scores'][:,i];ref=refs.mean();se=refs.std(ddof=1)/np.sqrt(len(refs))
        lower=min(lo.min(),ref-2*se);upper=max(hi.max(),ref+2*se)
        pad=.14*(upper-lower);ylim=(float(lower-pad),float(upper+pad));limits.append(ylim)
        color=f'C{j}'
        ax.axhspan(ref-2*se,ref+2*se,color='.4',alpha=.12,linewidth=0)
        ax.axhline(ref,color='.25',ls=':',lw=1.7,zorder=3)
        ax.fill_between(Ls,lo,hi,color=color,alpha=.28,linewidth=0,zorder=1)
        ax.plot(Ls,mu,color=color,marker='o',ms=4,lw=2.3,zorder=4)
        ax.axvline(2**20,color='.35',ls='--',lw=1.6,zorder=2)
        ax.set_xscale('log',base=2)
        powers=[7,10,14,17,20,24] if min_power==7 else [16,18,20,22,24]
        ax.xaxis.set_major_locator(FixedLocator(2.**np.array(powers)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:r'$2^{%d}$'%round(np.log2(x))))
        ax.set_xlim(2**min_power,2**24);ax.set_ylim(ylim)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.ticklabel_format(axis='y',style='plain',useOffset=False)
        ax.set_title(f'{q}% policy quantile\n'+r'$a=%.3f$'%z['actions'][i],color=color)
        ax.set_xlabel(r'Density-bank size $L$')
        ax.grid(alpha=.18)
    axes[0].set_ylabel(r'Actor score $\widehat{s}_L(a)$')
    legend=[Line2D([0],[0],color='.2',lw=2.3,marker='o',ms=4,label='MC mean (16 repetitions)'),
            Patch(facecolor='.4',alpha=.28,label='10–90% MC range'),
            Line2D([0],[0],color='.25',ls=':',lw=1.7,label=r'Independent $2^{24}$ reference'),
            Line2D([0],[0],color='.35',ls='--',lw=1.6,label=r'Training $L=2^{20}$')]
    fig.legend(handles=legend,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,.015),fontsize=10)
    fig.suptitle('Reverse KL: score convergence (independent y-axis zoom)',fontsize=15,y=.98)
    fig.subplots_adjust(left=.07,right=.98,top=.77,bottom=.24,wspace=.29)
    for ext in ['png','pdf','svg']:fig.savefig(output/f'{stem}.{ext}',dpi=300)
    plt.close(fig)
    return limits


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--commit',required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    path=args.results/'reverse_s0/scores.npz';z=np.load(path)
    assert z['scores'].shape==(16,18,140)
    plt.style.use('default');plt.rcParams.update({'font.size':11,'axes.titlesize':12,
        'axes.labelsize':12,'pdf.fonttype':42,'svg.fonttype':'none'})
    limits=plot(z,args.output,'score_vs_L_reverse_zoom',7)
    late_limits=plot(z,args.output,'score_vs_L_reverse_zoom_large_L',16)
    caption=r'''**Score convergence with independent vertical scales.** A frozen reverse-KL actor (seed 0, 100K updates) is evaluated at its fixed 10th-, 50th-, and 90th-percentile actions. Each panel retains the original score units and uses its own y-axis limits, chosen to contain the full displayed 10–90% Monte Carlo range with padding. Solid curves and shaded bands show the mean and 10–90% range of 16 independent repetitions. Horizontal dotted lines show independent reference means from four banks of size $2^{24}$; these are Monte Carlo estimates, not exact scores. The faint gray band spans the reference mean plus or minus two standard errors. Vertical dashed lines indicate the training density bank size $L=2^{20}$. A supplementary close-up restricts the x-axis to $L\ge2^{16}$ and independently rescales each y-axis. No observations within the shown x-axis ranges are omitted from the reported mean or percentile bands.'''
    (args.output/'caption_reverse_zoom.md').write_text(caption+'\n')
    meta={'plot_commit':args.commit,'source_scores_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
          'actor':'reverse_s0','actions':z['actions'][128:131].tolist(),'training_updates':100000,
          'main_y_limits':limits,'large_L_y_limits':late_limits,'numerical_data_changed':False,
          'y_limits_rule':'10–90% MC envelope plus reference ±2SE, with 14% padding on either side'}
    (args.output/'REVERSE_ZOOM_PROVENANCE.json').write_text(json.dumps(meta,indent=2)+'\n')
    report=args.output/'report.md';s=report.read_text()
    start=s.index('## 측정량')
    text='''# Reverse KL: score 추정 수렴 확대 보기

![세 action별 y축 확대](score_vs_L_reverse_zoom.png)

**세 action을 별도 패널로 나누고, 각 패널의 y축을 score 변동 범위에 맞게 확대했다.** 원래 score 값과 단위를 그대로 사용하며, y축을 0부터 시작시키거나 공유하지 않는다. Seed 0의 100K actor, 세 고정 action, 기존 16회 MC 추정 자료 모두 동일하다.

- 패널 제목: 학습된 policy action의 10%·50%·90% 분위수와 실제 action 값.
- 굵은 실선: 16회 MC 평균. 색 음영: 추정값의 10–90% 구간이며 평균의 신뢰구간이 아니다.
- 가로 점선: 독립 L=2²⁴ bank 4개의 MC 기준 평균. 희미한 회색 띠: 기준 평균 ±2 SE. 정확한 적분값은 아니다.
- 세로 점선: 실제 학습에 사용한 L=2²⁰.

각 y축 범위는 표시된 모든 L의 10–90% 구간과 기준 평균 ±2 SE를 포함한 뒤 위아래 14% 여유를 두어 정했다. **패널마다 y축 범위가 다르므로 그림에서의 높이만으로 오차 크기를 비교하지 않는다.**

[전체 L 범위 PDF](score_vs_L_reverse_zoom.pdf) · [SVG](score_vs_L_reverse_zoom.svg)

## 큰 L 부근을 더 확대

![큰 L에서의 score 수렴](score_vs_L_reverse_zoom_large_L.png)

이 그림은 같은 데이터를 L≥2¹⁶ 구간으로 제한하고 각 y축을 다시 확대했다. L=2²⁰ 주변의 작은 추정 변동을 확인하는 보조 그림이다. L=2⁷부터의 전체 범위는 위 그림에 보존했다.

[큰 L 확대 PDF](score_vs_L_reverse_zoom_large_L.pdf) · [SVG](score_vs_L_reverse_zoom_large_L.svg)

'''
    s=text+s[start:]
    s=s.replace('서로 다른 action들의 score를 한 y축에 표시해 작은 추정 차이는 기존 개별 패널보다 덜 두드러질 수 있다.',
                '확대 그림은 같은 score 자료의 작은 변동을 보여주며, 통계나 score 단위를 변경하지 않는다.')
    cap=s.index('## 논문용 caption');rep=s.index('## 재현',cap)
    s=s[:cap]+'## 논문용 caption\n\n'+caption+'\n\n'+s[rep:]
    s+=f'\n- Y-axis 확대 figure commit: `{args.commit}`. MC 수치는 변경하지 않았다.\n'
    report.write_text(s)
    print(json.dumps(meta,indent=2))


if __name__=='__main__':main()
