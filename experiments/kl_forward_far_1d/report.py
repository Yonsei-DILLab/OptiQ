"""Full-range histograms plus local zooms; no smoothed student densities."""
import argparse,base64,html,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import logsumexp

STEPS=[0,100,500,1000,2000,5000,10000,20000]
SIZES=[64,128,256,512]
def truth(x):
    return np.exp(logsumexp(-.5*((x[:,None]-np.array([-10,0,10]))/.1)**2,axis=1)-np.log(3*.1*np.sqrt(2*np.pi)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);figdir=a.out/'figures';figdir.mkdir(exist_ok=True)
    data={};stats={};details=[];records=[]
    for n in SIZES:
        data[n]=[]
        for s in range(4):
            p=a.root/f'N{n}M{n}/runtime/runs/forward_L0_s{s}'
            assert (p/'COMPLETE.json').exists(),p
            run=json.loads((p/'RUN.json').read_text());c=run['config']
            assert c['n']==c['m']==n and c['mean_output_init_scale']==1 and c['action_bound']==20 and c['target_centers']==[-10,0,10]
            ts=[json.loads((p/f'metrics_{t:05d}.json').read_text()) for t in STEPS]
            arr=np.load(p/'samples_20000.npz');d=ts[-1]
            first=next((t for t,v in zip(STEPS,ts) if v['three_peak_pass']),None)
            persistent=next((STEPS[i] for i in range(len(ts)-1) if all(v['three_peak_pass'] for v in ts[i:])),None)
            data[n].append(dict(path=p,metrics=ts,arrays={k:arr[k] for k in arr.files},first=first,persistent=persistent))
            records.append(dict(n=n,seed=s,first_observed=first,persistent_observed=persistent,metrics=ts))
            details.append(f'| {n}×{n} | {s} | {d["histogram_TV"]:.4f} | '+', '.join(f'{100*x:.1f}%' for x in d['mode_mass'])+' | '+', '.join(f'{100*x:.1f}%' for x in d['core_mass'])+f' | {first if first is not None else "미확인"} | {persistent if persistent is not None else "미확인"} |')
        stats[n]={k:dict(mean=float(np.mean([v['metrics'][-1][k] for v in data[n]])),sd=float(np.std([v['metrics'][-1][k] for v in data[n]],ddof=1))) for k in ['histogram_TV','basin_TV','wasserstein1','sigma_mean','backup_error']}
        stats[n]['passes']=sum(v['metrics'][-1]['three_peak_pass'] for v in data[n])
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#147f9c','#cf6539','#7b51a4','#26956b'];fig,axes=plt.subplots(4,4,figsize=(17,11),sharex=True)
    x=np.linspace(-20,20,16001)
    for col,n in enumerate(SIZES):
        for s,v in enumerate(data[n]):
            ax=axes[s,col];arr=v['arrays'];d=v['metrics'][-1]
            ax.plot(x,truth(x),'k--',lw=1,label='Exact target')
            ax.stairs(arr['histogram_mass']/np.diff(arr['edges']),arr['edges'],color=colors[col],lw=.8,label='32,768 actions')
            ax.set(title=f'{n} x {n} | seed {s} | TV={d["histogram_TV"]:.3f}',xlabel='Action',ylabel='Density',xlim=(-20,20));ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=7);fig.suptitle('Forward KL | centers (-10, 0, 10), width 0.1 | head init scale 1 | 20K updates',fontsize=15)
    fig.tight_layout(rect=[0,0,1,.97]);fig.savefig(figdir/'all_final_histograms.png',dpi=160);plt.close(fig)
    figures=['all_final_histograms.png']
    for n in SIZES:
        fig,axs=plt.subplots(4,3,figsize=(13,10))
        for s,v in enumerate(data[n]):
            arr=v['arrays']
            for col,c in enumerate([-10,0,10]):
                ax=axs[s,col];x=np.linspace(c-.6,c+.6,2001)
                ax.plot(x,truth(x),'k--',label='Exact target');ax.stairs(arr['histogram_mass']/np.diff(arr['edges']),arr['edges'],lw=1,color=colors[SIZES.index(n)],label='Sample histogram')
                ax.set(xlim=(c-.6,c+.6),xlabel='Action',ylabel='Density',title=f'seed {s} | mode {c} | core mass {v["metrics"][-1]["core_mass"][col]:.3f}');ax.grid(alpha=.15)
        axs[0,0].legend(fontsize=7);fig.suptitle(f'{n} x {n}: per-mode zooms | global density scale (no renormalization)',fontsize=14)
        fig.tight_layout(rect=[0,0,1,.97]);name=f'zooms_N{n}.png';fig.savefig(figdir/name,dpi=150);plt.close(fig);figures.append(name)
    fig,axs=plt.subplots(1,3,figsize=(14,4))
    for n,col in zip(SIZES,colors):
        for ax,k in zip(axs[:2],['histogram_TV','basin_TV']):
            ys=np.array([[v[k] for v in d['metrics']] for d in data[n]])
            ax.plot(STEPS,ys.mean(0),'-o',ms=3,label=f'{n} x {n}',color=col)
            ax.fill_between(STEPS,ys.mean(0)-ys.std(0,ddof=1),ys.mean(0)+ys.std(0,ddof=1),alpha=.15,color=col);ax.set(xlabel='Updates',ylabel=k);ax.grid(alpha=.15)
        axs[2].plot(STEPS,[sum(d['metrics'][i]['three_peak_pass'] for d in data[n]) for i in range(len(STEPS))],'-o',ms=3,color=col,label=f'{n} x {n}')
    axs[2].set(xlabel='Updates',ylabel='Seeds passing three-peak diagnostic',ylim=(-.1,4.1),yticks=range(5));axs[0].legend();fig.tight_layout();fig.savefig(figdir/'learning_curves.png',dpi=150);plt.close(fig);figures.append('learning_curves.png')
    rows=[]
    for n,q in stats.items():
        rows.append(f'| {n}×{n} | '+' | '.join(f'{q[k]["mean"]:.4f} ± {q[k]["sd"]:.4f}' for k in ['histogram_TV','basin_TV','wasserstein1'])+f' | {q["passes"]}/4 |')
    total=sum(v['passes'] for v in stats.values());summary=f'16개 run 모두 20,000 updates를 완료했다. 최종 세 봉우리 분리 기준은 총 {total}/16개가 통과했다. 크기별로는 '+', '.join(f'{n}×{n}: {v["passes"]}/4' for n,v in stats.items())+'이다.'
    md=f'''# Forward KL: 중심 (-10, 0, 10)의 좁은 세 mode

{summary}

## 실험 설정

| 항목 | 설정 |
|---|---|
| Target | 중심 (-10, 0, 10), 표준편차 0.1, 동일 질량의 Gaussian mixture |
| Action / mean 범위 | [-20,20]; mean = 20·tanh(raw head) |
| Mean head initialization | variance scaling 1; 초기화만 변경하며 매 update의 별도 multiplier가 아님 |
| Conditional sigma | log sigma ∈ [-5,-1], 초기 sigma = exp(-1); 물리 단위 유지 |
| N=M | 64,128,256,512 |
| Batch / seeds | 32개의 독립 OT 없는 그룹 / 0–3 |
| 학습 | 20K updates, Adam 3e-4, hidden256×256 GELU, latent 1D normal |
| Energy / temperature | Q(a)=0.25 log f(a), tau=0.25 |
| Forward loss | 동일한 finite-mixture proposal에서 추출한 후보에 exp(Q/tau)/proposal weight를 주고 marginal NLL 학습 |

기존 [-1,1] 실험에서 mode 위치와 action 범위는 확대했지만, mode 폭과 sigma 범위는 확대하지 않았다. 단순 좌표 변환으로 같은 문제를 다시 푼 실험이 아니다. Target의 sample이나 mode label은 학습에 사용하지 않는다.

## 그림과 지표를 읽는 법

모든 student 그림은 실제 **32,768개 action의 histogram**이다. Policy density 적분이나 KDE smoothing은 없다. 범위 [-20,20]를 4096 bins로 나눠 bin 폭을 약 0.00977로 유지했다. 점선만 analytic target density다. 확대 그림의 세로축은 전체 분포의 density이며, 각 mode 안에서 다시 정규화하지 않았다.

Histogram TV는 모든 bin의 확률 차이 절댓값 합의 절반이다. Basin TV는 경계 (-20,-5,5,20)로 나눈 세 영역의 질량 차이다. W1은 action 단위의 1-Wasserstein 거리다. 모두 낮을수록 좋다. 넓은 분포도 각 basin에 질량을 둘 수 있으므로 basin TV만으로 mode 복구를 판단하지 않는다.

실행 전에 정한 **세 봉우리 분리 기준**은 (1) 각 중심 ±0.1 core에 목표 core 확률의 절반 이상이 들어가고 (2) mode 중간(-5,5)의 ±0.1 구간 밀도가 양옆 core 중 낮은 밀도의 절반 이하인 것이다. 목표 core 질량은 각각 약 22.76%이므로 최소 약 11.38%가 필요하다. 이는 보조적인 분리 기준이지 정확한 분포 복구를 증명하지 않는다. Histogram과 core mass를 함께 확인해야 한다.

## 최종 결과

| N×M | Histogram TV ↓ | Basin TV ↓ | W1 ↓ | 세 봉우리 기준 |
|---|---|---|---|---|
'''+ '\n'.join(rows)+'''

4 seeds 평균 ± 표본 표준편차다.

![전체 최종 histogram](figures/all_final_histograms.png)

![학습 중 변화](figures/learning_curves.png)

## Mode별 확대

'''+ '\n\n'.join(f'![N=M{n} 확대](figures/zooms_N{n}.png)' for n in SIZES)+'''

## Seed별 결과와 복구 시점

| N×M | Seed | Histogram TV | 세 basin 질량 | 세 core 질량 | 첫 기준 통과 | 이후 저장 시점까지 유지 |
|---|---|---|---|---|---|---|
'''+ '\n'.join(details)+'''

시점은 0,100,500,1000,2000,5000,10000,20000 updates에서만 관찰한다. ‘유지’는 최소 두 저장 시점에 걸쳐 이후 모두 통과한 경우다. 저장 사이의 transient 회복·붕괴까지 검증한 것은 아니다.

## 보관 및 한계

학습 commit: `1153ff9a797173681a980fffea6f2e75414fe40d`, Slurm `2314650` array0–15. Source, config, 전체 actor·Adam·RNG checkpoint, sample 및 독립 teacher probe는 `dildata:/data1/heejoonorm/OptiQ/studies/20260923_forward_far/campaign`에 보관한다.

Mean bound 변경, mode 간격 확대, 작은 mode 폭 유지가 함께 적용된 fixed-Q 실험이다. 이전 실험과의 차이를 mean initialization 하나의 효과로 해석할 수 없다. N과 M도 함께 늘렸으므로 두 효과를 분리하지 않는다.
'''
    (a.out/'report.md').write_text(md)
    (a.out/'AGGREGATE.json').write_text(json.dumps(dict(stats=stats,runs=records),indent=2)+'\n')
    h='<html lang="ko"><meta charset="utf-8"><title>Forward distant modes</title><style>body{font-family:system-ui;max-width:1400px;margin:40px auto;line-height:1.7;color:#243047}img{width:100%}pre{white-space:pre-wrap;font-family:inherit;background:#f5f7fa;padding:20px}</style><body><h1>Forward KL: (-10, 0, 10)</h1><p>'+html.escape(summary)+'</p>'
    for f in figures:h+='<img src="data:image/png;base64,'+base64.b64encode((figdir/f).read_bytes()).decode()+'">'
    h+='<pre>'+html.escape(md)+'</pre></body></html>';(a.out/'report.html').write_text(h)
    print(json.dumps(stats,indent=2))

if __name__=='__main__':main()
