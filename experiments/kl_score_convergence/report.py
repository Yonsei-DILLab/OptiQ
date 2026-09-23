"""Build the fixed-actor L convergence report from completed diagnostic runs."""
import argparse,base64,html,json,datetime
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHODS=['forward_L0','reverse_L128','reverse_L256','reverse_L1024','reverse_L4096']
COLORS=['#2171b5','#d95f02','#7570b3','#1b9e77','#e7298a']
def label(name):return name.replace('forward_L0','Forward').replace('reverse_L','Reverse train L=').replace('_s',' / seed ')
def fmtL(value):return f'{value:,}' if value is not None else '미확인'

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True);figdir=out/'figures';figdir.mkdir(exist_ok=True)
    records=[]
    for path in args.data.glob('*/SUMMARY.json'):
        if (path.parent/'COMPLETE.json').exists():records.append((json.loads(path.read_text()),path.parent))
    records.sort(key=lambda v:(METHODS.index(v[0]['name'].rsplit('_s',1)[0]),int(v[0]['name'].rsplit('_s',1)[1])))
    assert records,'No completed diagnostics'
    allpass=all(x['strict_1pct_L'] is not None for x,_ in records)
    strict=max(x['strict_1pct_L'] for x,_ in records) if allpass else None
    practical=max(x['practical_5pct_L'] for x,_ in records) if all(x['practical_5pct_L'] is not None for x,_ in records) else None
    tested=max(x['largest_L'] for x,_ in records)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(5,4,figsize=(17,16),sharex=True,sharey=True)
    for ax in axes.flat:ax.set_visible(False)
    for s,folder in records:
        method,seed=s['name'].rsplit('_s',1);row=METHODS.index(method);ax=axes[row,int(seed)];ax.set_visible(True)
        data=np.load(folder/'scores.npz');ls=data['Ls'];e=data['policy_errors'];grid=data['grid_errors']
        ax.fill_between(ls,np.maximum(e.min(0),1e-8),np.maximum(e.max(0),1e-8),color=COLORS[row],alpha=.2)
        ax.plot(ls,np.maximum(e.mean(0),1e-8),'-o',markersize=3,color=COLORS[row],label='Policy probes: mean / range')
        ax.plot(ls,np.maximum(grid.max(0),1e-8),'--',color='#777',alpha=.8,label='Common grid: worst bank')
        ax.axhline(.01,color='black',ls=':',label='1%');ax.axhline(.05,color='#999',ls=':')
        ax.set(xscale='log',yscale='log',title=label(s['name']),xlabel='Density-bank L',ylabel='Relative score RMSE')
        ax.grid(alpha=.2)
    first=next(ax for ax in axes.flat if ax.get_visible());first.legend(fontsize=8)
    fig.suptitle('Frozen actor, fixed actions | 16 independent banks | shading = min–max',fontsize=17)
    fig.tight_layout(rect=[0,0,1,.97]);fig.savefig(figdir/'all_checkpoints.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,5))
    for k,method in enumerate(METHODS):
        group=[s for s,_ in records if s['name'].rsplit('_s',1)[0]==method]
        if not group:continue
        common=sorted(set.intersection(*[set(r['L'] for r in s['results']) for s in group]))
        values=[max(next(r['relative_RMSE_max'] for r in s['results'] if r['L']==L) for s in group) for L in common]
        ax.plot(common,values,'o-',color=COLORS[k],label=f'{label(method)} ({len(group)} seeds)')
    ax.axhline(.01,color='black',ls=':',label='1%');ax.axhline(.05,color='#999',ls=':',label='5%')
    ax.set(xscale='log',yscale='log',xlabel='Density-bank L',ylabel='Worst observed relative score RMSE',title='Worst bank and checkpoint within each training method')
    ax.legend(fontsize=9);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(figdir/'worst_case.png',dpi=170);plt.close(fig)
    selected=[(s,d) for s,d in records if s['name'].endswith('_s0')]
    fig,axes=plt.subplots(len(selected),1,figsize=(11,3.1*len(selected)),squeeze=False)
    for ax,(s,folder) in zip(axes[:,0],selected):
        d=np.load(folder/'scores.npz');n=json.loads((folder/'RUN.json').read_text())['config']['policy_probes'];a=d['actions'][n:]
        for L,color in zip([128,4096,16384,int(d['Ls'][-1])],['#d95f02','#7570b3','#1b9e77','#2171b5']):
            idx=int(np.where(d['Ls']==L)[0][0]);v=d['scores'][:,idx,n:]
            ax.plot(a,v.mean(0),color=color,lw=1.2,label=f'L={L:,}')
            ax.fill_between(a,v.min(0),v.max(0),color=color,alpha=.12)
        ax.plot(a,d['reference'][n:],'k--',lw=1.4,label='Quadrature reference')
        ax.set(title=label(s['name']),xlabel='Same fixed action',ylabel='Policy action score');ax.grid(alpha=.2)
    axes[0,0].legend(ncol=5,fontsize=9);fig.tight_layout();fig.savefig(figdir/'score_functions.png',dpi=150);plt.close(fig)
    rows=[]
    for s,folder in records:
        e=next(r['relative_RMSE_max'] for r in s['results'] if r['L']==4096)
        q=json.loads((folder/'QUADRATURE.json').read_text())
        rows.append([label(s['name']),f'{100*e:.2f}%',fmtL(s['practical_5pct_L']),fmtL(s['strict_1pct_L']),fmtL(s['grid_strict_1pct_L']),str(q['levels'][-1]['nodes']),f"{100*s['float32_vs_float64_relative_RMSE']:.5f}%"])
    headers=['Checkpoint','L4096 최대 오차','5% 최소 L','1% 최소 L','격자 1% L','적분 nodes','float32 차이']
    table='| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(r)+' |' for r in rows)
    status=f'{len(records)}/20 checkpoints 완료'
    conclusion=f'현재 완료된 checkpoint 모두에서 관찰된 5% 기준은 L={fmtL(practical)}, 엄격한 1% 기준은 L={fmtL(strict)}다.'
    if strict is None:conclusion=f'L을 최대 {tested:,}까지 확인했지만, 완료된 checkpoint 전체를 포괄하는 1% 수렴 기준은 아직 충족되지 않았다. 5% 기준: {fmtL(practical)}.'
    md=f'''# 고정된 actor의 density-score: L은 얼마나 커야 하는가?

{status}. 작성 시각: {datetime.datetime.now().isoformat(timespec='seconds')}.

**{conclusion}** 이 숫자는 아래의 고정된 최종 actor와 action들에서의 실증 기준이며, 다른 차원이나 학습 전체에 대한 보장은 아니다.

## 무엇을 고정했고 무엇을 바꿨나

학습 완료한 TRG actor(20K updates)를 그대로 고정했다. 각 actor의 실제 저장 action 512개와 모든 actor에 공통인 257개 action 격자를 사용했다. 모든 L에서 action은 동일하다. 학습이나 parameter update는 수행하지 않았다.

측정 대상은 목표 Boltzmann 분포가 아니라 **학습된 actor 자신의 score** s(a)=∂logπθ(a)/∂a다. L개 독립 latent로 만든 mixture의 score를 비교한다. 서로 독립인 bank16개를 쓰고, 각 bank 안에서는 L을 늘릴 때 기존 latent를 보존했다.

## 기준값과 통과 조건

1D latent에 대해 Gaussian kernel과 그 action derivative를 직접 수치 적분했다. 적분 격자를 계속 늘려 score 차이가 0.001+0.0001×|score| 이하인 refinement가 연속 두 번 나타나는지 확인했다. latent [-12,12] 밖의 기여 상한도 확인했다. 이는 검증한 수치 reference이며 해석적 정답이라는 뜻은 아니다.

각 bank의 오차는 같은512개 policy action에서 계산한 score RMSE를 reference score RMS로 나눈다. **16개 bank의 관찰된 최대 오차**와 L→2L의 최대 변화가 모두 기준 이하이고, 모든 더 큰 측정 L에서도 유지되는 최소 L을 찾았다. 1%와5%를 함께 보고한다. 이는 confidence bound가 아니다. 262,144에서 엄격한 기준을 찾지 못한 actor는1,048,576까지 확장했다.

공통 action 격자의 결과는 별도 stress test다. policy가 드물게 방문하는 action까지 포함하므로 주평가와 구분한다. 표의 미확인은 최대 측정 범위에서 기준을 만족한 L을 확인하지 못했다는 뜻이다.

![방법별 최대 오차](figures/worst_case.png)

## Checkpoint별 결과

{table}

L4096 최대 오차는16개 bank 중 최대 policy-probe NRMSE다. 적분 nodes는 마지막 reference의 latent 적분점 수다. float32 차이는 같은 최대 bank의 score를float32/float64로 계산한 차이를 reference score RMS로 나눈 값이다.

![전체 checkpoint 수렴](figures/all_checkpoints.png)

## 같은 action에서 실제 score 곡선

각 학습 조건의 seed0을 표시했다. 선은16개 bank 평균, 음영은최솟값–최댓값이다. 음영이 좁아지고 점선 reference에 접근하는지를 함께 확인해야 한다.

![같은 action의 score](figures/score_functions.png)

## 이 결과로 말할 수 있는 범위

이 분석은 density-score의 Monte Carlo 근사가 안정되는 L을 찾는다. score가 안정되었다고 actor가 목표분포를 정확하게 학습한 것은 아니다. 또한 reverse update에는 s(a)−Q′(a)/τ가 들어가므로, 두 항이 거의 상쇄되는 상황에서는 score 자체의 작은 오차도 잔여 gradient에 비해 클 수 있다. SUMMARY.json에 reference direction RMS도 저장했다.

최종 checkpoint에 대한 결과다. 학습 초중반, 새로운 상태, 고차원으로 일반화하려면 별도의 진단이 필요하다. 원본 checkpoint hash와 actor parameter hash를 확인해 분석 중 actor가 변하지 않았음을 검사했다.

분석 source: b1cf6a1d9310df3b3c59b6aaf069d77c09c68817. 학습 source: ae0c370f5d19765089030e62b64f4f0f60a30c37. 원본 자료는 dildata:/data1/heejoonorm/OptiQ/studies/20260923_kl_score_convergence/campaign/runtime/results 에 보관한다.
'''
    (out/'report.md').write_text(md)
    h='<html lang="ko"><meta charset="utf-8"><title>Frozen actor score convergence</title><style>body{max-width:1250px;margin:40px auto;font-family:system-ui;line-height:1.65;color:#213047}img{width:100%;margin:20px 0}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border-bottom:1px solid #ddd}h1{font-size:30px}</style><body>'
    h+=f'<h1>고정 actor: density-score의 L 수렴성</h1><p>{html.escape(status)}</p><p><b>{html.escape(conclusion)}</b></p><p>동일 actor·동일 action, 독립 latent bank16개. 기준 score는 해상도를 높여 검증한1D 수치 적분이다. 주평가는 실제 policy action512개, 공통257개 action 격자는 별도 stress test다. 관찰된 최대 NRMSE와 L→2L 변화가1% 또는5% 이하로 유지되는 최소L을 찾았다.</p>'
    for name in ['worst_case.png','all_checkpoints.png','score_functions.png']:
        h+='<img src="data:image/png;base64,'+base64.b64encode((figdir/name).read_bytes()).decode()+'">'
    h+='<table><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in headers)+'</tr>'
    h+=''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</table><p>미확인: 측정한 최대L까지 기준 미충족. 1%/5%는 실증 오차 기준이며 confidence bound가 아니다. 다른 차원이나 학습 전체에 대한 보장은 아니다. 상세 수식·설정·해석은 같은 폴더의report.md 및실험PROTOCOL.md 참조.</p></body></html>'
    (out/'report.html').write_text(h)
    (out/'AGGREGATE.json').write_text(json.dumps(dict(completed=len(records),strict_1pct_L=strict,practical_5pct_L=practical,largest_L=tested,checkpoints=[s for s,_ in records]),indent=2)+'\n')
    print(json.dumps(dict(completed=len(records),strict_L=strict,practical_L=practical,report=str(out/'report.md'))))

if __name__=='__main__':main()
