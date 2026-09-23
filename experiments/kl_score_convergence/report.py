"""Self-contained frozen-actor score convergence report; raw scores remain immutable."""
import argparse,base64,html,json,datetime
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
METHODS=['forward_L0','reverse_L128','reverse_L256','reverse_L1024','reverse_L4096']
COLORS=['#2171b5','#d95f02','#7570b3','#1b9e77','#e7298a']
def label(n):return n.replace('forward_L0','Forward').replace('reverse_L','Reverse train L=').replace('_s',' / seed ')
def fmt(x):return f'{x:,}' if x is not None else '범위 내 미확인'
def table(head,rows):return '| '+' | '.join(head)+' |\n|'+'|'.join(['---']*len(head))+'|\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)
def stable_group(ss,key):return max(s[key] for s in ss) if all(s[key] is not None for s in ss) else None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);figdir=a.out/'figures';figdir.mkdir(exist_ok=True)
    records=[]
    for p in a.data.glob('*/SUMMARY.json'):
        if (p.parent/'COMPLETE.json').exists():records.append((json.loads(p.read_text()),p.parent))
    records.sort(key=lambda r:(METHODS.index(r[0]['name'].rsplit('_s',1)[0]),int(r[0]['name'].rsplit('_s',1)[1])))
    assert records
    ss=[s for s,_ in records];strict=stable_group(ss,'strict_1pct_L');practical=stable_group(ss,'practical_5pct_L');tested=max(s['largest_L'] for s in ss)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(5,4,figsize=(17,16),sharex=True,sharey=True)
    for ax in axes.flat:ax.set_visible(False)
    for s,p in records:
        m,seed=s['name'].rsplit('_s',1);row=METHODS.index(m);ax=axes[row,int(seed)];ax.set_visible(True)
        d=np.load(p/'scores.npz');Ls=d['Ls'];e=d['policy_errors'];grid=d['grid_errors']
        ax.fill_between(Ls,np.maximum(e.min(0),1e-8),np.maximum(e.max(0),1e-8),color=COLORS[row],alpha=.2)
        ax.plot(Ls,e.mean(0),'-o',ms=3,color=COLORS[row],label='Policy actions: mean / range')
        ax.plot(Ls,grid.max(0),'--',color='#777',label='Grid: worst bank')
        ax.axhline(.01,color='k',ls=':',label='1%');ax.axhline(.05,color='#999',ls=':',label='5%')
        ax.set(xscale='log',yscale='log',title=label(s['name']),xlabel='Evaluation density-bank L',ylabel='Score NRMSE');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('Frozen checkpoints | same actions for every L | 16 independent banks',fontsize=17)
    fig.tight_layout(rect=[0,0,1,.97]);fig.savefig(figdir/'all_checkpoints.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5.6))
    for k,m in enumerate(METHODS):
        group=[(s,p) for s,p in records if s['name'].rsplit('_s',1)[0]==m]
        for i,(s,p) in enumerate(group):
            d=np.load(p/'scores.npz');ax.plot(d['Ls'],d['policy_errors'].max(0),'-o',ms=3,color=COLORS[k],alpha=.7,label=label(m) if i==0 else None)
    ax.axhline(.01,color='k',ls=':',label='1%');ax.axhline(.05,color='#777',ls=':',label='5%')
    ax.set(xscale='log',yscale='log',xlabel='Evaluation density-bank L',ylabel='Worst-bank score NRMSE',title='Each curve = one frozen checkpoint (max over 16 banks)')
    ax.grid(alpha=.2);ax.legend(ncol=2,fontsize=9);fig.tight_layout();fig.savefig(figdir/'worst_case.png',dpi=160);plt.close(fig)
    chosen=['forward_L0_s0','reverse_L128_s0','reverse_L128_s2']
    selected=[(s,p) for name in chosen for s,p in records if s['name']==name]
    fig,axes=plt.subplots(len(selected),1,figsize=(11,3.8*len(selected)),squeeze=False)
    for ax,(s,p) in zip(axes[:,0],selected):
        d=np.load(p/'scores.npz');n=512;x=d['actions'][n:]
        ls=list(dict.fromkeys([4096,65536,int(d['Ls'][-1])]))
        for L,c in zip(ls,['#d95f02','#7570b3','#1b9e77']):
            j=int(np.where(d['Ls']==L)[0][0]);v=d['scores'][:,j,n:]
            ax.plot(x,v.mean(0),color=c,lw=1,label=f'L={L:,}')
            ax.fill_between(x,v.min(0),v.max(0),color=c,alpha=.18)
        ax.plot(x,d['reference'][n:],'k--',lw=1.3,label='Quadrature reference')
        ax.set(title=label(s['name']),xlabel='Same fixed action',ylabel='Policy action score');ax.grid(alpha=.2);ax.legend(ncol=4,fontsize=9)
    fig.tight_layout();fig.savefig(figdir/'score_functions.png',dpi=160);plt.close(fig)
    sigma_records=[(s,p,json.loads((p/'SIGMA.json').read_text())) for s,p in records if (p/'SIGMA.json').exists()]
    fig,ax=plt.subplots(figsize=(8,5))
    for k,m in enumerate(METHODS):
        group=[(s,p,g) for s,p,g in sigma_records if s['name'].rsplit('_s',1)[0]==m]
        if not group:continue
        x=[g['fraction_below_001']*100 for s,p,g in group];y=[100*next(r['relative_RMSE_max'] for r in s['results'] if r['L']==4096) for s,p,g in group]
        ax.scatter(x,y,color=COLORS[k],label=label(m),s=55)
    ax.set(xlabel='Latent fraction with conditional sigma < 0.01 (%)',ylabel='Worst-bank score NRMSE at L=4096 (%)',title='Narrow conditional kernels coincide with harder score estimation')
    ax.legend(fontsize=9);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(figdir/'sigma_association.png',dpi=160);plt.close(fig)
    group_rows=[]
    for m in METHODS:
        group=[s for s in ss if s['name'].rsplit('_s',1)[0]==m]
        if not group:continue
        es=[100*next(r['relative_RMSE_max'] for r in s['results'] if r['L']==4096) for s in group]
        group_rows.append([label(m),f'{min(es):.2f}–{max(es):.2f}%',fmt(stable_group(group,'practical_5pct_L')),fmt(stable_group(group,'strict_1pct_L'))])
    detail=[]
    for s,p in records:
        q=json.loads((p/'QUADRATURE.json').read_text());e=next(r['relative_RMSE_max'] for r in s['results'] if r['L']==4096)
        detail.append([label(s['name']),f'{100*e:.2f}%',fmt(s['practical_5pct_L']),fmt(s['strict_1pct_L']),fmt(s['largest_L']),f"{100*s['results'][-1]['relative_RMSE_max']:.2f}%",fmt(q['levels'][-1]['nodes'])])
    hard=[s for s in ss if s['strict_1pct_L'] is None]
    narrow=[(s,g) for s,p,g in sigma_records if g['fraction_below_001']>.4]
    precision=max(s['float32_vs_float64_relative_RMSE'] for s in ss)*100
    status=f'{len(records)}/20개 최종 checkpoint 분석 완료'
    conclusion=f'전체 actor에 대한 5% 기준은 L={fmt(practical)}까지 필요했다. 1% 기준은 {len(hard)}개 checkpoint에서 최대 측정 범위까지 확인하지 못했다.'
    if strict is not None:conclusion=f'전체 actor에서 5% 기준은 L={fmt(practical)}, 1% 기준은 L={fmt(strict)}까지 필요했다.'
    h1=['학습한 actor','L=4096 오차 범위(4 seeds)','4 seeds 모두 5%','4 seeds 모두 1%']
    h2=['Actor','L4096 오차','5% 최소 L','1% 최소 L','측정 최대 L','최대 L 오차','Reference nodes']
    md=rf'''# 고정 actor의 score 수렴: L은 얼마나 커야 하는가?

{status}. 작성 시각: {datetime.datetime.now().isoformat(timespec='seconds')}.

**L=4096을 충분히 정확한 score 근사라고 보기 어렵다.** 필요한 L은 학습된 actor에 따라 크게 달랐다. {conclusion}

{table(h1,group_rows)}

표의 오차는 각 checkpoint에서 독립 bank16개 중 최대 score NRMSE다. 마지막 두 열은 해당 학습 조건의4 seeds를 모두 만족시키는 기준이다. 학습에 사용한 L과 이번 평가의 density-bank L은 서로 다른 값이다. 예를 들어 `Reverse train L=128`도 평가 때는 최대 수백만 latent를 사용한다. **1%/5%는 우리가 정한 실증 오차 기준이며, 이론적 보장이나 confidence bound가 아니다.**

![고정 actor별 L과 score 오차](figures/worst_case.png)

## 1. 무엇을 고정하고 비교했나

직전의3-mode 1D TRG 실험에서 학습이 완료된20개 actor를 사용했다. Forward4seeds와 Reverse 학습L=128/256/1024/4096 각각4seeds다. 각 actor는 N=M=128, batch32, temperature0.25로20,000 updates 학습했다. 이번 진단에서는 actor·optimizer를 전혀 갱신하지 않았다.

- 각 checkpoint에 저장된 실제 policy action 중 첫512개를 고정했다. 주평가는 이512개에서 수행했다.
- 모든 actor에 공통인[-0.999,0.999]의257개 격자 action도 검사했다. 이는 드물게 방문하는 action까지 포함하는 별도 stress test다.
- **각 actor 내에서는 모든 L과 bank에서 동일한 action을 사용했다.** Actor 사이의 주평가 action은 각각의 policy 표본이고, 공통 격자만 actor 사이에도 동일하다.
- 서로 독립인 latent bank16개를 사용했다. 각 bank에서는 L을 늘릴 때 이전 latent를 유지하는 nested-prefix 방식이다.
- L=128부터262,144까지2배씩 증가시켰다.1% 판정이 안 된 actor는1,048,576까지, reference를 확정한 뒤에도 미통과인6개만 최대8,388,608까지 사후 확장했다. 범위를 확장했지만 판정 tolerance는 바꾸지 않았다.

대상 actor는 squashed Gaussian이 아니라 현재 TRG의 **box-truncated Gaussian conditional mixture**다. Mean은[-1,1]이고 conditional standard deviation은 $e^{{-5}}\leq\sigma\leq e^{{-1}}$다. Sampled action에서의 score는 다음과 같다.

$$
\pi_\theta(a)=\int k_\theta(a\mid z)\varphi(z)\,dz,\qquad
s_\theta(a)=\partial_a\log\pi_\theta(a).
$$

$$
\widehat s_L(a)=\frac{{\sum_{{\ell=1}}^L k_\theta(a\mid z_\ell)(\mu_\ell-a)/\sigma_\ell^2}}{{\sum_{{\ell=1}}^L k_\theta(a\mid z_\ell)}}.
$$

Conditional truncation normalizer는 mixture 가중치에 정확하게 포함했다. Action에 대한 미분에서는 상수다. **여기서 비교하는 것은 목표 Boltzmann의 score가 아니라 학습된 actor 자신의 score**다. 따라서 actor가 목표분포에 잘 맞지 않아도 그 actor의 score 추정 수렴은 독립적으로 평가할 수 있다.

## 2. 큰 L 하나를 정답이라고 가정하지 않았다

1D latent이므로 별도의 수치 적분으로 기준값을 만들었다.

$$
s_{{\mathrm{{ref}}}}(a)=\frac{{\int k_\theta(a\mid z)(\mu_\theta(z)-a)\sigma_\theta(z)^{{-2}}\varphi(z)\,dz}}{{\int k_\theta(a\mid z)\varphi(z)\,dz}}.
$$

Latent[-12,12]에서 composite16-point Gauss–Legendre 적분을 사용했다. 모든769개 action에서, 해상도를 높였을 때 score 차이가 $0.001+0.0001|s_{{\mathrm{{ref}}}}(a)|$ 이하인 경우가 연속 두 번 나올 때까지 해상도를 높였다.20개 모두 이 기준을 통과했다. 구간 밖 Gaussian latent tail의 score 오차 상한도1e-5 미만인지 확인했다. **검증된 수치 reference이며 해석적 exact solution은 아니다.**

Actor forward와 파라미터는 원래float32 그대로 두고, kernel과 합산만float64를 사용했다. 원래 최대 bank의float32 재계산과 비교한 NRMSE 차이는 최대 {precision:.5f}%였다. 따라서 아래의 수%–수십% 오차를 단순한 누적 roundoff로 설명하기는 어렵다. 추가8M bank는float64로 계산했으며, float32 비교는 확장 전 최대L에서 수행했다.

## 3. 수렴의 정의

고정 policy action 집합을 $\mathcal A$라고 하면,

$$
S=\sqrt{{\frac1{{|\mathcal A|}}\sum_{{a\in\mathcal A}}s_{{\mathrm{{ref}}}}(a)^2}},\qquad
E_r(L)=\frac{{\sqrt{{\frac1{{|\mathcal A|}}\sum_{{a\in\mathcal A}}(\widehat s_{{L,r}}(a)-s_{{\mathrm{{ref}}}}(a))^2}}}}{{S}}.
$$

Score가0인 개별 action에서 상대오차를 나누지 않고, 전체 reference RMS로 정규화했다. L→2L 간 score 변화도 같은S로 나눴다. 다음 조건을 모두 만족하는, **측정한 L 중 최소값**을 보고한다.

1. 16개 bank 중 최대 $E_r(L)$이1% 또는5% 이하다.
2. 같은 bank에서 L→2L score 변화도 같은 기준 이하다.
3. 더 큰 모든 관찰L에서도 위 기준을 유지한다. 적어도 하나의 더 큰L이 있어야 판정한다.

따라서 마지막L 하나가 우연히 기준 아래에 들어온 경우를 수렴 판정으로 사용하지 않는다. 이 기준은 각 actor의 관찰 범위에 대한 것이며, 모든 미래 IID bank에서 그 오차를 보장하지는 않는다.

## 4. Checkpoint별 결과

{table(h2,detail)}

`범위 내 미확인`은 측정 범위와 위의 추가L 검증 조건에서 통과점을 확인하지 못했다는 뜻이다. 낮은L에서 이미 통과한 actor는 모두8M까지 재계산하지 않았다. Reference nodes는 latent 수치 적분점 수로, Monte Carlo L과 구분한다.

![20개 actor의 수렴 곡선](figures/all_checkpoints.png)

각 패널의 실선은16개 bank 평균, 음영은 최솟값–최댓값, 회색 점선은 공통 격자에서의 최대 오차다. 주평가와 stress test를 합쳐서 한 숫자로 요약하지 않았다.

## 5. 같은 action에서 score가 어떻게 달라지는가

대표로 Forward seed0, Reverse trainL128 seed0, 같은 조건의 어려운 seed2를 나란히 봤다. 아래 그림은 모두 동일257개 격자에서 계산했다. 실선은bank 평균, 음영은bank간 범위, 검은 점선은수치 적분 reference다. 큰L에서 음영이 좁아지는 것과 reference에 접근하는 것을 함께 봐야 한다.

![동일 action의 score 곡선](figures/score_functions.png)

## 6. 왜 actor마다 필요한 L이 크게 다른가

모든 actor에 공통된 별도65,536개 latent를 넣어 conditional sigma를 조사했다. 어려운 reverse actor5개에서는 sigma<0.01인 latent가 약43–64%를 차지했고, 상당수가 하한 $e^{{-5}}\simeq0.00674$에 있었다. Forward actor에서는 sigma<0.01인 latent가 없었다. 정확한 비율은각결과의SIGMA.json에 저장했다.

![Conditional sigma와 score 근사 난이도](figures/sigma_association.png)

Score는 단순한 density 평균이 아니라 density로 가중한 $(\mu-a)/\sigma^2$의 비율 추정이다. Conditional kernel이 좁으면 같은 action의 density와 미분에 실질적으로 기여하는 latent가 적어질 수 있어, bank 재표집에 민감해지는 것으로 해석할 수 있다. **지금 확인한 것은 sigma와 오차의 동반 변화다. Sigma 하한을 바꾼 인과 실험을 한 것은 아니다.**

## 7. Forward와 Reverse 비교 실험에 주는 의미

현재 결과는 **L=128–4096만으로 reverse score의 근사 오차를 충분히 제거했다고 주장하기 어렵다**는 뜻이다. 특히 어려운 최종 actor에서는4096개 bank로 계산한 score가 reference와 크게 달랐다. 이 상태에서 얻은 forward/reverse 성능 차이를 KL 방향만의 차이로 해석해서는 안 된다.

1D 실험에서 근사 오차를 거의 제거한 대조군이 필요하다면, latent 수치 적분으로 검증한 score를 쓰는 방법이 가장 직접적이다. 다만 이번 작업은 고정 actor의 진단이며, 수치 적분 score로 새로운 학습을 수행한 결과는 아니다. 실제 학습에는 actor가 계속 변하므로 초기·중기 checkpoint도 검사해야 한다.

또한 reverse policy update의 action 방향에는 $s_\theta(a)-Q'(a)/\tau$가 들어간다. 두 항이 거의 상쇄되면 score 자체의1% 오차도 남은 gradient 대비 클 수 있다. 이번1%/5% 기준이 parameter gradient나 학습 성능의 동일한 오차를 보장하지는 않는다. SUMMARY.json에 reference 방향의RMS도 함께 저장했다.

이 결과를 다른 차원·상태·학습시점에 보편적인 충분L로 일반화하지 않는다. 고정 actor의 수렴과 목표 Boltzmann 분포를 정확히 학습했는지도 별개다.

## 8. 재현과 보관

- 학습 source: `ae0c370f5d19765089030e62b64f4f0f60a30c37`.
- 최초 진단 source: `b1cf6a1d9310df3b3c59b6aaf069d77c09c68817`, Slurm2314179.
- 기준값 정밀화: `74581d6310bd65442430ee033280d2a5f34e2532`, Slurm2314234. 원래Monte Carlo score는변경하지 않았다.
- 추가L 진단: `d86d9123804fc5e2d1500fdfe8e2773d74f5c389`, Slurm2314318. 원래score 배열보존과동일prefix 재계산일치를검증했다.
- 각단계checkpoint SHA와actor parameter hash를검증했다. Source/config/protocol은실행전heejoon에commit·push했다.
- 원본자료: `dildata:/data1/heejoonorm/OptiQ/studies/20260923_kl_score_convergence/campaign/runtime/extended_results/`.
- `results/`는최초진단, `refined_results/`는reference정밀화, `extended_results/`는추가L을포함한최종본이다. 이전자료를덮어쓰지않았다.
'''
    (a.out/'report.md').write_text(md)
    h='<html lang="ko"><meta charset="utf-8"><title>Frozen policy score convergence</title><style>body{max-width:1250px;margin:40px auto;padding:0 20px;font-family:system-ui;line-height:1.7;color:#243047}img{width:100%;margin:20px 0}table{border-collapse:collapse;font-size:13px;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}h1{font-size:30px}pre{white-space:pre-wrap;background:#f5f7fa;padding:20px;font-family:inherit}</style><body>'
    h+=f'<h1>고정 actor의 score 수렴: L은 얼마나 커야 하는가?</h1><p>{html.escape(status)}</p><p><b>L=4096을 충분히 정확한 근사라고 보기 어렵다. {html.escape(conclusion)}</b></p><p>학습된 actor를 고정하고 동일한 action에서16개 독립 latent bank를 비교했다. 기준값은 검증한1D latent 수치 적분이다. 실제policy action512개에서의최대bank NRMSE와 L→2L 변화를 모두1%/5% 기준으로 검사했다. 최종checkpoint에 대한 실증 기준이며 학습전체의보장은 아니다.</p>'
    for head,rows in [(h1,group_rows),(h2,detail)]:
        h+='<table><tr>'+''.join('<th>'+html.escape(v)+'</th>' for v in head)+'</tr>'
        h+=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table><br>'
    for name,caption in [('worst_case.png','각 선은 하나의 고정 checkpoint. 16banks 중 최대 오차.'),('score_functions.png','같은 격자 action의 score. 선:평균, 음영:bank간 범위, 검은 점선:수치 적분 reference.'),('sigma_association.png','Sigma와 근사 난이도의 연관성. 인과관계를 입증한 실험은 아니다.'),('all_checkpoints.png','20개 actor 전체. 실선·음영:policy action; 회색:공통 격자 stress test.')]:
        h+='<p>'+html.escape(caption)+'</p><img src="data:image/png;base64,'+base64.b64encode((figdir/name).read_bytes()).decode()+'">'
    h+='<h2>전체 설명과 수식 원문</h2><p>수식이 포함된 Markdown 원문은 같은 폴더 report.md에 있다. 아래에도 원문을 보존했다.</p><pre>'+html.escape(md)+'</pre></body></html>'
    (a.out/'report.html').write_text(h)
    (a.out/'AGGREGATE.json').write_text(json.dumps(dict(completed=len(records),strict_1pct_L=strict,practical_5pct_L=practical,largest_L=tested,checkpoints=ss),indent=2)+'\n')
    print(json.dumps(dict(completed=len(records),strict_L=strict,practical_L=practical,report=str(a.out/'report.md'))))
if __name__=='__main__':main()
