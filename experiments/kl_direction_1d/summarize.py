"""Summarize completed training; no quadrature policy-score reference is used."""
import argparse,json,html,base64
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import ndtr
METHODS=['forward_L0','reverse_L128','reverse_L256','reverse_L1024','reverse_L4096']
LABELS=['Forward','Reverse L=128','Reverse L=256','Reverse L=1024','Reverse L=4096']
COLORS=['#2171b5','#d95f02','#7570b3','#1b9e77','#e7298a']
def table(head,rows):return '| '+' | '.join(head)+' |\n|'+'|'.join(['---']*len(head))+'|\n'+'\n'.join('| '+' | '.join(r)+' |' for r in rows)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);figdir=a.out/'figures';figdir.mkdir(exist_ok=True)
 groups={};raw={}
 for m in METHODS:
  groups[m]=[]
  for seed in range(4):
   p=a.data/f'{m}_s{seed}';assert (p/'COMPLETE.json').exists();d=json.loads((p/'metrics_20000.json').read_text());assert d['step']==20000;groups[m].append(d)
   raw[f'{m}_s{seed}']=d
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 x=np.linspace(-1,1,2001);centers=np.array([-.6,0,.6]);Z=np.mean(ndtr((1-centers)/.1)-ndtr((-1-centers)/.1));target=np.exp(-.5*((x[:,None]-centers)/.1)**2).mean(1)/(.1*np.sqrt(2*np.pi)*Z)
 fig,axes=plt.subplots(5,4,figsize=(17,14),sharex=True,sharey=True)
 for i,m in enumerate(METHODS):
  for seed in range(4):
   d=np.load(a.data/f'{m}_s{seed}'/'samples_20000.npz');ax=axes[i,seed];metric=groups[m][seed]
   ax.plot(x,target,'k--',lw=1.2,label='Exact target')
   ax.stairs(d['histogram_mass']/np.diff(d['edges']),d['edges'],color=COLORS[i],lw=1.2,label='32,768 action histogram')
   ax.set(title=f'{LABELS[i]} | seed {seed} | TV={metric["histogram_TV"]:.3f}',xlabel='Action',ylabel='Density');ax.grid(alpha=.15);ax.set_ylim(0,2.7)
 axes[0,0].legend(fontsize=8);fig.suptitle('Final learned densities | N=M=128 | batch=32 | 20,000 updates',fontsize=17)
 fig.tight_layout(rect=[0,0,1,.97]);fig.savefig(figdir/'final_histograms.png',dpi=135);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(12,4.2))
 steps=[0,100,500,1000,2000,5000,10000,20000]
 for i,m in enumerate(METHODS):
  for ax,key in zip(axes,['histogram_TV','basin_TV']):
   values=np.array([[json.loads((a.data/f'{m}_s{s}'/f'metrics_{t:05d}.json').read_text())[key] for t in steps] for s in range(4)])
   ax.plot(steps,values.mean(0),'-o',ms=3,color=COLORS[i],label=LABELS[i]);ax.fill_between(steps,values.mean(0)-values.std(0,ddof=1),values.mean(0)+values.std(0,ddof=1),alpha=.15,color=COLORS[i]);ax.set(xlabel='Actor updates',ylabel=key);ax.grid(alpha=.2)
 axes[0].legend(fontsize=9);fig.suptitle('Training quality | mean ± SD over 4 seeds');fig.tight_layout();fig.savefig(figdir/'training_quality.png',dpi=160);plt.close(fig)
 rows=[];details=[];aggregates={}
 for m,label in zip(METHODS,LABELS):
  rows.append([label]+[f'{np.mean([d[k] for d in groups[m]]):.3f} ± {np.std([d[k] for d in groups[m]],ddof=1):.3f}' for k in ['histogram_TV','basin_TV','wasserstein1','backup_error']])
  aggregates[m]={k:dict(mean=float(np.mean([d[k] for d in groups[m]])),sample_sd=float(np.std([d[k] for d in groups[m]],ddof=1))) for k in ['histogram_TV','basin_TV','wasserstein1','backup_error']}
  for seed,d in enumerate(groups[m]):details.append([label,str(seed),f'{d["histogram_TV"]:.3f}']+[f'{100*v:.1f}%' for v in d['mode_mass']])
 heads=['方法 / Method','Histogram TV ↓','Basin-mass TV ↓','Wasserstein-1 ↓','Backup signed error']
 md='''# TRG 1D 3-mode: Forward / Reverse KL 학습 결과

20/20개 학습이 모두20,000 updates까지 완료됐다. **현재 구현과 설정에서는 Forward가 전체 분포 및 mode별 질량 배분에서 더 좋았다. 다만 Forward도 정확한 세 봉우리 복구에는 실패한 seed가 있다.**

## 설정과 지표

- 목표는 중심(-0.6,0,0.6), 표준편차0.1, 동일 질량의3-component Gaussian mixture를[-1,1]로 제한한 분포다. Q(a)=0.25 log f(a), temperature0.25로 목표를 고정했다.
- Actor는 TRG의 implicit conditional truncated Gaussian,256×256 GELU, log sigma 범위[-5,-1]. N=M=128, batch32, Adam3e-4,20K updates, seed0–3이다.
- Forward는 현재 actor proposal에서 뽑은 후보를 importance weighting한 Direct GMM NLL이다. Reverse는 M개 action과 독립적인 L개 density latent bank로 근사한 score를 사용하는 pathwise update다. Forward는 추가L을 사용하지 않는다.
- 그림은 학습 후 실제 action32,768개를 뽑은 histogram이다. Policy density를 수치 적분한 그림이 아니다. 검은 점선은 알려진 목표 density다.
- Histogram TV:256개 bin에서 actor와 목표의 질량 차이 절댓값을 더한 뒤2로 나눈 값.0이면 일치한다.
- Basin-mass TV:경계(-1,-0.3,0.3,1)로 나눈 세 구역의 총확률 질량만 비교한TV. 각 구역 안의 봉우리 모양은 검사하지 않는다.
- Coverage는 구역 질량이 목표 질량의25% 이상인지를 센다. 모든 run에서3/3이지만, 이것은 세 density peak를 정확히 복구했다는 뜻이 아니다.

## 최종 성능

'''+table(heads,rows)+'''

수치는4seeds의 평균±표본표준편차다. Backup error는 E_actor[Q]−E_target[Q]이며 critic을 학습한 실험이 아니므로 critic bias나 RL return으로 해석하지 않는다. Reverse가 backup scalar에는 더 가까워도, mode별 질량이나 전체 분포는 더 부정확할 수 있다.

![학습 곡선](figures/training_quality.png)

## 실제 학습된 분포 전체

Seed를 섞어 histogram을 만들지 않았다. 한 seed의 왼쪽 집중과 다른 seed의 오른쪽 집중이 평균에서 사라질 수 있기 때문이다.

![전체20개 최종 histogram](figures/final_histograms.png)

## 관찰한 결과

1. Forward의 histogramTV는 평균0.245, basinTV는0.021이다. 세 구역의 질량은 대체로30–36%로 목표의1/3에 가깝다. 하지만3개seed의histogramTV는약0.27로, 봉우리 사이에 남는 넓은 질량 때문에 전체density는 정확하지 않다.
2. Reverse L128/256/1024의histogramTV는평균0.333–0.339다. 대부분한쪽끝구역에약50%, 나머지구역들에약23–27%를배분한다. 끝구역의방향은seed에따라다르다.
3. Reverse L4096 평균TV는0.303이지만 개선은seed2(TV0.207)가주로만든다. 나머지3개는0.331–0.342다. Seed2에서도질량은왼쪽41.7%,가운데16.5%,오른쪽41.8%로균등분배가아니다.
4. Forward는약5K updates이후평균TV가0.25부근에서정체한다. Reverse도대부분2K–5K이후0.33–0.35부근에서크게개선되지않는다. L4096seed2는후반에예외적으로개선된다.
5. L을키우면Reverse의평균성능이일부좋아졌으나,4seeds전체가일관되게좋아졌거나Forward수준의질량배분을회복했다고보기는어렵다.

## Seed별 구역 질량

'''+table(['방법','Seed','Histogram TV','왼쪽 질량','가운데 질량','오른쪽 질량'],details)+'''

## 해석 범위

이표와histogram은실제정책sample과알려진목표분포를비교한것으로,후속policy-score수치적분reference의정확성논의와독립적이다. 다만원인을순수한KL방향차이로확정할수는없다. Reverse의유한L score근사, Forward의유한후보importance weighting,공유신경망최적화가함께작용한다.

후속고정actor진단에서는작은L에서score의bank간변동이큰checkpoint를관찰했다. 그진단의수치적분reference는해상도증가에대한수치적일관성을확인한것이지,신경망integrand의내부적분오차상계를인증한것은아니다. 따라서후속L 임계값도그reference를기준으로한실증값으로읽어야한다.

## 재현

학습source:ae0c370f5d19765089030e62b64f4f0f60a30c37. 전체20개를최종checkpoint에서평가했다. 원본학습·checkpoint·evaluation은dildata:/data1/heejoonorm/OptiQ/studies/20260923_kl_direction_1d/campaign/runtime/runs/에보관한다. 이보고서는새학습을하지않고저장된metrics_20000.json과samples_20000.npz로작성했다.
'''
 md=md.replace('方法 / Method','방법')
 (a.out/'report.md').write_text(md);(a.out/'AGGREGATE.json').write_text(json.dumps(dict(completed=20,aggregates=aggregates,runs=raw),indent=2)+'\n')
 h='<html lang="ko"><meta charset="utf-8"><title>TRG Forward Reverse KL</title><style>body{font-family:system-ui;max-width:1400px;margin:40px auto;line-height:1.65;color:#243047}img{width:100%}pre{white-space:pre-wrap;font-family:inherit;padding:20px;background:#f5f7fa}</style><body><h1>TRG 1D 3-mode: Forward / Reverse KL</h1><p>20개완료. 현재설정에서는Forward가전체분포와질량배분에우세하지만,두방법모두정확한목표분포복구는미달했다. 실제sample평가는policy-score reference와독립적이다.</p>'
 for n in ['training_quality.png','final_histograms.png']:h+='<img src="data:image/png;base64,'+base64.b64encode((figdir/n).read_bytes()).decode()+'">'
 h+='<pre>'+html.escape(md)+'</pre></body></html>';(a.out/'report.html').write_text(h)
 print(json.dumps(aggregates,indent=2))
if __name__=='__main__':main()
