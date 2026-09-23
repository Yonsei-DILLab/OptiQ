"""Forward size comparison; three-peak separation rule prespecified in PROTOCOL."""
import argparse,json,base64,html
from pathlib import Path
import numpy as np
from scipy.special import ndtr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
CENTERS=np.array([-.6,0,.6]);Z=np.mean(ndtr((1-CENTERS)/.1)-ndtr((-1-CENTERS)/.1))
STEPS=[0,100,500,1000,2000,5000,10000,20000]
def target_probability(lo,hi):return np.mean(ndtr((hi-CENTERS)/.1)-ndtr((lo-CENTERS)/.1))/Z

def peaks(samples):
 x=np.asarray(samples).ravel();core=np.array([np.mean((x>=c-.1)&(x<c+.1)) for c in CENTERS]);valley=np.array([np.mean((x>=c-.1)&(x<c+.1)) for c in [-.3,.3]])
 truth=np.array([target_probability(c-.1,c+.1) for c in CENTERS]);ratios=valley/np.maximum(np.minimum(core[:-1],core[1:]),1/len(x))
 return dict(passed=bool(np.all(core>=.5*truth) and np.all(ratios<=.5)),core_mass=core.tolist(),target_core_mass=truth.tolist(),valley_mass=valley.tolist(),valley_core_ratio=ratios.tolist())

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--old',type=Path,required=True);ap.add_argument('--new',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);figdir=a.out/'figures';figdir.mkdir(exist_ok=True)
 roots={128:a.old,256:a.new/'N256M256/runtime/runs',512:a.new/'N512M512/runtime/runs'};data={};diagnostics={};stats={}
 for size,root in roots.items():
  data[size]=[];diagnostics[size]=[]
  for seed in range(4):
   p=root/f'forward_L0_s{seed}';assert (p/'COMPLETE.json').exists();run=json.loads((p/'RUN.json').read_text());c=run['config'];assert c['n']==c['m']==size and c['mean_output_init_scale']==.0001 and c['steps']==20000 and c['batch']==32
   d=json.loads((p/'metrics_20000.json').read_text());assert d['step']==20000;data[size].append(d)
   ds=[dict(step=t,**peaks(np.load(p/f'samples_{t:05d}.npz')['actions'])) for t in STEPS]
   valid=[i for i in range(len(ds)) if all(row['passed'] for row in ds[i:])]
   first=STEPS[valid[0]] if valid else None
   diagnostics[size].append(dict(seed=seed,snapshots=ds,first_persistent_observed_step=first if valid and valid[0]<len(STEPS)-1 else None,final_only=bool(valid and valid[0]==len(STEPS)-1)))
  stats[size]={k:dict(mean=float(np.mean([d[k] for d in data[size]])),sd=float(np.std([d[k] for d in data[size]],ddof=1))) for k in ['histogram_TV','basin_TV','wasserstein1','sigma_mean','backup_error']}
  stats[size]['three_peak_passes']=sum(d['snapshots'][-1]['passed'] for d in diagnostics[size])
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False});x=np.linspace(-1,1,2001);truth=np.exp(-.5*((x[:,None]-CENTERS)/.1)**2).mean(1)/(.1*np.sqrt(2*np.pi)*Z)
 colors={128:'#777777',256:'#1976b5',512:'#14876b'};maxheight=float(truth.max())
 for size,root in roots.items():
  for seed in range(4):
   d=np.load(root/f'forward_L0_s{seed}'/'samples_20000.npz');maxheight=max(maxheight,float((d['histogram_mass']/np.diff(d['edges'])).max()))
 fig,axes=plt.subplots(4,3,figsize=(16,12),sharex=True,sharey=True)
 for col,(size,root) in enumerate(roots.items()):
  for seed in range(4):
   ax=axes[seed,col];d=np.load(root/f'forward_L0_s{seed}'/'samples_20000.npz');ax.plot(x,truth,'k--',lw=1.2,label='Exact target');ax.stairs(d['histogram_mass']/np.diff(d['edges']),d['edges'],color=colors[size],lw=1.2,label='32,768 action histogram')
   passed=diagnostics[size][seed]['snapshots'][-1]['passed']
   ax.set(title=f'{size} x {size} | seed {seed} | TV={data[size][seed]["histogram_TV"]:.3f} | 3 peaks: {passed}',xlabel='Action',ylabel='Density',ylim=(0,maxheight*1.1));ax.grid(alpha=.15)
 axes[0,0].legend(fontsize=8);fig.suptitle('Forward KL | mean-head scale=0.0001 | batch=32 | 20,000 updates',fontsize=16);fig.tight_layout(rect=[0,0,1,.97]);fig.savefig(figdir/'all_final_histograms.png',dpi=150);plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(14,4.3))
 for size,root in roots.items():
  for ax,key in zip(axes[:2],['histogram_TV','basin_TV']):
   ys=np.array([[json.loads((root/f'forward_L0_s{s}'/f'metrics_{t:05d}.json').read_text())[key] for t in STEPS] for s in range(4)])
   ax.plot(STEPS,ys.mean(0),'-o',ms=3,color=colors[size],label=f'{size} x {size}');ax.fill_between(STEPS,ys.mean(0)-ys.std(0,ddof=1),ys.mean(0)+ys.std(0,ddof=1),color=colors[size],alpha=.16);ax.set(xlabel='Actor updates',ylabel=key);ax.grid(alpha=.2)
  ys=[sum(int(d['snapshots'][i]['passed']) for d in diagnostics[size]) for i in range(len(STEPS))];axes[2].plot(STEPS,ys,'-o',color=colors[size],label=f'{size} x {size}')
 axes[2].set(xlabel='Actor updates',ylabel='Seeds passing the 3-peak diagnostic',ylim=(-.1,4.1),yticks=range(5));axes[2].grid(alpha=.2);axes[0].legend();fig.suptitle('4 seeds | error: mean ± SD | peak count: observed snapshots');fig.tight_layout();fig.savefig(figdir/'learning_curves.png',dpi=160);plt.close(fig)
 rows=[];details=[]
 for size in roots:
  q=stats[size]
  rows.append('| '+f'{size}×{size}'+' | '+' | '.join(f'{q[k]["mean"]:.4f} ± {q[k]["sd"]:.4f}' for k in ['histogram_TV','basin_TV','wasserstein1'])+f' | {q["three_peak_passes"]}/4 |')
  for seed,d in enumerate(diagnostics[size]):
   final=d['snapshots'][-1];when=d['first_persistent_observed_step'];tag=f'{when:,}' if when is not None else ('20K에서만 통과' if d['final_only'] else '미확인')
   details.append(f'| {size}×{size} | {seed} | {data[size][seed]["histogram_TV"]:.4f} | '+', '.join(f'{100*v:.1f}%' for v in data[size][seed]['mode_mass'])+' | '+', '.join(f'{v:.3f}' for v in final['valley_core_ratio'])+f' | {tag} |')
 md='''# Forward KL: N=M128 /256 /512에서 세 mode 복구

신규256×256 및512×512의8개 run이20,000 updates까지 완료됐다. 기존128×128의4개 완료 run과 비교한다. 모든 조건의 mean head 초기화 scale은0.0001이다.

## 동일하게 유지한 설정

- 1D latent·action, zero state. Target은중심(-0.6,0,0.6),폭0.1,동일질량 Gaussian mixture를[-1,1]에 제한한 분포.
- Q=0.25 log f,temperature0.25,batch32,Adam3e-4,20K updates,seeds0–3.
- TRG conditional truncated Gaussian,256×256 GELU,log sigma[-5,-1],초기sigma=exp(-1),mean head variance scale1e-4.
- Numerical training 변경은N과M뿐이다. 학생component 수와teacher 후보 수를동시에늘린다. 초기파라미터는크기와무관하지만이후난수배열shape가달라지므로같은seed라고동일trajectory를강제하지않는다.
- Actor/loss/proposal/평가코드는같다. 불필요한추가score진단만생략하며평가RNG와training RNG는분리돼있다.

## 지표 및 '세 mode'의 의미

HistogramTV는256개bin의전체분포오차,basinTV는경계(-1,-.3,.3,1)로구분한세구역의질량배분오차다. 낮을수록좋다. 모든그림은실제32,768개policy action histogram이며policy적분이나KDE smoothing을사용하지않는다.

세구역에질량이있다는것만으로세봉우리가복구됐다고하지않는다. 사전에정한보조판정은각중심의±0.1구간에목표확률의절반이상이모이고,두골짜기구간[-.4,-.2],[.2,.4]의평균밀도가양옆core구간중낮은밀도의절반이하여야한다. 각구간폭은0.2로같다. 이는실용적인봉우리분리기준이며정확한density복구나local maxima개수를수학적으로보장하지않는다. 실제histogram을함께확인한다.

## 최종 결과

| N×M | HistogramTV ↓ | BasinTV ↓ | W1 ↓ | 3-peak 기준통과 |
|---|---|---|---|---|
'''+ '\n'.join(rows)+'''

4seeds평균±표본표준편차다.

![최종분포](figures/all_final_histograms.png)

![학습과세봉우리분리](figures/learning_curves.png)

## Seed별 세부 결과와 시점

| N×M | Seed | 최종HistogramTV | 왼쪽·가운데·오른쪽질량 | 두valley/core ratio | 이후모든저장시점에서통과시작 |
|---|---|---|---|---|---|
'''+ '\n'.join(details)+'''

시점은저장된0,100,500,1000,2000,5000,10000,20000 update에서만판단한다. 최소2개연속저장시점부터끝까지기준을만족하면시작시점을표시한다.20K에서만통과한경우는별도표시한다. 평가시점사이에붕괴나회복이없었다는보장은아니다.

이번비교는N과M을함께바꾸므로학생latent 수와teacher 후보 수각각의기여를분리하지않는다. 모든결과는고정Q actor-fitting이고RL 성능결론으로직접일반화하지않는다.

## 재현과 보관

신규학습commit:e7de5bc4416f5aa0cca0cab4a75cbf26e65019f2. Slurm2314539,array0–7. 기존128baseline commit:ae0c370f5d19765089030e62b64f4f0f60a30c37. Source·checkpoint·원본samples는dildata:/data1/heejoonorm/OptiQ/studies/20260923_forward_sizes/campaign에보관한다. 세봉우리판정규칙은실행전PROTOCOL.md에명시했다.
'''
 (a.out/'report.md').write_text(md);(a.out/'AGGREGATE.json').write_text(json.dumps(dict(completed_new=8,completed_total=12,stats=stats,final_metrics=data,peak_diagnostics=diagnostics),indent=2)+'\n')
 h='<html lang="ko"><meta charset="utf-8"><title>Forward size sweep</title><style>body{font-family:system-ui;max-width:1450px;margin:40px auto;line-height:1.65;color:#243047}img{width:100%}pre{white-space:pre-wrap;font-family:inherit;padding:20px;background:#f5f7fa}</style><body><h1>Forward KL: N=M128 /256 /512</h1>'
 for n in ['all_final_histograms.png','learning_curves.png']:h+='<img src="data:image/png;base64,'+base64.b64encode((figdir/n).read_bytes()).decode()+'">'
 h+='<pre>'+html.escape(md)+'</pre></body></html>';(a.out/'report.html').write_text(h)
 print(json.dumps(dict(stats=stats,persistent_steps={n:[d['first_persistent_observed_step'] for d in ds] for n,ds in diagnostics.items()}),indent=2))
if __name__=='__main__':main()
