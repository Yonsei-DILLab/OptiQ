"""Paired old/new initializer comparison using actual sampled histograms."""
import argparse,json,html,base64
from pathlib import Path
import numpy as np
from scipy.special import ndtr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--old',type=Path,required=True);ap.add_argument('--new',type=Path,required=True);ap.add_argument('--validation',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);figdir=a.out/'figures';figdir.mkdir(exist_ok=True)
 old=[];new=[]
 for seed in range(4):
  name=f'forward_L0_s{seed}';assert (a.old/name/'COMPLETE.json').exists() and (a.new/name/'COMPLETE.json').exists()
  old.append(json.loads((a.old/name/'metrics_20000.json').read_text()));new.append(json.loads((a.new/name/'metrics_20000.json').read_text()))
  cfg=json.loads((a.new/name/'RUN.json').read_text())['config'];assert cfg['mean_output_init_scale']==1 and cfg['n']==cfg['m']==128 and cfg['batch']==32 and cfg['steps']==20000
 x=np.linspace(-1,1,2001);centers=np.array([-.6,0,.6]);Z=np.mean(ndtr((1-centers)/.1)-ndtr((-1-centers)/.1));target=np.exp(-.5*((x[:,None]-centers)/.1)**2).mean(1)/(.1*np.sqrt(2*np.pi)*Z)
 plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
 maximum=0
 for root in [a.old,a.new]:
  for seed in range(4):
   d=np.load(root/f'forward_L0_s{seed}'/'samples_20000.npz');maximum=max(maximum,(d['histogram_mass']/np.diff(d['edges'])).max())
 fig,axes=plt.subplots(4,2,figsize=(12,12),sharex=True,sharey=True)
 for seed in range(4):
  for col,(root,scale,color,metrics) in enumerate([(a.old,'0.0001','#888888',old),(a.new,'1','#067f91',new)]):
   d=np.load(root/f'forward_L0_s{seed}'/'samples_20000.npz');ax=axes[seed,col]
   ax.plot(x,target,'k--',lw=1.3,label='Exact target');ax.stairs(d['histogram_mass']/np.diff(d['edges']),d['edges'],color=color,lw=1.3,label='32,768 action histogram')
   ax.set(title=f'Mean-head scale={scale} | seed {seed} | TV={metrics[seed]["histogram_TV"]:.3f}',xlabel='Action',ylabel='Density',ylim=(0,maximum*1.1));ax.grid(alpha=.2)
 axes[0,0].legend(fontsize=9);fig.suptitle('Forward KL | only mean-head initialization changed | N=M=128, batch=32, 20K',fontsize=15)
 fig.tight_layout(rect=[0,0,1,.97]);fig.savefig(figdir/'paired_histograms.png',dpi=150);plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(14,4.3));steps=[0,100,500,1000,2000,5000,10000,20000]
 for root,label,color in [(a.old,'scale=0.0001','#888888'),(a.new,'scale=1','#067f91')]:
  for ax,key in zip(axes,['histogram_TV','basin_TV','sigma_mean']):
   ys=np.array([[json.loads((root/f'forward_L0_s{s}'/f'metrics_{t:05d}.json').read_text())[key] for t in steps] for s in range(4)])
   ax.plot(steps,ys.mean(0),'-o',ms=3,color=color,label=label);ax.fill_between(steps,ys.mean(0)-ys.std(0,ddof=1),ys.mean(0)+ys.std(0,ddof=1),color=color,alpha=.16)
   ax.set(xlabel='Actor updates',ylabel=key);ax.grid(alpha=.2)
 axes[0].legend();fig.suptitle('Paired 4 seeds | mean ± SD');fig.tight_layout();fig.savefig(figdir/'learning_curves.png',dpi=160);plt.close(fig)
 stats={}
 for label,ds in [('0.0001',old),('1',new)]:
  stats[label]={k:dict(mean=float(np.mean([d[k] for d in ds])),sd=float(np.std([d[k] for d in ds],ddof=1))) for k in ['histogram_TV','basin_TV','wasserstein1','backup_error','sigma_mean']}
 improvement=(stats['0.0001']['histogram_TV']['mean']-stats['1']['histogram_TV']['mean'])/stats['0.0001']['histogram_TV']['mean']*100
 rows=[]
 for seed,(o,n) in enumerate(zip(old,new)):
  rows.append(f'| {seed} | {o["histogram_TV"]:.4f} | {n["histogram_TV"]:.4f} | {o["basin_TV"]:.4f} | {n["basin_TV"]:.4f} | '+', '.join(f'{100*x:.1f}%' for x in n['mode_mass'])+' |')
 aggregate=[]
 for k in ['histogram_TV','basin_TV','wasserstein1','backup_error','sigma_mean']:
  aggregate.append('| '+k+' | '+' | '.join(f'{stats[l][k]["mean"]:.4f} ± {stats[l][k]["sd"]:.4f}' for l in ['0.0001','1'])+' |')
 validation=json.loads(a.validation.read_text())
 md=f'''# Forward KL: mean head 초기화 scale 0.0001 → 1

4개 seed 모두 20,000 updates까지 완료됐다. 동일 seed의 기존 완료 결과와 비교했다. **Mean head scale만 1로 바꾸는 것으로는 세 mode의 정확한 복구가 재현되지 않았다.** 평균 histogram TV는 기존 0.2455에서 0.2490으로 비슷했고, 1–2개 봉우리를 잡고 나머지를 넓게 덮는 현상이 남았다.

## 변경한 것과 유지한 것

학습상 변경은 mean_output_init_scale 하나다. 이값은 Flax variance_scaling(scale, fan_avg, uniform)에들어간다. 따라서0.0001→1은평균출력층가중치의분산10,000배,표준편차100배를뜻한다. 학습내내mean에상수를곱하는구조가아니라초기화차이다. 네트워크의초기latent별mean다양성과mean경로의초기gradient전달에영향을줄수있다.

초기화검사에서4seeds모두mean kernel을제외한모든파라미터가정확히동일했다. Mean kernel은같은난수의100배였고sigma초기출력도동일했다. 이 검사는 로컬 CPU 초기화 검사다. 실제 GPU run의 update0 평가에서도 mean 분산이 약 9,980–9,997배 증가했고 초기 sigma는 동일했다. 이는 표준편차 약100배 증가와 일치한다. 초기부터 목표의 세 mode에 mean이 배치되었다는 뜻은 아니다.

- 목표: action[-1,1],중심(-0.6,0,0.6),폭0.1,동일질량의Gaussian mixture.
- Q=0.25 log f,temperature0.25,N=M128,batch32,Adam3e-4,20K updates,seeds0–3.
- TRG conditional truncated Gaussian,256×256 GELU,log sigma[-5,-1],초기sigma=exp(-1).
- Actor/loss/teacher/proposal/optimizer코드는이전실험과동일하다. 관련5개수치소스파일의SHA동일성을확인했다.
- 불필요한보조L score진단만생략했다. 그진단은원래도training RNG나파라미터를변경하지않는다. 실제sample평가시점은동일하다.

## 지표 읽는 법

HistogramTV는256개bin에서목표질량과32,768개실제policy sample의질량차이를비교한다. BasinTV는경계(-1,-0.3,0.3,1)의세구역총질량만비교한다. 둘다낮을수록좋다. BasinTV가낮다고구역내peak모양까지정확한것은아니므로histogram을같이본다. 그림에는policy적분이나KDE smoothing을사용하지않았다.

## 결과

| 지표 | scale0.0001 | scale1 |
|---|---|---|
'''+ '\n'.join(aggregate)+'''

4seeds의평균±표본표준편차다. Backup error는E_actor[Q]−E_target[Q]이고학습critic의bias나RL return이아니다. Sigma는conditional Gaussian의underlying scale이다.

| Seed | 기존 HistogramTV | scale1 HistogramTV | 기존 BasinTV | scale1 BasinTV | scale1 왼쪽·가운데·오른쪽 질량 |
|---|---|---|---|---|---|
'''+ '\n'.join(rows)+'''

![동일seed초기화비교](figures/paired_histograms.png)

![학습곡선](figures/learning_curves.png)

## 범위와 재현

현재 확인한 것은 이 초기화 값 하나의 변경으로 팀원이 관찰한 성공을 재현하지 못했다는 사실이다. 팀원의 성공 설정과 actor 구조, latent 설정, sigma 초기화 및 proposal 설정 등을 대조해야 원인을 좁힐 수 있다.

이것은 Forward만의 초기화 ablation이다. Reverse를scale1로재학습하지않았으므로KL방향간최종우열을재평가한결과는아니다. 학습source d3560aaae7d58b0bf561fbf057c8331296b746f9,Slurm2314483. 기존baseline source ae0c370f5d19765089030e62b64f4f0f60a30c37. 4개job모두별도폴더에서실행해기존checkpoint를보존했다.

중앙원본보관:dildata:/data1/heejoonorm/OptiQ/studies/20260923_forward_mean_init1/campaign. 초기화검증:INIT_VALIDATION.json. 보고서:reports/20260923_forward_mean_init1/report.md 및report.html.
'''
 (a.out/'report.md').write_text(md)
 (a.out/'AGGREGATE.json').write_text(json.dumps(dict(completed=4,stats=stats,histogram_tv_reduction_percent=improvement,old=old,new=new,initialization_validation=validation),indent=2)+'\n')
 page='<html lang="ko"><meta charset="utf-8"><title>Forward mean initialization</title><style>body{font-family:system-ui;max-width:1250px;margin:40px auto;line-height:1.7;color:#243047}img{width:100%}pre{white-space:pre-wrap;font-family:inherit;background:#f5f7fa;padding:20px}</style><body><h1>Forward KL: mean head scale0.0001 →1</h1>'
 for n in ['paired_histograms.png','learning_curves.png']:page+='<img src="data:image/png;base64,'+base64.b64encode((figdir/n).read_bytes()).decode()+'">'
 page+='<pre>'+html.escape(md)+'</pre></body></html>';(a.out/'report.html').write_text(page)
 print(json.dumps(dict(stats=stats,reduction=improvement),indent=2))
if __name__=='__main__':main()
