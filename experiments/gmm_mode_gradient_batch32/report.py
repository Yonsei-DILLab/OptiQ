"""Live self-contained overview; incomplete runs remain visibly incomplete."""
import json,io,base64,html
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runtime';DEST=OUT/'report'
PLAN=json.loads((Path(__file__).parent/'plan.json').read_text())
LABELS=['Direct GMM','Mode only','Mode + confidence'];COLORS=['#16899d','#8b62aa','#e46b35']
def main():
 DEST.mkdir(parents=True,exist_ok=True);(DEST/'figures').mkdir(exist_ok=True)
 records=[]
 fig,axes=plt.subplots(4,2,figsize=(15,16),sharex=True)
 for ax,(n,m) in zip(axes.flat,PLAN['sizes']):
  x=np.linspace(-1,1,1500);density=sum(np.exp(-.5*((x-c)/.1)**2) for c in [-.6,0,.6])/(3*.1*np.sqrt(2*np.pi))
  ax.plot(x,density,'k--',lw=2,label='Target')
  for method,label,color in zip(PLAN['methods'],LABELS,COLORS):
   for seed in PLAN['seeds']:
    run=OUT/'runs'/f'{method}_N{n}_M{m}_s{seed}';files=sorted(run.glob('eval_*.npz'))
    if not files:continue
    file=files[-1];d=np.load(file);step=int(file.stem.split('_')[-1]);centers=(d['edges'][1:]+d['edges'][:-1])/2
    ax.plot(centers,d['histogram']/np.diff(d['edges']),color=color,alpha=.5,lw=1,label=f'{label} (seed {seed}, {step})')
    records.append(dict(method=method,n=n,m=m,seed=seed,step=step,tv=float(.5*np.abs(d['histogram']-d['target_bin_mass']).sum()),complete=(run/'COMPLETE.json').exists()))
  ax.set(title=f'N={n}, M={m}',xlabel='Action',ylabel='Density',ylim=(0,4));ax.grid(alpha=.2)
  if n==16:ax.legend(fontsize=6)
 fig.suptitle('Latest available checkpoints — unsmoothed 32768-action histograms',fontsize=16);fig.tight_layout();fig.savefig(DEST/'figures/histograms.png',dpi=140);plt.close(fig)
 # All seeds, do not hide a failure behind pooled mean density.
 fig,axes=plt.subplots(4,2,figsize=(15,14))
 for ax,(n,m) in zip(axes.flat,PLAN['sizes']):
  for method,label,color in zip(PLAN['methods'],LABELS,COLORS):
   for seed in PLAN['seeds']:
    p=OUT/'runs'/f'{method}_N{n}_M{m}_s{seed}'/'history.jsonl'
    if not p.exists():continue
    data=[json.loads(x) for x in p.read_text().splitlines()]
    ax.plot([x['step'] for x in data],[x['histogram_tv'] for x in data],color=color,alpha=.65,label=label if seed==0 else None)
  ax.set(title=f'{n} x {m}',xlabel='Update',ylabel='Histogram TV',ylim=(0,1));ax.grid(alpha=.2);ax.legend(fontsize=8)
 fig.tight_layout();fig.savefig(DEST/'figures/tv.png',dpi=140);plt.close(fig)
 rows=['| Method | N×M | Seed | Step | TV | Completed |','|---|---|---|---:|---:|---|']
 for r in records:rows.append(f"| {r['method']} | {r['n']}×{r['m']} | {r['seed']} | {r['step']} | {r['tv']:.4f} | {r['complete']} |")
 text='''# 3-mode Direct GMM: mode별 gradient 선택 — batch 32

**진행 중인 실험의 자동 요약. 서로 다른 step의 값을 최종 성능으로 비교하지 않는다.**

기존 frozen Q, modes (−0.6,0,0.6), 폭0.1, temperature0.25. Actor는 기존 squashed Gaussian,
초기σ0.5, logσ[-5,1], 256×2 GELU, Adam3e-4, batch32,20K updates다.
Baseline, mode 선택만, mode 선택+confidence를 8크기×4seeds에서 비교한다.
H는 각 latent에 배분된 teacher mass 중 각 mode가 차지하는 비율이다. 해당 mode의
μ/logσ gradient만 남긴 뒤 공유 network로 역전파한다. 별도의 gradient norm 보정은 없다.
Confidence도 stop-gradient하며, 원래 marginal NLL gradient를 의도적으로 바꾸는 개입이다.

Density는 각 checkpoint의32768 action histogram,256bins,별도KDE smoothing 없음.
TV는 histogram bin 질량과 target의 정확한CDF bin 질량 차이의 절댓값 합/2다.
같은 색의 가는 선은 네seed. 더 자세한 teacher/H/gradient/latent 이동은 run별 diagnostic 파일에 저장한다.

![최신 checkpoint의 histogram](figures/histograms.png)

![학습 중 TV](figures/tv.png)

'''+ '\n'.join(rows)+'\n'
 (DEST/'report.md').write_text(text)
 content='<h1>3-mode gradient routing — live results</h1><p>Latest checkpoints; incomplete runs are not final comparisons.</p>'
 for file in ['histograms.png','tv.png']:
  image=base64.b64encode((DEST/'figures'/file).read_bytes()).decode();content+=f'<img style="width:100%" src="data:image/png;base64,{image}">'
 content+='<pre>'+html.escape('\n'.join(rows))+'</pre>'
 (DEST/'report.html').write_text('<!doctype html><html><meta charset="utf-8"><title>Mode gradient toy</title><body style="max-width:1500px;margin:auto;font-family:Arial">'+content+'</body></html>')
 (DEST/'summary.json').write_text(json.dumps(records,indent=2)+'\n')
 print('Report',len(records),'/',96,'runs',sum(r['complete'] for r in records),'complete',flush=True)
if __name__=='__main__':main()
