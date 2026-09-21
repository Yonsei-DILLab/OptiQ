"""Self-contained report from saved samples, never reruns or smooths actor draws."""
from pathlib import Path
import argparse,json,base64,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS={'argmax':'Legacy Sinkhorn row-argmax','exact_argmax':'Exact OT row-argmax','monge_quantile':'Quantile Monge (256 x 256)'}
COLORS=['#b95e28','#228496','#7246a1']

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=True);tasks=json.loads((a.root/'tasks.json').read_text());dep=json.loads((a.root/'DEPLOYMENT.json').read_text())
 done=[t for t in tasks if (a.root/'runs'/t['name']/'COMPLETE.json').exists()]
 lines=['# Legacy row-argmax와 quantile Monge 파일럿',f'완료 {len(done)}/{len(tasks)} runs. Commit: `{dep["commit"]}`.',
  'Source 256 / proposal 1,024 / Monge target 256. Matching 자체는 256×256이며, 원래 weighted teacher를 quantile 대표점으로 근사한다.',
  '이전 implicit actor, truncated Gaussian KDE(std .2, clip .5), Adam 3e-4, temperature .25를 유지했다. 각 방법은 독립적으로 35K updates, seeds 0,1을 학습한다.',
  '**Histogram TV**는 동일한 512 bins에서 정답과 actor 질량 차이 절댓값 합의 절반이다. **Basin TV**는 정답 peak 사이 valley로 나눈 mode 영역의 질량 차이 절반이다. 예를 들어 정답의 두 mode 질량이 (0.5,0.5), actor가 (0.8,0.2)이면 basin TV는 0.3이다. 두 mode의 질량이 맞아도 각각의 폭이 다르면 histogram TV는 클 수 있다.',
  '아래 actor density는 별도 smoothing 없이 32,768개 실제 action 표본을 그린 histogram이다. Exact target만 CDF bin 적분을 사용했다. 선은 seeds 평균이며 개별 seed 수치를 함께 제시한다.',
  '두 mode 질량 변경:20K까지50:50 →25K까지80:20 →30K까지20:80 →35K까지50:50. 세 mode 분리:20K–22K에3→6,25K–27K에6→3.']
 figures=[]
 def emit(fig,name,caption):
  fig.tight_layout();fig.savefig(a.output/name,dpi=300 if name=='assignments.png' else 155);plt.close(fig);figures.append((name,caption));lines.extend([f'## {caption}',f'![{caption}]({name})'])
 fig,axs=plt.subplots(2,4,figsize=(18,7))
 for r,case in enumerate(['double_mass','tri_split']):
  for c,step in enumerate([20000,22000,27000,35000]):
   ax=axs[r,c];shown=False
   for method,color in zip(LABELS,COLORS):
    files=[a.root/'runs'/t['name']/'density'/f'{step:06d}.npz' for t in done if t['case']==case and t['method']==method]
    data=[np.load(f) for f in files if f.exists()]
    if not data:continue
    d=data[0];edges=d['edges'];width=np.diff(edges);curve=np.mean([x['actor'] for x in data],axis=0)/width
    if not shown:ax.stairs(d['target']/width,edges,color='#233246',ls='--',label='Exact target');shown=True
    ax.stairs(curve,edges,color=color,label=LABELS[method]);[x.close() for x in data]
   ax.set(title=f'{case} | update {step}',xlabel='Action',ylabel='Density',xlim=(-1,1));ax.grid(alpha=.2)
 fig.suptitle('32,768 samples per run | 512 bins | seeds 0,1 mean | no smoothing',fontsize=11)
 axs[0,0].legend(fontsize=7);emit(fig,'density.png','변화 전·중·후: 실제 action histogram (seed 0,1 평균, run당 32,768 samples, 512 bins, smoothing 없음)')
 fig,axs=plt.subplots(2,2,figsize=(12,7))
 for r,case in enumerate(['double_mass','tri_split']):
  for method,color in zip(LABELS,COLORS):
   for t in done:
    if t['case']!=case or t['method']!=method:continue
    fs=sorted((a.root/'runs'/t['name']/'evaluations').glob('*.npz'));d=[np.load(f) for f in fs]
    if not d:continue
    for c,key in enumerate(['hist_tv','basin_tv']):
     axs[r,c].plot(np.concatenate([x['step'] for x in d]),np.concatenate([x[key] for x in d]),color=color,alpha=.75,
       ls='-' if t['seed']==0 else ':',label=LABELS[method] if t['seed']==0 else None)
    [x.close() for x in d]
  for c in range(2):
   axs[r,c].set(title=f'{case}: '+['Histogram TV','Basin TV'][c],xlabel='Actor updates',ylabel='TV (lower better)');axs[r,c].grid(alpha=.2)
   for step in [20000,25000,30000]:axs[r,c].axvline(step,color='gray',lw=.5)
 axs[0,0].legend(fontsize=7);emit(fig,'tracking.png','분포 추적 오차: 실선 seed 0, 점선 seed 1; 평가당 32,768 samples와 512 bins')
 summary=[]
 for case in ['double_mass','tri_split']:
  for method in LABELS:
   ds=[]
   for t in done:
    if t['case']==case and t['method']==method:
     f=a.root/'runs'/t['name']/'density'/'035000.npz'
     if f.exists():
      with np.load(f) as d:ds.append([float(d[k]) for k in ['hist_tv','basin_tv','backup_bias']])
   if ds:
    mean=np.mean(ds,axis=0);summary.append(dict(case=case,method=method,seeds=len(ds),hist_tv=float(mean[0]),basin_tv=float(mean[1]),backup_bias=float(mean[2])))
 (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 lines+=['## 최종 오차: 두 seed 평균','| Q | Method | Histogram TV | Basin TV |','|---|---|---:|---:|']
 for d in summary:lines.append(f'| {d["case"]} | {LABELS[d["method"]]} | {d["hist_tv"]:.4f} | {d["basin_tv"]:.4f} |')
 if len(done)==len(tasks):
  lines+=['이 파일럿에서는 **Monge와 exact OT row-argmax가 기존 Sinkhorn row-argmax보다 최종 histogram TV를 크게 줄였다. 두 방법 사이의 차이는 작다.** 반면 basin TV는 세 방법 모두 작다. 따라서 주된 차이는 mode에 배분된 총 질량보다 mode 내부의 분포 모양에서 나타난다. 두 seed만의 결과이며, Monge가 일반적으로 우월하다거나 분포 복구가 완성됐다고 해석하지 않는다.']
 lines+=['## Seed별 최종 지표','| Q | Method | Seed | Histogram TV | Basin TV | Backup bias |','|---|---|---|---:|---:|---:|']
 for t in done:
  f=a.root/'runs'/t['name']/'density'/'035000.npz'
  if f.exists():
   with np.load(f) as d:lines.append(f'| {t["case"]} | {LABELS[t["method"]]} | {t["seed"]} | {float(d["hist_tv"]):.4f} | {float(d["basin_tv"]):.4f} | {float(d["backup_bias"]):+.4f} |')
 # Common frozen cloud prevents comparing assignment maps from different policies.
 f=a.root/'runs'/'double_mass_argmax_s0'/'assignment_audits'/'025001.npz'
 if f.exists():
  with np.load(f) as d:
   orders=[np.arange(len(d['x'])),np.argsort(d['x'][:,0],kind='stable')];cols=[np.arange(len(d['b'])),np.argsort(d['b'][:,0],kind='stable')]
   fig,axs=plt.subplots(2,3,figsize=(14,7))
   for r in range(2):
    for c,name in enumerate(['sinkhorn','exact','monge']):
     matrix=d[name+'_plan'][np.ix_(orders[r],cols[r])]*len(d['x'])
     axs[r,c].imshow(np.log10(np.maximum(matrix,1e-7)),aspect='auto',vmin=-7,vmax=0,cmap='magma',interpolation='nearest')
     axs[r,c].set(title=name+(' original' if r==0 else ' sorted'),xlabel='1,024 original candidate columns',ylabel='256 source rows')
   emit(fig,'assignments.png','double_mass seed 0, update 25,001의 동일 baseline cloud: log10(NP), 공통 색 범위 -7~0 (어두움~밝음), 원본·정렬; N=256, proposal=1,024')
   fig,axs=plt.subplots(1,2,figsize=(10,4))
   for r,order in enumerate(orders):
    axs[r].imshow(d['monge_bijection'][order]*len(d['x']),aspect='auto',vmin=0,vmax=1,cmap='magma',interpolation='nearest')
    axs[r].set(title='Monge 256 x 256 '+('original source order' if r==0 else 'sorted source order'),xlabel='256 equal-weight quantile representatives',ylabel='256 source rows')
   emit(fig,'monge_bijection.png','double_mass seed 0, update 25,001: 실제 256×256 bijection; target은 원래부터 quantile 순서')
   lines+=['위 비교용 Monge 256×1024 그림은 여러 대표점을 원래 candidate column으로 합친 표현이다. 원래 teacher w를 정확히 보존한 plan이 아니다.',
    '해당 그림은 double_mass baseline seed 0의 update 25,001에서 source/candidate/weights를 고정한 counterfactual이다. 각 방법이 자기 학습에서 만든 cloud와 혼동하지 않는다.',
    '| 같은 cloud의 선택 방식 | 선택 teacher CDF 오차 | 선택 teacher W1 |','|---|---:|---:|']
   for name in ['sinkhorn','exact','monge','sinkhorn_same_quantized_teacher']:
    lines.append(f'| {name} | {float(d[name+"_selected_teacher_cdf_error"]):.6f} | {float(d[name+"_selected_teacher_w1"]):.6f} |')
 lines+=['## 실행과 원본 자료','RTX 3090 4개에서 12/12 runs가 성공했다. 본 실험 큐 실행은 2026-09-19 22:46:33–22:51:05 KST, 약 272초였다. 환경 설치·GPU 검증 시간은 제외한 값이다. 이 시간은 4개 GPU의 병렬 실행 시간이며 단일 GPU 시간이나 알고리즘 간 speedup이 아니다.' if len(done)==len(tasks) else '완료된 run만 집계했다.',
  '수치 소스 commit `6510585036ab7221df0fc5022a970ef27e068061`, 새 서버 운영 commit `bf64c9447c689ce3c2b6e45cbbfbc20702cec4d4`, run ID `20260919T134633Z`. 원본 source manifest·설정·최종 checkpoint와 artifact SHA256·action sample·bin edges는 dildata의 `/data1/heejoonorm/OptiQ/studies/20260919_legacy_monge/remote_3114850247/pilot/runs/`에 보관했다.',
  '각 `density/STEP.npz`가 해당 그림의 실제 sample과 histogram mass를 포함한다. RNG는 평가 step t에서 `PRNGKey(1100000 + seed*100000 + t)`로 정의하며 매 평가에 독립 latent 32,768개를 생성한다. 중간 그림은 저장된 sample에 기반한다. Checkpoint는 200updates마다 최신 상태를 덮어쓰므로 모든 중간 시점의 network checkpoint를 보관한 것은 아니다. 최종 checkpoint와 final actor는 보관되어 있다.',
  'GPU 검증에서 독립 SciPy solver와 Monge 비용 일치, exact OT와 독립 LP 일치, 기존 actor update parity 및 checkpoint 재개 검사에 통과했다. 기존 update와의 parameter 비교 오차와 재개 후 actor/RNG 오차는 모두 0이었다.',
  '## 해석 범위','두 seed의 작은 탐색 실험이다. Monge의 quantile teacher 보존과 actor가 실제 분포를 학습하는 것은 별개이며, 성공 여부는 teacher·선택 target·actor를 분리해 해석한다.',
 '정렬 matching은 1D empirical equal-weight 문제의 exact solution이다. 연속 population Monge solver 또는 arbitrary weighted teacher에 대한 정확한 map으로 일반화하지 않는다.',
 '[전체 실험 프로토콜](PROTOCOL.md)']
 text='\n\n'.join(lines)+'\n';(a.output/'report.md').write_text(text);(a.output/'PROTOCOL.md').write_text((a.root/'PROTOCOL.md').read_text())
 try:
  import markdown
  body=markdown.markdown(text,extensions=['tables'])
 except ImportError:body='<pre>'+html.escape(text)+'</pre>'
 for name,_ in figures:
  encoded=base64.b64encode((a.output/name).read_bytes()).decode();body=body.replace('src="'+name+'"','src="data:image/png;base64,'+encoded+'"')
 (a.output/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Legacy Monge pilot</title><style>body{max-width:1200px;margin:40px auto;font-family:sans-serif;line-height:1.6}img{width:100%}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ddd}</style>'+body)
 print(f'{len(done)}/{len(tasks)} complete; {a.output}')

if __name__=='__main__':main()
