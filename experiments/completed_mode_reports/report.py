"""Final, all-seed reports from immutable completed-run exports (no training)."""
import argparse,base64,datetime,hashlib,html,io,json,re,shutil
from pathlib import Path
import numpy as np
from scipy.special import ndtr
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import markdown
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--reports',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args()
labels={'baseline':'Direct GMM','mode_only':'Mode selection','mode_confidence':'Selection + confidence'}
colors={'baseline':'#09899d','mode_only':'#8252b0','mode_confidence':'#dc763c'}
fsizes=[(16,16),(64,64),(64,256),(128,128),(128,256),(512,512),(512,2048)]
bsizes=[(16,16),(64,64),(128,128),(256,256),(1024,1024),(2048,2048),(64,4096),(2048,4096)]
stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M KST')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
def npz(p):
 with np.load(p) as f:return {k:f[k] for k in f.files}
def load(name,B):
 root=a.data/name/'campaign';rec={}
 commit=json.loads((root/'SOURCE_MANIFEST.json').read_text())['commit']
 for run in sorted((root/'runtime/runs').iterdir()):
  c=json.loads((run/'config.json').read_text());done=json.loads((run/'COMPLETE.json').read_text());h=[json.loads(x) for x in (run/'history.jsonl').read_text().splitlines()]
  assert done['step']==h[-1]['step']==20000 and c['source_commit']==commit
  e=npz(run/'eval_020000.npz');d=npz(run/'diagnostic_020000.npz');hist=np.histogram(e['samples'],e['edges'])[0]/len(e['samples'])
  assert len(e['samples'])==32768
  np.testing.assert_allclose(hist,e['histogram'],atol=1e-12);np.testing.assert_allclose(e['target_bin_mass'].sum(),1,atol=3e-7)
  tv=float(.5*np.abs(hist-e['target_bin_mass']).sum());assert abs(tv-h[-1]['histogram_tv'])<1e-8
  ref=json.loads((run/'REFERENCE.json').read_text()) if (run/'REFERENCE.json').exists() else dict(centers=[-.6,0,.6],scales=[.1]*3,weights=[1/3]*3,boundaries=[-1,-.3,.3,1],mode_mass=[1/3]*3)
  key=(c.get('landscape','three'),c['n'],c['m'],c['method'],c['seed'])
  r=dict(c=c,done=done,h=h,e=e,d=d,ref=ref,path=str(run),batch=B,tv=tv,commit=commit)
  rec[key]=r
 return rec
f=load('20260921_gmm_mode_selection_frozen',128);b={1:load('20260921_gmm_mode_gradient',1),32:load('20260921_gmm_mode_gradient_batch32',32)}
assert len(f)==168 and all(len(x)==96 for x in b.values())
def vals(rec,L,size,method,field='tv'):
 return np.array([rec[(L,*size,method,s)]['tv'] if field=='tv' else rec[(L,*size,method,s)]['h'][-1][field] for s in range(4)])
def pm(v):return f'{np.mean(v):.4f} ± {np.std(v,ddof=1):.4f}'
def table(head,rows):return '| '+' | '.join(head)+' |\n|'+'|'.join(['---']*len(head))+'|\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)
def density(x,ref):
 c=np.array(ref['centers']);sd=np.array(ref['scales']);w=np.array(ref['weights']);z=np.sum(w*(ndtr((1-c)/sd)-ndtr((-1-c)/sd)))
 return (w*np.exp(-.5*((x[:,None]-c)/sd)**2)/(np.sqrt(2*np.pi)*sd)).sum(1)/z
def figsave(O,name,fig):fig.savefig(O/'figures'/f'{name}.png',dpi=150,bbox_inches='tight');plt.close(fig)
def fig(name,caption):return f'\n![{caption}](figures/{name}.png)\n\n{caption}\n'
def make_density(O,rec,Ls,sizes,name):
 fig_,axs=plt.subplots(len(sizes),len(Ls),figsize=(5*len(Ls),2.7*len(sizes)),squeeze=False,layout='constrained')
 for row,size in enumerate(sizes):
  for col,L in enumerate(Ls):
   ax=axs[row,col];r=rec[(L,*size,'baseline',0)];x=np.linspace(-1,1,2000);ax.plot(x,density(x,r['ref']),'--',c='#233249',lw=1.5,label='Exact target')
   for method in ['baseline','mode_only']:
    e=[rec[(L,*size,method,s)]['e'] for s in range(4)];ed=e[0]['edges'];den=np.array([v['histogram']/np.diff(ed) for v in e]);ax.stairs(den.mean(0),ed,color=colors[method],lw=1.2,label=labels[method])
   ax.set(title=f'{L} | {size[0]} x {size[1]} | all 4 seeds',xlabel='Action',ylabel='Density',xlim=(-1,1));ax.grid(alpha=.15)
 axs[0,0].legend(fontsize=8);fig_.suptitle('20K updates | 32,768 actual samples/run | histogram mean, no smoothing')
 figsave(O,name,fig_)
def render(O,text):
 (O/'report.md').write_text(text)
 # Offline equations are rendered images; Markdown retains original LaTeX.
 def mathimage(m):
  expr=m[1].replace('\\operatorname','\\mathrm').replace('\\mathbf','\\mathrm').replace('\\frac12','\\frac{1}{2}')
  fig_=plt.figure(figsize=(.1,.1));fig_.text(0,0,'$'+expr+'$',fontsize=15);buf=io.BytesIO();fig_.savefig(buf,format='png',dpi=150,bbox_inches='tight',pad_inches=.12,transparent=True);plt.close(fig_)
  return '<div class="math"><img alt="'+html.escape(m[1],quote=True)+'" src="data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()+'"></div>'
 body=markdown.markdown(re.sub(r'\$\$(.*?)\$\$',mathimage,text,flags=re.S),extensions=['tables','fenced_code'])
 def embed(m):
  path=O/m[1]
  if path.exists():return 'src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'"'
  return m[0]
 body=re.sub(r'src="([^"<>]+\.png)"',embed,body)
 css='body{max-width:1450px;margin:35px auto;padding:0 24px;font:16px/1.75 system-ui;color:#24324a}h2{margin-top:40px;border-top:1px solid #ddd;padding-top:20px}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid #ddd;padding:7px;text-align:left}th{background:#eef3f7}pre,code{background:#f4f6f8;overflow-wrap:anywhere}pre,.math{overflow:auto}.math{text-align:center}a{color:#087b99}'
 (O/'report.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(text.splitlines()[0].lstrip('# '))+'</title><style>'+css+'</style><body>'+body+'</body></html>')

O=a.reports/'20260921_gmm_mode_selection_frozen';O.mkdir(exist_ok=True);(O/'figures').mkdir(exist_ok=True)
Ls=['needle3','comb6','rugged7'];summary=[]
for L in Ls:
 for size in fsizes:
  x,y=vals(f,L,size,'baseline'),vals(f,L,size,'mode_only')
  summary.append(dict(landscape=L,n=size[0],m=size[1],baseline_tv_mean=float(x.mean()),selection_tv_mean=float(y.mean()),paired_delta_mean=float((y-x).mean()),selection_wins=int((y<x).sum())))
make_density(O,f,Ls,fsizes,'density_all_sizes')
# Seed-level overview and full learning curves; all four independent runs retained.
for L in Ls:
 fig_,axs=plt.subplots(7,4,figsize=(17,18),layout='constrained')
 for row,size in enumerate(fsizes):
  for s in range(4):
   ax=axs[row,s];r=f[(L,*size,'baseline',s)];x=np.linspace(-1,1,2000);ax.plot(x,density(x,r['ref']),'--',c='#24324a',lw=1)
   for m in ['baseline','mode_only']:
    e=f[(L,*size,m,s)]['e'];ax.stairs(e['histogram']/np.diff(e['edges']),e['edges'],color=colors[m],lw=1,label=labels[m])
   ax.set(title=f'{size[0]} x {size[1]} | seed {s}',xlim=(-1,1));ax.grid(alpha=.12)
 axs[0,0].legend(fontsize=8);fig_.suptitle(L+' | all seeds at 20K | raw histograms');figsave(O,'seeds_'+L,fig_)
 fig_,axs=plt.subplots(7,2,figsize=(13,18),layout='constrained')
 for row,size in enumerate(fsizes):
  for m in ['baseline','mode_only']:
   hs=[f[(L,*size,m,s)]['h'] for s in range(4)]
   for col,xkey in enumerate(['step','train_seconds']):
    for s,h in enumerate(hs):axs[row,col].plot([q[xkey] for q in h],[q['histogram_tv'] for q in h],color=colors[m],alpha=.65,lw=1,label=labels[m] if s==0 else None)
  for col in range(2):axs[row,col].set(title=f'{size[0]} x {size[1]}',xlabel='Optimizer updates' if col==0 else 'Recorded training seconds',ylabel='Histogram TV',ylim=(0,.8));axs[row,col].grid(alpha=.2)
 axs[0,0].legend();fig_.suptitle(L+' | four seed trajectories (not confidence intervals)');figsave(O,'tracking_'+L,fig_)
# Matched size/seed, not selected for advantage; density stages group0 plus true actor histogram.
rep=(128,128);seed=0
fig_,axs=plt.subplots(2,3,figsize=(16,7),layout='constrained')
for row,m in enumerate(['baseline','mode_only']):
 for col,L in enumerate(Ls):
  r=f[(L,*rep,m,seed)];d,e,ref=r['d'],r['e'],r['ref'];ax=axs[row,col];ed=np.linspace(-1,1,81);x=np.linspace(-1,1,2000)
  ax.plot(x,density(x,ref),'--',c='#24324a',label='Exact target')
  ax.stairs(np.histogram(d['candidates'],ed)[0]/len(d['candidates'])/np.diff(ed),ed,color='#929aa8',alpha=.6,label='Proposal (group 0)')
  ax.stairs(np.histogram(d['candidates'],ed,weights=d['weights'])[0]/np.diff(ed),ed,color='#d49b1f',alpha=.8,label='Weighted teacher (group 0)')
  ax.stairs(np.histogram(e['samples'],ed)[0]/32768/np.diff(ed),ed,color=colors[m],label='Actor (32,768 draws)')
  ax.set(title=f'{L} | {labels[m]}',xlabel='Action',ylabel='Density');ax.grid(alpha=.15)
axs[0,0].legend(fontsize=8);fig_.suptitle('128 x 128 | seed 0 | 20K | coarse 80-bin view, no smoothing');figsave(O,'teacher_actor',fig_)
for sorted_ in [False,True]:
 fig_,axs=plt.subplots(2,3,figsize=(16,7),layout='constrained')
 for row,m in enumerate(['baseline','mode_only']):
  for col,L in enumerate(Ls):
   d=f[(L,*rep,m,0)]['d'];J=d['joint'];mass=J.sum(1);R=np.divide(J,mass[:,None],out=np.zeros_like(J),where=mass[:,None]>0)
   if sorted_:R=R[np.argsort(d['training_mu'][:,0])][:,np.argsort(d['candidates'])]
   im=axs[row,col].imshow(np.ma.masked_less_equal(R,0),origin='lower',aspect='auto',interpolation='nearest',norm=LogNorm(1e-5,1),cmap='magma');axs[row,col].set(title=f'{L} | {labels[m]}',xlabel='Candidate index',ylabel='Student index')
 fig_.colorbar(im,ax=axs.ravel().tolist(),label='Conditional row mass R');fig_.suptitle('128 x 128 | seed 0 | 20K | group 0 | '+('rows/columns sorted by action' if sorted_ else 'original sampling order'));figsave(O,'assignment_'+('sorted' if sorted_ else 'raw'),fig_)
fig_,axs=plt.subplots(2,3,figsize=(16,6),layout='constrained')
for row,m in enumerate(['baseline','mode_only']):
 for col,L in enumerate(Ls):
  d=f[(L,*rep,m,0)]['d'];H=d['H'];order=np.argsort(d['training_z'][:,0]);im=axs[row,col].imshow(H[order].T,origin='lower',aspect='auto',vmin=0,vmax=1);axs[row,col].set(title=f'{L} | {labels[m]}',xlabel='Latent z order',ylabel='KNOWN basin index')
fig_.colorbar(im,ax=axs.ravel().tolist(),label='H(basin | component)');fig_.suptitle('Oracle-basin aggregation H, NOT unsupervised mode detection');figsave(O,'oracle_H',fig_)
fig_,axs=plt.subplots(2,3,figsize=(14,8),layout='constrained')
for row,m in enumerate(['baseline','mode_only']):
 for col,L in enumerate(Ls):
  d=f[(L,*rep,m,0)]['d'];C=d['gradient_cosine'];im=axs[row,col].imshow(C,vmin=-1,vmax=1,cmap='coolwarm')
  for i in range(len(C)):
   for j in range(len(C)):axs[row,col].text(j,i,f'{C[i,j]:.1f}',ha='center',va='center',fontsize=8)
  axs[row,col].set(title=L+' | '+labels[m],xlabel='Basin',ylabel='Basin')
fig_.colorbar(im,ax=axs.ravel().tolist(),label='Parameter-gradient cosine');fig_.suptitle('Original marginal NLL mode gradients | batch mean 128 | 128 x 128, seed 0, 20K');figsave(O,'gradient_cosine',fig_)
rows=[];allrows=[];runjson=[];teacherrows=[]
for L in Ls:
 for size in fsizes:
  x,y=vals(f,L,size,'baseline'),vals(f,L,size,'mode_only');rows.append([L,f'{size[0]}×{size[1]}',pm(x),pm(y),f'{(y-x).mean():+.4f}',f'{(y<x).sum()}/4'])
  for m in ['baseline','mode_only']:
   rs=[f[(L,*size,m,s)] for s in range(4)];tvmean=[];tvgroup=[]
   for r in rs:
    target=np.array(r['ref']['mode_mass']);bm=r['d']['batch_teacher_mode_mass'];tvmean.append(float(.5*np.abs(bm.mean(0)-target).sum()));tvgroup.append(float(.5*np.abs(bm-target).sum(1).mean()))
    h=r['h'][-1];allrows.append([L,f'{size[0]}×{size[1]}',labels[m],r['c']['seed'],f"{r['tv']:.4f}",f"{h['basin_tv']:.4f}",f"{h['backup_error']:+.4f}",f"{h['coverage_fraction']:.2f}",f"{h['specialist_fraction']:.3f}",f"{r['done']['train_seconds']:.1f}",f"{r['done']['diagnostic_seconds']:.1f}"])
    runjson.append(dict(landscape=L,n=size[0],m=size[1],method=m,seed=r['c']['seed'],**h,completion=r['done'],source_commit=r['commit']))
   teacherrows.append([L,f'{size[0]}×{size[1]}',labels[m],pm(tvmean),pm(tvgroup),pm(vals(f,L,size,m,'basin_tv'))])
(O/'all_runs.md').write_text('# 全168 run 최종 수치\n\n'+table(['Q','N×M','방법','Seed','Histogram TV','Basin TV','Backup error','Coverage','Specialist','Training s','Diagnostic s'],allrows))
(O/'teacher_table.md').write_text('# 최종 teacher 진단\n\n128그룹 평균 teacher의 basin TV와, 그룹별 basin TV 평균은 서로 다르다. 후자가 크고 전자가 작으면 그룹 간 표본 변동의 상쇄가 크다는 뜻이다. 각 값은 4-seed 평균 ± 표준편차.\n\n'+table(['Q','N×M','방법','TV(mean teacher)','mean TV(teacher)','Actor basin TV'],teacherrows))
(O/'summary.json').write_text(json.dumps(dict(snapshot=stamp,completed=168,failed=0,source_commit=next(iter(f.values()))['commit'],analysis_commit=a.commit,paired=summary,runs=runjson),indent=2,ensure_ascii=False))
text=fr'''# Frozen Q: Direct GMM vs mode selection — batch 128 최종 결과

**168/168 완료, 실패 0.** 3종류 Q × 7크기 × 2방법 × 4 seeds, 모두 20,000 optimizer updates. 분석 시점: {stamp}.

**작은 N×M에서는 mode selection의 개선이 크지만, 큰 설정에서는 이점이 줄거나 Direct GMM보다 나빠진다.** 특히 needle3·rugged7의 512×2048에서는 선택 방식의 최종 오차가 더 크다. “mode를 선택하면 항상 더 정확해진다”는 결과가 아니다.

이 실험은 **정답 Q의 valley 경계로 candidate의 basin을 미리 정한 oracle ablation**이다. 사용자가 의도한 heatmap 기반 자동 grouping을 구현한 실험은 아니다. 두 문제를 혼동하지 않도록, 아래에서 candidate-level R과 정답 basin으로 집계한 H를 분리해 표시한다.

## 0. 설정과 지표

| 항목 | 값 |
|---|---|
| 환경 | 상태 하나, 고정 1D Q, action [-1,1]; critic 학습 없음 |
| Q | 아래 Gaussian mixture의 log density ×0.25; 기존 frozen Q와 동일 |
| Actor | 기존 toy v5 squashed conditional Gaussian, 256×2 GELU, z는 1D 정규분포 |
| Sigma | 초기 .5, log sigma 범위 [-5,1] |
| Optimizer | Adam 3e-4, gradient clipping·EMA 없음 |
| Teacher | 현재 actor conditional mixture, sigma floor .05, density correction |
| Temperature | .25 |
| Batch | 같은 update 직전 parameter에서 128개 독립 N×M 그룹을 뽑아 gradient 평균 후 Adam 한 번 |
| N×M | 16×16, 64×64, 64×256, 128×128, 128×256, 512×512, 512×2048 |
| Seeds / updates | 0–3 / 각각 20K |
| 기본 density 그림 | run별 실제 32,768 action의 512-bin histogram, smoothing 없음 |

$$f(a)=\sum_k\rho_k\mathcal{{N}}(a;c_k,h_k^2),\quad Q(a)=0.25\log f(a),\quad p^*(a)=\frac{{f(a)}}{{\int_{{-1}}^1 f(x)dx}}.$$

| Q | 중심 | 폭 | 명목 mixture mass |
|---|---|---|---|
| needle3 | −.7, 0, .7 | .035, .18, .06 | .3, .4, .3 |
| comb6 | −.75, −.45, −.15, .15, .45, .75 | 모두 .035 | 모두 1/6 |
| rugged7 | −.9, −.6, −.32, .02, .3, .58, .9 | .025, .075, .02, .11, .035, .06, .018 | .12, .18, .10, .20, .12, .16, .12 |

**Histogram TV**는 512개 bin의 actor mass와 정답 mass의 절대차 합을 2로 나눈 값이다. Mode 내부의 폭·모양·질량을 모두 반영하며 낮을수록 좋다. **Basin TV**는 density valley 사이 구간별 총질량으로 같은 계산을 한다. Basin TV가 작아도 mode 내부 모양은 틀릴 수 있다. **Coverage**는 actor basin mass가 정답 mass의 25% 이상인 basin의 비율이며, 정확한 복구 판정과 다르다. **Backup error**는 sample mean Q에서 정답 Boltzmann expectation을 뺀 값이며 RL return은 아니다.

$$\mathrm{{TV}}_{{hist}}=\frac12\sum_b|\hat p_b-p_b^*|.$$

아래 overview는 모든 4 seeds의 histogram 평균이다. Seed별 그림도 전부 제공하므로 평균이 실패한 seed를 가리지 않는지 함께 확인할 수 있다.
'''+fig('density_all_sizes','모든 Q × N×M의 최종 density. 각 선은 4 seeds의 실제 histogram 평균, 검은 점선만 exact target이다.')+r'''## 1. 전체 4-seed 비교

평균 ± **seed 간 표준편차**다. 신뢰구간은 아니다. 마지막 열은 selection의 TV가 낮았던 paired seed 수다. 작은 차이까지 통계적으로 유의한 개선으로 부르지 않는다.

'''+table(['Q','N×M','Direct GMM TV','Selection TV','차이 selection−GMM','Selection 우세 seed'],rows)+r'''
핵심은 조건별 차이다. 16×16에서는 세 Q 모두 큰 개선이 나타난다. 반면 needle3는 128×256부터 Direct GMM이 평균상 더 좋고, rugged7도 512×512·512×2048에서 Direct GMM이 더 좋다. Comb6는 중간 크기에서 비교 순서가 뒤집힌다. 따라서 mode-selection의 역할을 작은 유한 mixture에서의 학습 보조와, 큰 mixture에서의 불필요한 gradient 제거 가능성으로 나누어 검토할 만하다. 이는 해석 가설이며 원인을 분리한 개입 실험은 아니다.

## 2. 학습 경로와 계산 시간

각 선은 seed 하나다. Training 시간은 기록된 training block 누적 시간(JIT 포함)이며, 진단·평가·I/O·Slurm 대기는 제외한다. 노드와 GPU 부하가 달라 정밀한 속도 benchmark는 아니다. 원본 수치표에는 진단 시간을 따로 적었다.
'''+''.join(fig('tracking_'+L,L+': updates 및 기록된 training 시간 대비 histogram TV, 모든 4 seeds.') for L in Ls)+r'''
## 3. Teacher와 actor를 나누어 보기

아래는 사전에 고른 **128×128, seed 0, 20K**다. Proposal과 weighted teacher는 batch 128 중 **group 0 하나**이며, actor는 별도의 32,768 samples다. Candidate 128개의 teacher histogram이 울퉁불퉁한 것은 finite sample 효과도 포함한다. 시인성을 위해 이 그림만 공통 80 bins를 썼다. 최종 TV는 원래 512 bins 그대로다.
'''+fig('teacher_actor','Proposal → importance-weighted teacher → actor. 같은 색/정의로 모든 Q와 방법을 비교한다.')+r'''
[전체 teacher 수치표](teacher_table.md)는 group 0에 의존하지 않고 128그룹 전체를 사용한다. 한쪽은 “평균 teacher의 TV”, 다른 쪽은 “각 teacher의 TV 평균”이다. 평균 teacher가 정확하다고 매 update 그룹의 teacher까지 정확하다는 뜻은 아니다. 또한 teacher basin TV만으로 within-mode density가 정확하다고 판단하지 않는다.

## 4. Candidate-level heatmap과 oracle mode 집계의 차이

$$\gamma_{ij}=\frac{k_i(b_j)}{\sum_l k_l(b_j)},\quad J_{ij}=w_j\gamma_{ij},\quad R_{ij}=\frac{J_{ij}}{\sum_l J_{il}}.$$

다음 R은 mode label 없이 계산한다. 실제 학습에 OT solver는 없으며 GMM의 effective assignment다. Raw와 sorted는 동일 행렬의 순서만 바꾼 것이다. Row normalization은 전체 usage alpha를 지우므로 밝은 row가 큰 gradient를 받는다는 뜻은 아니다.
'''+fig('assignment_raw','R, 원본 순서. 128×128, seed 0, 20K, group 0.')+fig('assignment_sorted','동일 R. Student 중심 tanh(mu)와 candidate action을 기준으로 각각 정렬.')+r'''
반면 학습에 사용한 H는 **알려진 basin label로 J를 집계**한다.

$$H_{mi}=\frac{\sum_{j:b_j\in B_m}J_{ij}}{\sum_jJ_{ij}},\quad m_i=\arg\max_m H_{mi}.$$

$$L_{route}=-\sum_{i,j}\mathrm{stopgrad}[J_{ij}\,\mathbf{1}(b_j\in B_{m_i})]\log k_i(b_j).$$

Confidence 곱, 남은 질량의 재정규화, 추가 1/N은 없다. 원래 marginal NLL의 정확한 gradient가 아니며, term 제거 때문에 gradient 크기도 달라진다. 같은 latent가 다음 update에서도 고정 component로 유지되는 구조는 아니다.
'''+fig('oracle_H','정답 basin으로 집계한 H. 자동 mode detector의 출력이 아니다.')+r'''
## 5. Gradient 간섭

아래는 각 학습된 actor에서 **원래 marginal NLL**을 basin별로 분해하고 128그룹 평균을 낸 parameter gradient 사이의 cosine이다. Routed gradient 자체를 표시한 것이 아니다. 학습 경로가 다르면 actor와 teacher도 달라져 panel 간 차이만으로 인과를 확정할 수 없다.
'''+fig('gradient_cosine','128×128, seed 0, 20K. −1은 반대 방향, +1은 같은 방향이며 gradient 크기는 나타내지 않는다.')+r'''
Output에서 다른 basin의 기여를 버려도 공유 network parameter 때문에 다른 latent가 움직일 수 있다. 따라서 이 구현을 parameter gradient의 직교화나 완전한 간섭 제거라고 부르지 않는다. 선택 방식이 항상 우세하지 않다는 최종 결과도 이 구분과 함께 읽어야 한다.

## 6. 모든 seed의 density
'''+''.join(fig('seeds_'+L,L+': 7크기 × 4seeds. 모든 완료 결과를 표시했다.') for L in Ls)+fr'''
## 7. 결론과 재현

이번 데이터는 **작은 N×M에서 oracle mode selection이 분포 fitting을 크게 돕는다**는 점을 지지한다. 큰 설정에서의 일관된 우위, 자동 mode 발견, MuJoCo 성능 개선은 입증하지 않는다. 다음 실험은 oracle 경계 대신 assignment에서 추정한 그룹을 사용하되, 잘못 나눈 그룹과 gradient 크기 변화까지 분리해 보아야 한다.

- [모든 168개 run 수치](all_runs.md), [구조화 요약](summary.json), [실험 계획](PROTOCOL.md).
- Source commit: `{next(iter(f.values()))['commit']}`. 분석 commit: `{a.commit}`.
- 완료 step·원본 표본 수·histogram·TV·정답 정규화를 재검증했다.
- 전체 원본: `dildata:/data1/heejoonorm/OptiQ/studies/20260921_gmm_mode_selection_frozen/campaign/runtime/runs/`.
- HTML은 그림·수식을 내장하여 오프라인에서 열 수 있다. Markdown은 수식 원문을 보존한다.
'''
render(O,text)

# Final update for the preceding 3-mode B1/B32 study, all192 runs.
O=a.reports/'20260921_gmm_mode_gradient_batch32_final';O.mkdir(exist_ok=True);(O/'figures').mkdir(exist_ok=True)
rows=[]
for size in bsizes:
 for m in labels:
  x,y=vals(b[1],'three',size,m),vals(b[32],'three',size,m)
  rows.append([f'{size[0]}×{size[1]}',labels[m],pm(x),pm(y),f'{(y-x).mean():+.4f}',f'{(y<.1).sum()}/4'])
for start in [0,3,6]:
 ss=bsizes[start:start+3];fig_,axs=plt.subplots(2,len(ss),figsize=(5*len(ss),7),squeeze=False,layout='constrained')
 for row,B in enumerate([1,32]):
  for col,size in enumerate(ss):
   ax=axs[row,col];ref=b[B][('three',*size,'baseline',0)]['ref'];x=np.linspace(-1,1,1400);ax.plot(x,density(x,ref),'--',c='#24324a',label='Exact target')
   for m in labels:
    ev=[b[B][('three',*size,m,s)]['e'] for s in range(4)];edges=ev[0]['edges'];d=np.array([e['histogram']/np.diff(edges) for e in ev]);ax.stairs(d.mean(0),edges,color=colors[m],label=labels[m]);ax.fill_between((edges[:-1]+edges[1:])/2,d.min(0),d.max(0),color=colors[m],alpha=.08)
   ax.set(title=f'B={B} | {size[0]} x {size[1]} | all 4 seeds',xlabel='Action',ylabel='Density',xlim=(-1,1));ax.grid(alpha=.15)
 axs[0,0].legend(fontsize=8);fig_.suptitle('20K updates | mean and min–max envelope of four sampled histograms');figsave(O,'density_'+str(start),fig_)
# Seed scatter exposes basin-of-attraction failures hidden by the average.
fig_,axs=plt.subplots(2,1,figsize=(14,8),layout='constrained')
for row,B in enumerate([1,32]):
 for j,m in enumerate(labels):
  means=[]
  for k,size in enumerate(bsizes):
   v=vals(b[B],'three',size,m);x=k+(j-1)*.22;axs[row].scatter(x+np.linspace(-.04,.04,4),v,c=colors[m],s=24);means.append(v.mean())
  axs[row].plot(np.arange(8)+(j-1)*.22,means,c=colors[m],lw=1,label=labels[m])
 axs[row].set(xticks=range(8),xticklabels=[f'{n}x{m}' for n,m in bsizes],ylabel='Histogram TV',title=f'Batch {B}: each dot is one seed');axs[row].grid(alpha=.15)
axs[0].legend();figsave(O,'seed_scatter',fig_)
data=[]
for B,rec in b.items():
 for (L,n,m,method,s),r in rec.items():data.append(dict(batch=B,n=n,m=m,method=method,seed=s,tv=r['tv'],history=r['h'],completion=r['done'],source_commit=r['commit']))
(O/'summary.json').write_text(json.dumps(dict(snapshot=stamp,completed={'1':96,'32':96},analysis_commit=a.commit,runs=data),indent=2))
allrows=[[r['batch'],f"{r['n']}×{r['m']}",labels[r['method']],r['seed'],f"{r['tv']:.4f}",f"{r['completion']['train_seconds']:.1f}",f"{r['completion']['diagnostic_seconds']:.1f}"] for r in data]
(O/'all_runs.md').write_text('# 全192 run\n\n'+table(['Batch','N×M','방법','Seed','TV','Training s','Diagnostic s'],allrows))
text=fr'''# 3-mode gradient selection: batch 1 vs 32 — 4-seed 최종 결과

**Batch 1: 96/96, batch 32: 96/96, 실패 0. 모두 20K updates.** 분석 시점 {stamp}. 앞선 seed 0 중심의 중간 보고서를 모든 4 seeds로 확장했다.

가장 중요한 수정은 **batch 32의 128×128에서 기존 Direct GMM이 항상 성공한 것은 아니라는 점**이다. Seed 0의 낮은 TV만 보면 개선이 컸지만, 다른 seed에서는 넓은 분포에 머무는 결과가 남았다. Mode selection은 이 크기에서 더 일관되게 복구했다. 반면 1024×1024 이상에서는 Direct GMM도 충분히 잘 복구하여 방법 간 차이가 작아진다.

## 0. 설정과 지표

중심 −.6,0,.6, 폭 .1, 동일 질량인 3-mode Gaussian mixture를 [-1,1]에서 정규화한다. Q=.25 log f, temperature=.25. 기존 v5 toy의 squashed Gaussian actor(256×2 GELU, log sigma [-5,1], 초기 sigma .5), Adam 3e-4, 현재 actor conditional-mixture proposal과 density correction을 유지했다. 이 설정은 최근 TRG MuJoCo의 truncated Gaussian actor와 다르다.

Batch B는 동일한 update 직전 parameter에서 B개의 독립 N×M 그룹을 뽑아 gradient를 평균한 뒤 Adam을 **한 번** 적용한다. B=32가 한 mixture의 component 수를 32N으로 만드는 것은 아니다. 같은 20K updates라도 표본·계산 예산은 같지 않다.

세 방법은 Direct GMM / mode selection / selection + confidence다. **모두 mode-gradient 개입에 사용하는 basin은 알려진 경계 −.3,.3으로 정의했다. 자동 mode discovery가 아니다.**

$$J_{{ij}}=w_j\gamma_{{ij}},\quad H_{{mi}}=\frac{{\sum_{{j\in B_m}}J_{{ij}}}}{{\sum_jJ_{{ij}}}},\quad m_i=\arg\max_mH_{{mi}}.$$

Mode selection은 담당 basin의 output gradient만 남기고, confidence 조건은 여기에 max H를 곱한다. 추가 norm 보정은 없다. 이 때문에 방향뿐 아니라 gradient 크기도 달라진다.

Histogram TV는 **256 bins**의 actor-vs-target mass 절대차 합의 절반이다. Density는 각 run의 32,768 actual samples로 그렸고 smoothing은 없다. 새 frozen-Q batch128 보고서는 512 bins와 다른 Q를 사용하므로 TV 수치를 곧바로 비교하면 안 된다.

## 1. 최종 density와 seed 차이

굵은 선은 4-seed histogram 평균, 옅은 영역은 seed 간 min–max 범위다. 평균선이 3 peaks를 갖더라도 모든 seed가 성공했다는 뜻은 아니다. 아래 seed별 점과 수치표를 함께 본다.
'''+''.join(fig('density_'+str(st),'같은 N×M의 batch 1(위)과 batch 32(아래), 모든 4 seeds.') for st in [0,3,6])+fig('seed_scatter','모든 완료 run의 최종 TV. 점 하나가 seed 하나, 연결선은 평균이다.')+r'''
## 2. 전체 결과

평균 ± seed 간 표준편차. 마지막 열은 참고용으로 정한 **TV<0.1** 도달 seed 수이며, 사전 정의한 성공 확률이나 유의성 검정이 아니다.

'''+table(['N×M','방법','Batch 1 TV','Batch 32 TV','차이 32−1','B32 TV<.1'],rows)+fr'''
## 3. 해석

- **작은 mixture에서는 batch 평균만으로 부족하다.** 16×16·64×64의 Direct GMM은 B=32에서도 높은 TV에 머문다. Mode selection은 크게 개선된다.
- **중간 크기에서는 seed 의존성을 봐야 한다.** 128×128의 Direct GMM은 seed 0에서는 좋아졌지만, 4-seed 평균은 약 .2855다. 단일 seed 결과를 보편적인 성공으로 표현했던 중간 해석은 범위를 줄여야 한다.
- **큰 크기에서는 원래 GMM도 잘 된다.** 1024×1024의 B32 Direct GMM 평균 TV는 .0393, mode selection은 .0402로 비슷하다. 선택 방식의 보편적 우위는 아니다.
- **Confidence는 일관된 추가 개선을 주지 않는다.** 일부 조건은 더 좋고 일부는 더 나쁘다. 이번 데이터만으로 confidence 곱이 필요하다고 말하기 어렵다.
- Output gradient를 골라도 공유 parameter를 통한 latent 간 간섭은 남는다. 잘 fitting한 actor에서 음의 mode-gradient cosine이 존재할 수 있으므로 cosine만으로 실패 원인을 정하지 않는다.

## 4. 원본과 보관

[전체 192개 run 수치](all_runs.md), [구조화 요약과 전체 trajectory](summary.json), [설정](PROTOCOL.md).

Batch 1 commit: `{next(iter(b[1].values()))['commit']}`. Batch 32 commit: `{next(iter(b[32].values()))['commit']}`. 분석 commit: `{a.commit}`.

입력 원본은 각각 `dildata:/data1/heejoonorm/OptiQ/studies/20260921_gmm_mode_gradient/campaign/runtime/runs/` 및 `20260921_gmm_mode_gradient_batch32/campaign/runtime/runs/`에 있다. 모든 192개 최종 histogram과 TV를 원본 sample에서 재계산했다. Training/diagnostic 시간은 전체 수치표에 기록했으며, 서로 다른 GPU에서의 정밀 wall-clock 비교로 사용하지 않는다.

HTML은 그림·수식을 내장했고, Markdown은 수식 원문을 유지한다.
'''
render(O,text)
print(json.dumps({'frozen':summary,'batch32_final':96,'reports':[str(a.reports/'20260921_gmm_mode_selection_frozen'),str(O)]},indent=2))
