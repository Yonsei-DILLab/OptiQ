from pathlib import Path
import sys,json,hashlib,datetime,shutil,re,base64,io,html,argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
import markdown

parser=argparse.ArgumentParser(description='Build GMM40 final report from saved results; no training.')
parser.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[3])
ROOT=parser.parse_args().workspace.resolve()
ST=ROOT/'studies/20260920_gmm40_comparison'; DATA=ST/'results'; EX=ST/'analysis_exports'
OUT=ROOT/'reports/20260920_gmm40_comparison'; FIG=OUT/'figures'; AUX=OUT/'data'
for p in [OUT,FIG,AUX]:p.mkdir(exist_ok=True,parents=True)
sys.path.insert(0,str(ROOT/'git-checkouts/OptiQ'))
from experiments.gmm40_comparison.metrics import evaluate,component_stats
C=['legacy_reference','legacy_matched','monge_matched','v5_ot_matched','gmm_matched','gmm_small','gmm_large','v5_ot_large']
NAME=['Legacy 2048x2048','Legacy 256x1024','Monge 256x1024','v5 OT 256x1024','GMM 256x1024','GMM 16x64','GMM 2048x2048','v5 OT 2048x2048']
LAB=dict(zip(C,NAME));COL=dict(zip(C,['#143d59','#438ca5','#b55a30','#b08aca','#1b8e65','#85b47c','#007d9c','#732a95']))
RUN={};H={};S={};T={};METRICS={};AGG={};VALID=[];BINS=np.linspace(-50,50,161)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
manifest=json.loads((EX/'EXPORT_MANIFEST.json').read_text());manifest_runs={x['run']:x for x in manifest['records']}
for c in C:
 for seed in range(4):
  name=f'{c}_s{seed}';p=DATA/name
  cfg=json.loads((p/'config.json').read_text());end=json.loads((p/'COMPLETE.json').read_text());h=[json.loads(l) for l in (p/'history.jsonl').read_text().splitlines()]
  assert end['step']==75000 and end['commit']==manifest['source_commit']
  assert [v['step'] for v in h]==list(range(0,75001,5000)),name
  s=dict(np.load(p/'samples_075000.npz'));assert s['samples'].shape==(32768,2) and np.isfinite(s['samples']).all()
  assert hashlib.sha256((p/'samples_075000.npz').read_bytes()).hexdigest()==manifest_runs[name]['sample_sha256']
  fresh,_=evaluate(s['samples'],s['reference'],s['locs'],s['scales']);old=h[-1]['metrics']
  err=max(abs(fresh[k]-old[k]) for k in fresh if isinstance(fresh[k],(float,int)))
  assert err<1e-5,(name,err)
  segments=[json.loads(l) for l in (p/'segments.jsonl').read_text().splitlines()]
  assert [(x['start'],x['end']) for x in segments]==[(i,i+15000) for i in range(0,75000,15000)]
  row=dict(condition=c,seed=seed,**old,train_seconds=h[-1]['train_seconds'],segment_wall_seconds=sum(x['segment_wall_seconds'] for x in segments),**h[-1]['diagnostics'])
  RUN[name]=dict(cfg=cfg,complete=end,wandb=json.loads((p/'WANDB.json').read_text()));METRICS[name]=row;H[name]=h;S[name]=s;T[name]=dict(np.load(EX/f'{name}_teacher.npz'));VALID.append(dict(run=name,recomputed_max_abs_error=err,sample_sha256=manifest_runs[name]['sample_sha256']))
 numeric=[k for k,v in METRICS[f'{c}_s0'].items() if isinstance(v,(float,int)) and k!='seed']
 AGG[c]={k:dict(mean=float(np.mean([METRICS[f'{c}_s{s}'][k] for s in range(4)])),sd=float(np.std([METRICS[f'{c}_s{s}'][k] for s in range(4)],ddof=1))) for k in numeric}
ref=S['legacy_reference_s0']['reference'];locs=S['legacy_reference_s0']['locs'];scales=S['legacy_reference_s0']['scales']
floor=json.loads((ST/'validation/VALIDATION_PASSED.json').read_text())['reference_sampling_floor']
np.savez_compressed(AUX/'histograms_75k.npz',edges=BINS,reference=np.histogram2d(*ref.T,bins=BINS)[0],**{n:np.histogram2d(*s['samples'].T,bins=BINS)[0] for n,s in S.items()})
(AUX/'summary.json').write_text(json.dumps(dict(source_commit=manifest['source_commit'],aggregate=AGG,runs=METRICS,reference_floor=floor),indent=2)+'\n')
(AUX/'verification.json').write_text(json.dumps(dict(complete_runs=32,failed_runs=len(list(DATA.glob('*/FAILED.json'))),final_step=75000,verified=VALID),indent=2)+'\n')
for src in [ST/'PROTOCOL.md',ST/'SOURCE_MANIFEST.json',EX/'EXPORT_MANIFEST.json']:shutil.copy2(src,AUX/src.name)
(AUX/'run_manifest.json').write_text(json.dumps(RUN,indent=2)+'\n')

NORM=LogNorm(1e-6,.005)
def hist(ax,x,weights=None,title='',bins=BINS,centers=True):
 counts=np.histogram2d(x[:,0],x[:,1],bins=bins,weights=weights)[0]
 mass=counts/(len(x) if weights is None else np.sum(weights));area=np.diff(bins)[0]**2
 im=ax.imshow(np.ma.masked_equal((mass/area).T,0),origin='lower',extent=[bins[0],bins[-1],bins[0],bins[-1]],norm=NORM,cmap='magma',interpolation='nearest',aspect='equal')
 ax.set_facecolor('#090613')
 if centers:ax.scatter(locs[:,0],locs[:,1],s=5,facecolors='none',edgecolors='#6be2e9',linewidths=.45)
 ax.set(title=title,xlabel='x1',ylabel='x2',xlim=(bins[0],bins[-1]),ylim=(bins[0],bins[-1]));return im

def save(fig,name):
 fig.savefig(FIG/f'{name}.png',dpi=150,bbox_inches='tight');plt.close(fig)

fig,axs=plt.subplots(3,3,figsize=(13,13),layout='constrained')
hist(axs.flat[0],ref,title='Target reference (independent samples)')
for ax,c in zip(axs.flat[1:],C):
 m=METRICS[f'{c}_s0'];im=hist(ax,S[f'{c}_s0']['samples'],title=f'{LAB[c]}\ncoverage {m["modes_covered"]}/40 | mass TV {m["mode_mass_tv"]:.3f}')
fig.colorbar(im,ax=axs.ravel().tolist(),label='Density (shared log scale)',shrink=.65)
fig.suptitle('GMM40 | 75K updates | seed 0 | 32,768 raw samples / panel\n160 x 160 bins in [-50,50]^2 | no smoothing',fontsize=16);save(fig,'density_overview')

fig,axs=plt.subplots(8,4,figsize=(14,26),layout='constrained')
for row,c in enumerate(C):
 for seed in range(4):
  name=f'{c}_s{seed}';m=METRICS[name];hist(axs[row,seed],S[name]['samples'],title=f'{LAB[c]} | seed {seed}\n{m["modes_covered"]}/40; TV {m["mode_mass_tv"]:.3f}')
fig.suptitle('All 32 final runs | 75K | 32,768 raw samples each | 160 x 160 bins | no smoothing\nSame axes and density scale as overview; no seed pooling',fontsize=16);save(fig,'all_seeds')

fig,axs=plt.subplots(2,2,figsize=(14,9),layout='constrained')
for ax,k,label in zip(axs.flat,['modes_covered','mode_mass_tv','histogram_tv','within_mode_cov_error'],['Modes covered (higher is better)','Mode-mass TV (lower is better)','2D histogram TV (lower is better)','Within-mode covariance error (lower is better)']):
 for c in C:
  v=np.array([[x['metrics'][k] for x in H[f'{c}_s{s}']] for s in range(4)]);mean=v.mean(0);sd=v.std(0,ddof=1);x=np.arange(0,75001,5000)/1000
  ax.plot(x,mean,label=LAB[c],color=COL[c],lw=2);ax.fill_between(x,np.maximum(0,mean-sd),mean+sd,color=COL[c],alpha=.10)
 ax.set(xlabel='Actor updates (thousands)',ylabel=label);ax.grid(alpha=.2)
 if k=='modes_covered':ax.set_ylim(0,41)
 if k=='histogram_tv':ax.axhline(floor[k],color='gray',ls=':',label='Independent target sample floor')
handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,fontsize=9)
fig.suptitle('Learning curves | mean +/- sample SD over seeds 0-3\n32,768 raw samples / evaluation; evaluation every 5K',fontsize=15);save(fig,'learning_curves')

fig,axs=plt.subplots(4,2,figsize=(14,13),layout='constrained')
for ax,c in zip(axs.flat,C):
 v=np.array([S[f'{c}_s{s}']['mass'] for s in range(4)]);m=v.mean(0);sd=v.std(0,ddof=1)
 ax.bar(np.arange(40),m,color=COL[c],alpha=.8);ax.errorbar(np.arange(40),m,yerr=sd,fmt='none',color='#333',lw=.7);ax.axhline(1/40,color='black',ls='--')
 ax.set(title=LAB[c],xlabel='Target mode index',ylabel='Posterior-assigned mass');ax.grid(axis='y',alpha=.15)
fig.suptitle('Mode allocation at 75K | 32,768 samples / seed | bar = mean, error = sample SD (4 seeds)\nDashed line = target mass 0.025; y-axis ranges vary by panel',fontsize=15);save(fig,'mode_mass')

fig,axs=plt.subplots(8,3,figsize=(13,30),layout='constrained')
for row,c in enumerate(C):
 name=f'{c}_s0';t=T[name];m=METRICS[name]
 hist(axs[row,0],t['b'],title=f'{LAB[c]}\nProposal: {len(t["b"]):,} candidates')
 hist(axs[row,1],t['b'],t['w'],title=f'Weighted teacher | ESS {m["ess"]:.0f}\nmode TV {m["teacher_mode_mass_tv"]:.3f}')
 hist(axs[row,2],S[name]['samples'],title=f'Actor: 32,768 raw samples\nmode TV {m["mode_mass_tv"]:.3f}')
fig.suptitle('Proposal -> weighted teacher -> actor | seed 0, 75K\nIndependent diagnostic teacher; 160 x 160 bins, same log density scale, no smoothing',fontsize=15);save(fig,'teacher_pipeline')

fig,axs=plt.subplots(8,2,figsize=(13,25),layout='constrained')
vmax=max(float(T[f'{c}_s0'][k].max()) for c in C for k in ['assignment_raw','assignment_sorted'])
for row,c in enumerate(C):
 t=T[f'{c}_s0'];method=RUN[f'{c}_s0']['cfg']['method'];term={'legacy':'hard argmax / N','monge':'hard empirical Monge / N','v5_ot':'Sinkhorn R / N','gmm':'w * responsibility'}[method]
 for col,k in enumerate(['assignment_raw','assignment_sorted']):
  a=t[k];ax=axs[row,col];im=ax.imshow(np.ma.masked_equal(a,0),origin='lower',aspect='auto',norm=LogNorm(1e-8,vmax),cmap='viridis',interpolation='nearest')
  ax.set_facecolor('#ececec');ax.set(title=f'{LAB[c]} | {term}\n'+('Original sample order' if col==0 else 'Sorted by nearest target mode, then x1'),xlabel='Teacher column block',ylabel='Student row block')
fig.colorbar(im,ax=axs.ravel().tolist(),label='Joint assignment mass (shared log scale)',shrink=.5)
fig.suptitle('Effective assignment at 75K, seed 0 | common color range\nContiguous block sums to at most 128 x 128; sorting before aggregation; zeros grey',fontsize=15);save(fig,'assignment_views')
np.savez_compressed(AUX/'assignment_views.npz',**{f'{c}_{k}':T[f'{c}_s0'][k] for c in C for k in ['assignment_raw','assignment_sorted','sort_rows','sort_cols','assignment_shape']})

chosen=['target','legacy_reference','monge_matched','v5_ot_large','gmm_matched','gmm_large']
fig,axs=plt.subplots(2,len(chosen),figsize=(18,7),layout='constrained')
for row,mode in enumerate([0,13]):
 for col,c in enumerate(chosen):
  x=ref if c=='target' else S[f'{c}_s0']['samples'];delta=x-locs[mode];bins=np.linspace(-4,4,41);count=np.histogram2d(*delta.T,bins=bins)[0]
  im=axs[row,col].imshow(np.ma.masked_equal(count.T/32768/.2**2,0),origin='lower',extent=[-4,4,-4,4],cmap='magma',norm=NORM,interpolation='nearest')
  axs[row,col].set_facecolor('#090613');axs[row,col].add_patch(Circle((0,0),scales[mode,0],fill=False,color='cyan',lw=.9));axs[row,col].set(title=('Target reference' if c=='target' else LAB[c])+f'\nfixed mode {mode}',xlabel='x1 - target center',ylabel='x2 - target center')
fig.suptitle('Within-mode close-ups | preselected target modes 0 and 13 | 75K, seed 0\n32,768 raw samples; 40 x 40 bins in local [-4,4]^2; no local renormalization; cyan = 1 sigma',fontsize=15);save(fig,'within_mode')

fig,axs=plt.subplots(1,3,figsize=(17,5),layout='constrained')
for c in C:
 for ax,key in zip(axs[:2],['train_seconds','segment_wall_seconds']):
  x=np.array([METRICS[f'{c}_s{s}'][key] for s in range(4)]);y=np.array([METRICS[f'{c}_s{s}']['histogram_tv'] for s in range(4)])
  ax.scatter(x,y,color=COL[c],s=20,alpha=.5);ax.scatter(x.mean(),y.mean(),color=COL[c],s=80,label=LAB[c],marker='D')
 for s in range(4):axs[2].scatter(METRICS[f'{c}_s{s}']['mode_mass_tv'],METRICS[f'{c}_s{s}']['within_mode_cov_error'],color=COL[c],s=30)
for ax,label in zip(axs[:2],['Measured training seconds','Active segment wall seconds']):ax.set(xscale='log',xlabel=label,ylabel='Final histogram TV');ax.grid(alpha=.2)
axs[2].set(xlabel='Final mode-mass TV',ylabel='Within-mode covariance error');axs[2].grid(alpha=.2)
fig.legend(*axs[0].get_legend_handles_labels(),loc='outside lower center',ncol=4,fontsize=9)
fig.suptitle('Final quality versus cost | 75K, 4 seeds | training and end-to-end segment clocks separated\nWall time includes evaluation/checkpoints/startup; excludes waiting while other runs occupy GPU3',fontsize=14);save(fig,'quality_time')

# Data tables and conclusion facts generated from all seeds.
def fmt(c,k,d=3):
 x=AGG[c][k];return f'{x["mean"]:.{d}f} ± {x["sd"]:.{d}f}'
def table(cols, rows):return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)+'\n'
def image(name,caption):return f'![{caption}](figures/{name}.png)\n\n{caption}\n'
main=table(['방법 / N×M','Mode coverage /40 ↑','Mode-mass TV ↓','2D histogram TV ↓','중심 오차 ↓','공분산 오차 ↓'],[[LAB[c],fmt(c,'modes_covered',1),fmt(c,'mode_mass_tv'),fmt(c,'histogram_tv'),fmt(c,'within_mode_center_error'),fmt(c,'within_mode_cov_error')] for c in C])
time_table=table(['방법','학습 구간 시간(초)','평가·저장 포함 활성 시간(초)','Sliced W₂ ↓'],[[LAB[c],fmt(c,'train_seconds',1),fmt(c,'segment_wall_seconds',1),fmt(c,'sliced_w2')] for c in C])
teacher_table=table(['방법','Actor mode TV','Teacher mode TV','Assignment–teacher mode TV','Teacher ESS / M'],[[LAB[c],fmt(c,'mode_mass_tv'),fmt(c,'teacher_mode_mass_tv'),fmt(c,'assignment_teacher_mode_tv',5),f'{AGG[c]["ess"]["mean"]:.1f} / {RUN[f"{c}_s0"]["cfg"]["M"]}'] for c in C])
seed_table=table(['방법','Seed','Coverage','Mode TV','Histogram TV','중심 오차','공분산 오차','Shape 계산 mode 수'],[[LAB[c],s,METRICS[f'{c}_s{s}']['modes_covered']]+[f'{METRICS[f"{c}_s{s}"][k]:.4f}' for k in ['mode_mass_tv','histogram_tv','within_mode_center_error','within_mode_cov_error']]+[METRICS[f'{c}_s{s}']['within_mode_evaluated_modes']] for c in C for s in range(4)])
finished=datetime.datetime.fromtimestamp(max(v['complete']['time'] for v in RUN.values()),datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S KST')
train_ratio_small=AGG['v5_ot_matched']['train_seconds']['mean']/AGG['gmm_matched']['train_seconds']['mean'];train_ratio_large=AGG['v5_ot_large']['train_seconds']['mean']/AGG['gmm_large']['train_seconds']['mean']
wall_ratio_small=AGG['v5_ot_matched']['segment_wall_seconds']['mean']/AGG['gmm_matched']['segment_wall_seconds']['mean'];wall_ratio_large=AGG['v5_ot_large']['segment_wall_seconds']['mean']/AGG['gmm_large']['segment_wall_seconds']['mean']
text=fr'''# GMM40: Legacy·Monge·v5 OT·Direct GMM 비교

**32/32 runs 완료 — 8조건 × seeds 0–3, 각 75,000 actor updates.** 마지막 완료: {finished}. 이 문서는 최종 75K 결과와 전체 학습 궤적을 다룬다. W&B의 중간 segment 종료 상태가 아니라 모든 `COMPLETE.json`과 75K 최종 샘플을 확인했다. 실패 기록은 0개다.

**가장 잘 복구한 것은 성공 설정을 사용한 Legacy 2048×2048이다.** 4개 seed 모두 40개 mode를 유지했고 mode 내부 모양도 가장 정확했다. Direct GMM은 256×1024·2048×2048에서 모두 40개 mode를 복구하며 v5 Sinkhorn OT보다 빠르고 정확했다. 반면 Monge 256×1024는 학습 도중 많은 mode를 잃었다. 처음 예상했던 “Legacy는 mode 내부 정확도가 낮고 Monge가 더 정확할 것”은 이번 결과에서 지지되지 않았다.

## 0. 실험 설정과 그림 읽는 법

### 목표 분포

이것은 **2D stationary energy-sampling 실험**이다. 환경 interaction, learned critic, return은 없다. 고정된 40개 Gaussian mixture의 log-density를 Q로 사용한다.

$$
p_*(x)=\frac{{1}}{{40}}\sum_{{k=1}}^{{40}}\mathcal{{N}}(x;c_k,\sigma_*^2 I),\qquad Q(x)=\log p_*(x).
$$

중심은 Torch seed 0으로 `[-40,40]²`에서 생성한 동일한 40개 위치이고, 각 축의 표준편차는 `softplus(1)=1.31326`이다. Temperature와 density correction beta는 모두 1이다. 따라서 이상적인 Boltzmann 목표는 원래 GMM40과 같다. Target 샘플과 mode label은 **평가에서만** 사용한다.

| 공통 항목 | 설정 |
|---|---|
| Seeds / 학습 길이 | 0,1,2,3 / 75K updates |
| State batch / latent 차원 | 1 / 2 |
| Learning rate | 첫 50K: 3e-4, 이후: 1e-4; Adam state 유지 |
| 평가 | 0, 5K, …, 75K; 각 run당 실제 actor sample 32,768개 |
| 학습 segment | 15K씩 순환, actor·Adam·RNG checkpoint에서 재개 |
| 장비 / precision | RTX 3090 GPU3 한 장, CPU6–9; JAX highest FP32 matmul |
| Teacher weight | `softmax(Q(b_j) − log q_proposal(b_j))` |
| 비교 오차 표시 | 4 seed 평균 ± 표본 표준편차; confidence interval 아님 |

여기서 **N은 한 update의 student latent 수**, **M은 importance weighting 이전 candidate 수**다. N이 네트워크가 영구 보유하는 독립 mixture component 개수라는 뜻은 아니다. 평가에서는 N개 component를 재사용하지 않고 32,768개의 latent를 새로 뽑는다. v5/GMM은 conditional Gaussian noise도 함께 뽑는다.

| 조건 | N×M | Actor | 학습 loss / assignment |
|---|---|---|---|
| Legacy reference | 2048×2048 | deterministic implicit, 512×5 | 기존 Sinkhorn row-argmax MSE |
| Legacy matched | 256×1024 | 같은 Legacy | 같은 row-argmax MSE |
| Monge matched | 256×1024 | 같은 Legacy | teacher 대표점 256개에 대한 exact 2D bijection + MSE |
| v5 OT matched / large | 256×1024 / 2048×2048 | conditional Gaussian, 256×2 | Sinkhorn conditional NLL |
| Direct GMM small / matched / large | 16×64 / 256×1024 / 2048×2048 | v5와 같은 conditional Gaussian | marginal GMM NLL, OT 없음 |

**Legacy/Monge 설정:** unbounded `x=G(z)`, stratified Gaussian latent, anchor 없음, Gaussian KDE std는 15K까지 8→1로 감소 후 유지. Squared cost를 평균으로 정규화하고 Sinkhorn epsilon은 0.01→0.0001, iterations=300이다. Monge는 Sinkhorn 대신 weighted candidate를 systematic resampling하여 N개의 동일질량 대표점을 만들고 `linear_sum_assignment`로 최적 bijection을 푼다. 따라서 원래 M개의 임의 weight를 가진 teacher에 대한 exact Monge라고 부르지 않는다.

**v5/GMM 설정:** `x=50 tanh(μ(z)+σ(z)ε)`, IID Gaussian latent, sigma 초기값 0.5, log-sigma 범위 [-5,1]. Teacher는 conditional squashed Gaussian mixture이며 pre-tanh std floor=0.05다. v5 OT는 `tanh(μ)`를 source로 사용하고 normalized action 좌표의 squared cost, epsilon=0.1, iterations=100을 쓴다. Legacy의 epsilon과 좌표·cost 정규화가 다르므로 숫자만 직접 비교하면 안 된다. GMM은 같은 actor·proposal을 쓰고 loss만 marginal NLL로 변경한다.

**통제된 비교는 동일 N×M의 v5 OT 대 Direct GMM, Legacy 256×1024 대 Monge 256×1024다.** Legacy와 v5 사이에는 network 크기, proposal, 좌표, latent sampling 차이가 있어 recipe 전체 비교로 해석한다. Legacy는 기존 성공 설정을 참고했지만, 당시와 달리 highest precision을 사용했으므로 bitwise 재현은 아니다. `heechan`/`v5-gmm40`에서 성공한 v5 GMM40 전용 설정은 찾지 못했다. 이번 v5 결과는 기본 v5를 문서화한 방식으로 GMM40에 적용한 결과다.

### 지표 정의

- **Mode-mass TV:** 각 sample을 정답 Gaussian들의 posterior responsibility로 나누어 mode별 질량을 계산한 뒤, 정답 1/40과의 차이 절댓값 합에 1/2을 곱한다. 낮을수록 좋다. 두 mode가 정답 50:50인데 80:20이면 TV=0.3이다. 이 값은 mode 내부 모양을 평가하지 않는다.
- **Mode coverage:** 정답 mode 중심에서 Mahalanobis 거리 3 이내에 배정된 posterior 질량이 nominal mass의 25% 이상이면 해당 mode를 복구한 것으로 센다. 임계값은 `0.25/40=0.00625`다. **40/40이라도 mode 폭이나 위치가 정확하다는 뜻은 아니다.**
- **2D histogram TV:** `[-50,50]²`의 160×160 bins와 외부 영역 한 bin에서 actor sample과 독립 정답 sample의 질량 차이를 비교한다. 작은 값이 좋다. 이번 32,768개 평가에서는 정답 샘플끼리 비교해도 약 **0.176**이므로 0이 기준이 아니다. Mode-mass TV의 정답 sampling floor는 약 **0.0124**다.
- **Mode 내부 중심 오차:** posterior로 가중한 mode 중심과 정답 중심 사이의 거리를 정답 sigma로 나눈다. **공분산 오차:** 정답 sigma로 정규화한 covariance eigenvalue들이 1에서 얼마나 벗어나는지 평균한다. 둘 다 낮을수록 좋다. 해당 mode 질량이 nominal의 10% 이상인 mode만 포함하며, 빠진 mode 수를 별도로 표시한다. 정답 샘플 floor는 중심 0.041, 공분산 0.044다.
- **Sliced W₂:** 고정된 32방향에 projection한 sample과 정답 sample 사이의 Wasserstein 거리. 물리 좌표 단위다. 한 지표만으로 승패를 정하지 않는다.

**그림은 별도 KDE smoothing이나 actor density 적분 없이 실제 샘플 histogram으로 그렸다.** Overview의 정답 패널도 독립 정답 sample 32,768개다. Seed 0 그림은 미리 고정한 예시이고 아래에 전체 32개 seed를 모두 제공한다. 학습 곡선은 seed 평균±SD이며 samples를 seed 간 섞어 하나의 policy처럼 표시하지 않았다.

{image('density_overview','그림 1. 전체 최종 분포: 75K, seed 0, 패널당 32,768개 실제 샘플, 160×160 bins, smoothing 없음. 모든 패널에 동일한 축·log-density 색 범위를 사용했다. 청록 원은 정답 mode 중심이다.')}

## 1. 최종 결과: mode 질량과 mode 내부 정확도를 분리해서 보기

{main}

**Legacy 2048×2048은 mode coverage뿐 아니라 중심·공분산 정확도까지 가장 좋다.** 특히 공분산 오차 0.095는 Direct GMM 2048×2048의 2.148보다 훨씬 작다. “Legacy가 각 mode를 잘 찾아도 mode 내부는 부정확할 것”이라는 가설은 이 성공 설정에서 성립하지 않는다.

**Direct GMM은 충분한 N×M에서 v5 OT보다 확실히 좋다.** 256×1024와 2048×2048 모두 네 seed 전부 40/40 coverage다. 하지만 histogram TV는 각각 0.439, 0.357로 Legacy reference 0.225보다 크다. 넓은 잔여 질량과 mode 내부 모양 오차가 남는다. 큰 GMM의 sliced W₂는 Legacy와 비슷하거나 약간 낮지만, 그것만으로 분포 복구가 더 정확하다고 판단할 수 없다.

**Legacy 256×1024와 Monge 256×1024는 mode를 잃는다.** Legacy는 32–36개, Monge는 6–13개를 유지했다. Monge의 중심 오차가 작아 보이는 부분은 살아남은 6–13개만 계산한 결과이며 전체 40개 복구를 뜻하지 않는다.

{image('mode_mass','그림 2. 40개 정답 mode별 최종 질량. 각 run에서 32,768개 sample을 평가한 뒤 4 seed 평균과 SD를 표시했다. 점선은 1/40. 큰 질량 집중을 숨기지 않도록 패널별 y축 범위는 다르다.')}

{image('within_mode','그림 3. 미리 정한 mode 0과 13의 확대 histogram. 75K seed 0, 전체 32,768개 sample 중 해당 창의 샘플을 표시하며 창 안에서 재정규화하지 않았다. 40×40 bins, smoothing 없음. 청록 원은 정답의 1-sigma 반경이다. 인접 mode가 가까우면 같은 창에 나타날 수 있다.')}

## 2. 언제 mode가 사라지고, 언제 복구되는가?

{image('learning_curves','그림 4. 5K 간격 학습 궤적, 4 seeds 평균±SD. 중심·공분산 지표는 충분한 mass가 있는 mode에서만 계산하므로 coverage와 함께 읽는다. Mode 수의 작은 출렁임에는 sampling 및 threshold 효과도 포함된다.')}

Legacy reference는 15K 시점에 네 seed 모두 40개를 잡고 최종까지 유지했다. Monge는 5K에서 평균 37.8개였지만 15K에 13.8개, 최종 9.25개로 감소했다. **처음부터 모든 mode를 못 찾았던 상황만으로 설명할 수 없다.** Teacher proposal 폭이 감소하는 구간과 mode 손실이 겹치지만 이 관찰만으로 annealing을 원인으로 확정하지 않는다.

Legacy 256×1024도 5K 평균 37.5개에서 최종 34.5개로 감소했다. 반대로 Direct GMM 256×1024는 5K 평균 31.8개에서 최종 40개로 증가했고, 2048×2048도 31.2→40으로 개선됐다. 작은 GMM 16×64는 평균 27개, v5 OT는 두 크기 모두 약 28개 부근에 남았다.

## 3. Teacher 문제인가, actor fitting 문제인가?

아래 teacher는 평가 때 actor로부터 **독립적으로 다시 생성한 단일 candidate set**이다. 실제 직전 training minibatch와 동일하지 않다. 특히 M=64의 teacher TV는 큰 Monte Carlo 변동을 포함한다. 이 숫자 하나로 persistent teacher bias나 오류의 최초 원인을 판정하지 않는다.

{teacher_table}

**Monge에서는 teacher도 이미 mode를 잃었다.** 최종 actor와 teacher의 mode TV가 모두 약 0.77이며, retained mode 중심의 proposal에 importance correction을 적용해도 없는 후보를 생성할 수는 없다. 원래 teacher와 Monge hard assignment의 mode 차이는 약 0.050이다. 이는 최종 상태에서 proposal/teacher 제한도 함께 존재한다는 증거다. “정확한 assignment만 풀면 global mode allocation이 해결된다”는 주장은 지지되지 않는다.

**v5 Sinkhorn은 marginal 수렴 실패만으로 성능을 설명하기 어렵다.** 실제 loss에 들어간 R/N의 column 질량과 teacher 사이 L1 residual은 256×1024에서 약 5.2e-6, 2048×2048에서 약 2.9e-6이다. 그러나 actor는 넓게 퍼져 있고 공분산 오차가 약 8.3–8.6이다. Global teacher 질량을 맞춘 coupling도 각 conditional Gaussian의 적절한 specialization과 최종 sample 정확도를 보장하지 않는다는 해석과 부합한다. Epsilon 자체의 인과 효과는 이번 비교만으로 분리하지 못한다.

**GMM 16×64의 한 번의 teacher는 매우 거칠다.** Teacher TV는 0.667인 반면 actor TV는 0.155이다. 여러 update를 통합한 actor가 단일 teacher draw보다 정확할 수 있으므로, teacher TV가 actor TV보다 크다는 사실 자체는 이상이 아니다. 후보 수를 키우면 teacher TV·actor TV·내부 모양 모두 개선된다.

{image('teacher_pipeline','그림 5. 모든 조건의 proposal → importance-weighted teacher → actor. 75K seed 0; proposal/teacher sample 수는 M이고 actor는 32,768개다. 160×160 bins, 동일 log-density scale, smoothing 없음. Sparse teacher histogram을 연속적인 밀도 추정으로 해석하지 않는다.')}

## 4. Assignment heatmap 읽는 법

Heatmap의 행은 해당 update에서 뽑힌 student latent, 열은 teacher candidate다. Latent는 update마다 바뀌므로 같은 row index를 영구적인 component identity로 해석하지 않는다. 아래는 각각 다른 학습 결과에서 나온 diagnostic teacher이므로, 모든 방법에 동일한 cloud를 준 counterfactual 그림은 아니다.

- **Legacy:** soft Sinkhorn plan 자체 대신 실제 회귀 target을 결정한 `1[argmax(row)=j]/N`을 표시한다. 여러 row가 같은 column을 골라도 된다. 원래 soft plan은 서버의 assignment NPZ에 보관되어 있다.
- **Monge:** resampling한 N개 teacher 대표점에 대한 bijection을 원래 M개 column으로 되돌려 `count/N`으로 표시한다. 동일 candidate가 대표점으로 여러 번 뽑힐 수 있다.
- **v5 OT:** row-normalized R/N이다. Row 총질량은 1/N이고 conditional NLL의 supervision을 정의한다.
- **GMM:** OT plan이 아니라 `w_j × gamma_ij`다. gamma는 candidate j에 대한 conditional Gaussian i의 posterior responsibility다. Column 합은 w_j지만 row 합은 균등할 필요가 없다.

{image('assignment_views','그림 6. 각 방법의 원본 sampling 순서와 정렬본을 함께 표시했다. 75K seed 0. 정렬은 가까운 정답 mode index, 그다음 x1 순이며 정렬 후 최대 128×128 contiguous block sum으로 표시했다. 두 보기와 모든 방법에서 같은 색 범위를 사용한다. 회색은 0이다. Block 안의 세부 sparsity는 합쳐지므로 full matrix와 동일한 해상도는 아니다.')}

정렬본이 mode별 block처럼 보이는 것은 **할당 구조를 시각적으로 정렬했기 때문**이기도 하다. Teacher가 일부 mode만 가지고 있어도 block은 나타난다. 또한 row별 Gaussian fitting 오차와 global mode 질량은 block 유무만으로 판단할 수 없다. 따라서 histogram, teacher support, 실제 mode mass를 함께 봐야 한다.

## 5. 계산 비용: Direct GMM의 장점은 뚜렷하다

{time_table}

동일 v5 actor의 matched 비교에서 Direct GMM은 Sinkhorn OT보다 학습 구간 기준 **{train_ratio_small:.1f}배(256×1024), {train_ratio_large:.1f}배(2048×2048)** 빨랐다. 평가·checkpoint·segment 초기화를 포함한 활성 시간 기준으로는 **{wall_ratio_small:.1f}배, {wall_ratio_large:.1f}배**다. 작은 batch=1 sampling 실험의 실제 측정이며 MuJoCo의 시간 배수로 그대로 옮길 수 없다.

학습 시간은 각 update block에서 actor params의 GPU 완료를 기다린 실측 누적값이다. Compilation도 포함하고 평가 시간은 제외한다. 활성 wall time은 15K segment 다섯 개의 시간 합이다. 다른 run 차례를 기다린 시간은 제외하므로 queue 전체 경과시간과 다르다.

{image('quality_time','그림 7. 시간 대비 최종 histogram TV 및 mode 질량–내부 공분산 오차. 작은 점은 각 seed, 왼쪽 두 패널의 큰 마름모는 4 seed 평균. 최종 75K 품질 비교이며 서로 다른 stopping rule의 time-to-target 측정은 아니다.')}

## 6. 이번 실험에서 가져갈 결론

1. **Legacy 성공 설정은 mode seeking만 잘하는 방법으로 축소하면 안 된다.** 이번에는 mode 질량과 내부 모양을 함께 가장 잘 복구했다. 다만 큰 network, stratified latent, proposal/epsilon schedule을 포함한 recipe 전체의 결과다.
2. **v5의 같은 actor·proposal에서는 Direct GMM이 기본 Sinkhorn conditional NLL보다 좋다.** 분포 복구와 계산 시간 모두에서 두 matched 크기에 일관된 차이가 있다. 이는 exact OT 전체에 대한 반증은 아니다. 이번 v5 OT는 entropic Sinkhorn 하나이고, Monge는 deterministic actor를 쓰는 별도 비교다.
3. **GMM의 N×M 의존성이 다시 보인다.** 16×64는 뚜렷하게 부족했고 256×1024부터 네 seed 모두 40개 mode를 복구했다. 큰 GMM은 더 정확하지만 Legacy reference 수준의 내부 모양까지 도달하지는 못했다. N과 M을 함께 변경했으므로 어느 쪽의 효과인지 분리하지 않는다.
4. **Empirical Monge를 정확하게 풀어도 학습 과정의 mode 소실은 막지 못했다.** 이 설정에서는 teacher/actor가 함께 좁아졌으며, resampling과 actor-proposal feedback도 분석할 필요가 있다. 남은 mode의 모양만 비교하면 전체 실패가 가려진다.

다음 분석을 선택한다면 Legacy 256×1024와 Monge의 mode 손실 직전 teacher·대표점 질량을 비교하고, v5의 cost scale/epsilon과 conditional row별 target spread를 확인하는 것이 이번 결과와 직접 연결된다. 이 보고서를 위해 새로운 학습 run은 추가하지 않았다.

## 부록 A. 전체 32개 최종 histogram

{image('all_seeds','그림 A1. 8조건×4seeds 전체. 각 panel은 독립된 run의 75K 결과로 32,768개 실제 sample, 160×160 bins, smoothing 없음. Overview와 같은 축·색 범위다. 실패해 보이는 seed를 제외하지 않았다.')}

## 부록 B. Seed별 수치

{seed_table}

Mode 내부 지표는 Monge에서 6–13개 mode, Legacy matched에서 32–36개 mode만 계산했다. v5/GMM에서는 mass threshold상 40개가 계산 대상이지만 coverage threshold는 만족하지 못할 수 있다. 두 threshold는 다르다.

## 부록 C. 재현·검증·보관

- Numerical source commit: `{manifest['source_commit']}` (`heejoon`, 실행 전 push).
- Source hash와 configuration: [SOURCE_MANIFEST](data/SOURCE_MANIFEST.json), [실험 protocol](data/PROTOCOL.md), [32개 run 설정 및 W&B 링크](data/run_manifest.json).
- 최종 32개 sample archive SHA-256를 원격 원본과 대조했다. Sample에서 주요 지표를 다시 계산해 저장값과 최대 오차 1e-5 이내임을 확인했다. 각 run의 0–75K 평가 16개 및 15K segment 5개를 확인했다. [검증 기록](data/verification.json).
- 집계/seed별 수치: [summary.json](data/summary.json). 실제 histogram count와 bin edge: [histograms_75k.npz](data/histograms_75k.npz). Assignment block sums 및 정렬 index: [assignment_views.npz](data/assignment_views.npz).
- 평가 RNG: actor는 `PRNGKey(1000000+seed)`에서 latent/noise를 생성; teacher diagnostic은 `PRNGKey(2000000+seed+update)`; reference는 NumPy seed 20260919. 평가 RNG는 training RNG와 독립이다. 원시 sample은 실제 actor checkpoint에서 저장한 32,768개이며 이후 resmoothing하지 않았다.
- 중앙 원본: `dildata:/data1/heejoonorm/OptiQ/studies/20260919_legacy_monge/remote_3114850247/gmm40/3f34dbedca70/`. Actor/optimizer/RNG checkpoint `latest.msgpack`, 전체 sample·teacher·assignment·history를 보관한다. [32개 최종 checkpoint와 sample hash 보관 검증](data/FINAL_BACKUP_VERIFIED.json).
- 실험 서버 원본: `/home/heejoonorm/OptiQ/legacy_monge/gmm40/3f34dbedca70/`.
- 리포트: `dildata:/data1/heejoonorm/OptiQ/reports/20260920_gmm40_comparison/` 및 로컬 같은 report 폴더. **report.html은 모든 그림과 수식을 내장한 단일 파일**이라 인터넷 없이 열 수 있다. Markdown 원문은 report.md이며 그림은 figures/에 있다.
'''
(OUT/'report.md').write_text(text)
# Render math to local SVG-free PNG for offline HTML; MD keeps original LaTeX intact.
math_count=0

def math_image(match):
 global math_count
 formula=match.group(1).strip().replace('\n',' ');math_count+=1
 buffer=io.BytesIO();fig=plt.figure(figsize=(.2,.2));artist=fig.text(0,0,'$'+formula+'$',fontsize=14)
 fig.savefig(buffer,format='png',dpi=160,bbox_inches='tight',pad_inches=.08,transparent=True);plt.close(fig)
 encoded=base64.b64encode(buffer.getvalue()).decode();return '\n<p class="equation"><img alt="'+html.escape(formula,quote=True)+'" src="data:image/png;base64,'+encoded+'"></p>\n'
htmltext=re.sub(r'\$\$(.*?)\$\$',math_image,text,flags=re.S)
body=markdown.markdown(htmltext,extensions=['tables','fenced_code','toc'])
for f in FIG.glob('*.png'):
 body=body.replace('src="figures/'+f.name+'"','src="data:image/png;base64,'+base64.b64encode(f.read_bytes()).decode()+'"')
style='''body{font-family:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",Arial,sans-serif;max-width:1180px;margin:45px auto;padding:0 25px;line-height:1.8;color:#182632;background:#fff}h1{font-size:32px;border-bottom:4px solid #087e86;padding-bottom:18px}h2{margin-top:56px;padding:10px 0;border-bottom:1px solid #bacbd2}h3{margin-top:30px}img{max-width:100%;height:auto}table{display:block;overflow-x:auto;border-collapse:collapse;font-size:13px;margin:22px 0}th,td{border:1px solid #dbe3e8;padding:8px 11px;text-align:left;white-space:nowrap}th{background:#eff6f8}tr:nth-child(even){background:#fafcfd}code{background:#eef2f5;padding:2px 4px;border-radius:3px;overflow-wrap:anywhere}a{color:#086e88}.equation{text-align:center;margin:26px 0}.equation img{max-height:80px}li{margin:9px 0}@media print{body{max-width:none;margin:0}h2{break-before:auto}img,table{break-inside:avoid}}'''
(OUT/'report.html').write_text('<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GMM40 — 32-run final report</title><style>'+style+'</style></head><body>'+body+'</body></html>')
print(json.dumps({'report':str(OUT),'runs':32,'figures':len(list(FIG.glob('*.png'))),'html_MB':(OUT/'report.html').stat().st_size/1e6,'math':math_count},indent=2))
