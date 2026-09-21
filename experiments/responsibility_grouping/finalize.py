"""Render the pre-registered analysis; no clustering refit or training."""
import argparse,base64,collections,datetime,hashlib,html,io,json,re,shutil
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import markdown

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args();O=a.out
s=json.loads((O/'summary.json').read_text());toy=s['toy'];rl=s['rl']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):
 for ext in ['png','pdf']:fig.savefig(O/'figures'/f'{name}.{ext}',dpi=170,bbox_inches='tight')
 plt.close(fig)

# Make eligible/ineligible cuts explicit. This does not alter any selected partition.
fig,axs=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
for col,recs in enumerate([[r for r in toy if r['family']=='double'],[r for r in toy if r['family']=='tri'],[r for r in rl if r['state']==0]]):
 for i,r in enumerate(recs):
  sc=r['scores'];x=np.array([v['K'] for v in sc]);y=np.array([v['silhouette'] for v in sc]);ok=np.array([v['eligible'] for v in sc]);color=f'C{i}'
  label=f'seed {r["seed"]}' if col<2 else f'{r["step"]:,} steps'
  axs[col].plot(x,y,'-',alpha=.7,color=color,label=label);axs[col].scatter(x[ok],y[ok],marker='o',color=color,s=27);axs[col].scatter(x[~ok],y[~ok],marker='x',color=color,s=38)
  if r['K']>1:axs[col].scatter([r['K']],[r['silhouette']],marker='*',s=160,facecolors='none',edgecolors=color,linewidths=1.5,zorder=5)
 axs[col].axhline(.25,c='gray',ls='--',lw=1);axs[col].set(xlabel='Requested groups K',ylabel='Silhouette',title=['2D double','2D tri','17D state 0'][col]);axs[col].grid(alpha=.2);axs[col].legend(fontsize=8)
fig.suptitle('Same pre-specified rule in every case\nCircle: passes minimum size | cross: too-small group | outlined star: selected K');save(fig,'selection_scores')

# Gamma is the clustering input; R is the user's familiar weighted, row-normalized view.
fig,axs=plt.subplots(2,3,figsize=(14,9),layout='constrained');heat=plt.get_cmap('magma').copy();heat.set_bad('black')
for row,(tag,title) in enumerate([('toy_double_s0','2D double, seed0'),('rl_650000_s0','17D Humanoid 650K, state0')]):
 z=np.load(O/'arrays'/f'{tag}.npz');ro=z['row_seriation'];co=z['col_seriation']
 for col,key in enumerate(['gamma','R']):
  im=axs[row,col].imshow(np.ma.masked_less_equal(z[key][ro][:,co],0),origin='lower',aspect='auto',interpolation='nearest',cmap=heat,norm=LogNorm(1e-5,1));axs[row,col].set(title=title+'\n'+['Posterior gamma (column sums = 1)','Teacher assignment R (row sums = 1)'][col],xlabel='Candidate, responsibility order',ylabel='Student, responsibility order');fig.colorbar(im,ax=axs[row,col],shrink=.6)
 im=axs[row,2].imshow(1-z['distance'][co][:,co],origin='lower',aspect='equal',cmap='viridis',vmin=0,vmax=1,interpolation='nearest');axs[row,2].set(title='Candidate similarity: 1 - Hellinger distance',xlabel='Candidate, same order',ylabel='Candidate, same order');fig.colorbar(im,ax=axs[row,2],shrink=.6)
fig.suptitle('Same candidates and same ordering | clustering uses gamma, not weighted R');save(fig,'posterior_vs_teacher')

def f(v):return f'{v:.3f}'
toytable='\n'.join(f'| {r["family"]} | {r["seed"]} | {r["K"]} | {f(r["silhouette"])} | {f(r["true_basin_ARI"])} | {f(r["stability_mean_ARI"])} | {f(r["joint_histogram_TV"])} |' for r in toy)
rlrows=[]
for step in [50000,650000]:
 rr=[r for r in rl if r['step']==step];bad=sum(not any(v['eligible'] for v in r['scores']) for r in rr);low=sum(r['K']==1 and any(v['eligible'] for v in r['scores']) for r in rr);acc=[r for r in rr if r['K']>1]
 rlrows.append(f'| {step:,} | {len(acc)}/128 | {bad} | {low} | '+', '.join(f'state {r["state"]}: K={r["K"]}, silhouette={r["silhouette"]:.3f}' for r in acc)+' |')
rltable='\n'.join(rlrows)
massrows=[]
for step in [50000,650000]:
 rr=[r for r in rl if r['step']==step]
 massrows.append(f'| {step:,} | {np.mean([r["teacher_ESS"] for r in rr]):.2f} | {np.mean([r["usage_ESS"] for r in rr]):.2f} |')
text=r'''# Responsibility로 candidate 묶기: 2D 시각 검증과 17D RL checkpoint

**2D의 두 mode는 정답 경계를 주지 않고도 4개 seed 모두 두 묶음으로 복구했다. 세 mode는 일부 seed에서 합쳐지거나 더 나뉘었다. 17D에서는 responsibility 정렬로 담당 영역이 더 잘 보이지만, 이번 자동 그룹 선택 규칙으로 뚜렷한 분할을 채택한 상태는 checkpoint별 128개 중 1개뿐이었다.**

이 결과는 responsibility 기반으로 학습된 담당 구조를 찾아낼 가능성을 보여준다. 동시에 **그 구조를 그대로 Q의 mode라고 정의하고 gradient를 차단하기에는 아직 부족하다.** 이번에는 기존 checkpoint를 읽어 분석했으며 actor·critic update는 0회다. MuJoCo mode-selection 학습도 새로 시작하지 않았다.

## 0. 무엇을 분석했나?

| 항목 | 2D toy | 17D RL |
|---|---|---|
| 데이터 | 기존 double/tri mass-shift 실험의 마지막 checkpoint | TRG Direct GMM Humanoid seed 0 |
| 비교 시점 | 35K actor updates, 목표 질량이 다시 균등해진 시점 | 50K / 650K environment steps |
| 학습 seed / 상태 | 두 환경 각각 seeds 0–3, 총 8개 snapshot | 50K replay에서 저장한 동일한 128개 probe states |
| N×M | 256×1024 | 64×64 |
| actor | v5 conditional squashed Gaussian, hidden 256×2, log sigma [-5,1] | TRG action-box truncated Gaussian, hidden 256×2, log sigma [-5,-1] |
| temperature | 0.25 | 0.25 |
| critic | 정해진 Q의 변화를 학습하던 toy | single critic, TD action MC K=64, live critic teacher |
| 그림용 actor samples | 저장된 실제 32,768개 action | 저장된 candidate-level matrix 사용 |
| 이번 추가 학습 | 없음 | 없음 |

2D 목표는 첫 좌표에만 여러 mode를 두고, 둘째 좌표는 독립적인 폭 0.35의 Gaussian으로 구성한 기존 설정이다. 두 mode는 중심 −0.65,+0.65, 폭 0.12이고, 세 mode는 중심 −0.6,0,+0.6, 폭 0.1이다. 아래 density를 [-1,1]²에서 정규화한다.

$$f(a_1,a_2)=\left[\frac{1}{L}\sum_{m=1}^{L}\mathcal{N}(a_1;c_m,h^2)\right]\mathcal{N}(a_2;0,0.35^2),\qquad Q=0.25\log f.$$

따라서 이것은 **분리하기 쉬운 기존 2D 사례**다. 두 좌표 모두 복잡하게 얽힌 새로운 2D target을 학습한 것은 아니다. 또한 2D와 17D는 actor 종류, N×M, 학습 문제까지 다르므로 결과 차이를 차원 수 하나의 효과로 해석하면 안 된다. 650K에서도 초기 replay의 동일한 상태들을 사용했으므로 후기 policy의 방문 상태를 대표한다고 보장하지 않는다.

## 1. 그룹을 찾는 데 사용한 정보

Student i의 conditional density를 kᵢ, teacher candidate를 bⱼ라 하면 posterior responsibility는 다음과 같다.

$$\gamma_{ij}=\frac{k_i(b_j)}{\sum_{\ell=1}^{N}k_\ell(b_j)},\qquad v_j=(\gamma_{1j},\ldots,\gamma_{Nj}).$$

두 candidate를 비슷한 student들이 설명하면 두 벡터가 비슷하다. 이 벡터 사이의 **Hellinger distance**를 계산하고, average-linkage hierarchical clustering을 적용했다. 단순 Euclidean distance 대신 확률벡터의 제곱근 좌표에서 거리를 계산한 것이다.

$$d(j,k)=\frac{1}{\sqrt{2}}\left\|\sqrt{v_j}-\sqrt{v_k}\right\|_2.$$

**그룹 membership을 계산할 때 action 좌표, Q값, importance weight, 정답 mode 경계·개수는 넣지 않았다.** Action 좌표는 완성된 묶음을 공간에 표시하고 색 번호를 왼쪽부터 붙이는 데만 사용했다. 정답 경계는 결과를 평가할 때만 사용했다. 이전 oracle-basin mode-selection 학습과 다른 지점이다.

2D에서는 pre-tanh Gaussian의 log density로 posterior를 복구했다. tanh Jacobian은 같은 candidate에서 모든 component에 공통이므로 posterior에서 소거된다. 17D에서는 box-truncated Gaussian의 정규화 상수를 포함했다. 두 경우 모두 float64 log-domain으로 posterior를 계산한 뒤 **동일한 grouping 함수**를 적용했다.

### 그룹 수를 어떻게 정했나?

분석 실행 전에 다음 규칙을 고정했다. 결과를 보고 임계값을 수정하지 않았다.

1. K=2,…,8로 계층을 잘라 후보 partition을 만든다.
2. 모든 그룹이 최소 max(2, ceil(0.01M))개의 candidate를 포함해야 한다.
3. 이 조건을 통과한 partition 중 평균 silhouette가 가장 큰 K를 고른다. 동률이면 작은 K를 고른다.
4. 최고 silhouette가 0.25 미만이거나 크기 조건을 모두 실패하면 **분할 보류**, 저장상 K=1로 둔다.

Silhouette는 같은 그룹 안의 평균 거리 aⱼ와 가장 가까운 다른 그룹까지 평균 거리 bⱼ를 비교한다.

$$S_j=\frac{b_j-a_j}{\max(a_j,b_j)},\qquad S=\frac{1}{M}\sum_j S_j.$$

이 규칙과 0.25는 탐색용 기준이며 mode 개수의 통계적 보증이 아니다. **K=1로 보류됐다는 것은 실제 Q가 unimodal이라는 뜻이 아니다.** 작은 고립 그룹 때문에 partition 전체가 탈락할 수도 있다. 그림에는 이산 그룹을 강제로 선택하지 않는 responsibility 순서도 제공했다. 이는 계층의 leaf 순서를 인접 거리 기준으로 바꾼 것이다. [Silhouette 정의](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html), [optimal leaf ordering](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.optimal_leaf_ordering.html).

## 2. Heatmap과 student 색은 무엇인가?

그룹 계산에 넣은 것은 위의 **γ**다. 반면 기존 보고서와 같은 heatmap의 색은 importance-weighted assignment를 행별 정규화한 **R**이다. 두 행렬을 혼동하면 안 된다.

$$J_{ij}=w_j\gamma_{ij},\qquad \alpha_i=\sum_jJ_{ij},\qquad R_{ij}=\frac{J_{ij}}{\alpha_i}.$$

γ의 각 **열**은 합이 1이고, R의 각 **행**은 합이 1이다. R의 밝은 칸은 해당 student가 받는 teacher 질량 중 그 candidate의 비중이 크다는 뜻이다. 사용량 αᵢ가 작은 student도 행 정규화 후에는 밝게 보일 수 있다.

발견된 candidate 그룹을 Cₘ이라 하면, student의 담당 색은 그룹별 R의 합으로 정했다.

$$H_{im}=\sum_{j\in C_m}R_{ij},\qquad m_i=\arg\max_m H_{im}.$$

이는 **표시용 할당**이다. Gradient masking이나 confidence weighting을 적용한 학습 결과가 아니다. Student는 이번 snapshot의 latent sample에 해당하며 영구적인 component ID가 아니다.

## 3. 2D: 좌표를 사용하지 않고도 공간적인 분리가 나오는가?

![2D 대표 seed의 actor, candidate 그룹, student 및 heatmap](figures/toy_2d_main.png)

대표 그림은 사전에 정한 seed 0이다. 왼쪽부터 다음을 표시했다.

- **Actor histogram:** 실제 32,768개 action을 64×64 bins로 센 density. KDE smoothing 없이 그렸다. 회색선은 정답 target의 20%,50%,80% peak-level contour다.
- **Candidate 색:** 오직 posterior 벡터에서 찾은 그룹. 점은 teacher 후보이며 actor histogram과 동일한 sample set은 아니다.
- **Student 중심:** tanh(μᵢ)를 표시했다. Conditional Gaussian의 중심이지 squashed action의 정확한 평균은 아니다. 둘째 좌표의 폭은 sigma로 표현할 수 있으므로 중심이 y≈0에 모인 것만으로 mode collapse라고 판단하지 않는다.
- **R heatmap:** 원래 값은 그대로 두고 그룹 및 responsibility 유사도 순으로 정렬했다. 위 색띠는 candidate group, 왼쪽 색띠는 student의 담당 group이다.

**Double seed 0:** 두 공간적 mode가 두 responsibility 그룹과 잘 맞는다. Teacher 질량도 그룹별 약 50.2%/49.8%다. Student 256개 중 135/121개가 각각의 그룹을 가장 많이 담당한다.

**Tri seed 0:** 왼쪽과 중앙 mode가 하나의 그룹으로 묶인다. Actor histogram에 세 봉우리가 보여도 자동 partition이 반드시 세 개가 되는 것은 아니다. 이 경우 그룹별 teacher 질량은 약 69.8%/30.2%다. 선택된 K=2 partition의 silhouette는 0.334이고 K=3의 0.304보다 높았다. K≥4에서는 작은 그룹이 생겨 크기 조건을 통과하지 못했다.

### 성공한 그림만 고르지 않고 네 seed 모두 보기

![2D 전체 네 seed](figures/toy_all_seeds.png)

**정답 basin ARI**는 발견한 그룹과 알려진 action-space basin을 사후 비교한 값이다. 1이면 같은 partition, 0 부근이면 우연 수준의 일치다. Basin 경계는 clustering에 제공하지 않았다. **Subset ARI**는 같은 candidate를 유지하면서 기존 student rows의 80%만 남겼을 때 partition이 얼마나 유지되는지다. 서로 다른 5개 subset에서 계산한 평균이며, 새로운 latent·candidate를 독립적으로 다시 뽑은 실험은 아니다.

| 환경 | Seed | 선택 K | Silhouette | 정답 basin ARI | Student-subset ARI | Actor joint histogram TV |
|---|---:|---:|---:|---:|---:|---:|
__TOYTABLE__

Double은 **4/4 seeds에서 K=2**, basin ARI 평균 **0.931**, subset ARI 평균 **0.962**다. 이 쉬운 문제에서는 의미 있는 담당 영역을 안정적으로 찾아냈다.

Tri는 **K=2,3,4,3**으로 갈린다. Seed 3은 세 공간적 mode와 잘 일치하지만, seed 1은 K=3이어도 공간 경계가 어긋난다. Seed 2의 추가 그룹은 teacher 질량 약 **1.69%**, 담당 student **1개**인 작은 그룹이다. 단순히 K가 정답과 같은지만 보면 이런 차이를 놓친다.

Histogram TV는 actor의 전체 2D 분포와 target의 bin mass 차이 절반의 합이다. ARI와 측정 대상이 다르다. 이번 grouping이 actor의 TV를 개선한 것은 아니다.

## 4. 17D Humanoid: 정렬은 도움이 되지만 자동 mode 분리는 아직 어렵다

![17D 동일 상태 두 checkpoint](figures/rl_main.png)

위는 50K, 아래는 650K의 **동일한 probe state 0**이다. 각 행은 같은 R을 원본 순서 / 행동 PC1 순서 / responsibility 순서 / 채택한 그룹 순서로 보여준다. 모든 R heatmap은 동일한 logarithmic color scale 1e-5–1이다. 이 그림의 PC1은 각 checkpoint에서 중심과 candidate를 합쳐 계산한 표시용 축이며 clustering 입력이 아니다.

Responsibility 정렬을 쓰면 비슷한 candidate를 담당하는 행들이 더 모여 보인다. 그러나 이 대표 상태에서는 두 checkpoint 모두 그룹 수를 보류했다. K=2,…,8의 모든 cut에 singleton 그룹이 남았고, raw silhouette 최고값도 각각 **0.151/0.163**으로 낮았다. **“담당 관계를 보기 쉬운 순서”와 “학습에 사용할 신뢰할 만한 이산 mode 경계”는 다른 결과다.**

### 동일한 128개 상태 전체

| Checkpoint | 분할 채택 | 모든 K에서 최소 크기 실패 | 크기 통과 K는 있지만 silhouette 부족 | 채택된 상태 |
|---|---:|---:|---:|---|
__RLTABLE__

각 checkpoint에서 127/128개 상태가 분할 보류다. 이를 “127개 상태가 단일 mode”라고 읽으면 안 된다. 특히 minimum-size 규칙은 outlier 하나만 있어도 큰 그룹들의 유용한 분할까지 거절할 수 있다. 한편 크기 조건을 통과하고도 silhouette가 낮은 상태도 많아, 이 결과를 크기 기준 하나의 탓으로 돌릴 수 없다. 이 분석에서는 기준을 바꿔 좋은 그림이 나오도록 맞추지 않았다.

첫 8개 상태의 subset ARI 평균은 50K에서 1.000, 650K에서 0.975지만, **대부분 원래와 subset 모두 K=1로 보류된 결과**다. 높은 ARI를 성공적인 다중 mode 발견의 증거로 쓰지 않는다. 채택된 state 114/63은 첫 8개에 포함되지 않아 subset 안정성을 검사하지 않았다.

![50K 첫 8개 상태](figures/rl_first8_50000.png)

![650K 첫 8개 상태](figures/rl_first8_650000.png)

### Teacher weight와 그림의 한계

$$\mathrm{ESS}_{\mathrm{teacher}}=\frac{1}{\sum_jw_j^2},\qquad\mathrm{ESS}_{\mathrm{student}}=\frac{1}{\sum_i\alpha_i^2}.$$

| Checkpoint | Teacher ESS, 128 states 평균 | Student usage ESS, 128 states 평균 |
|---|---:|---:|
__MASSTABLE__

64개 candidate 중 teacher 질량은 소수 후보에 집중되어 있다. ESS는 mode 개수가 아니다. 또한 이번 grouping은 w를 입력하지 않으므로, teacher에서 거의 무시되는 candidate도 거리·군집 개수 선택에는 동등하게 참여한다. **Importance 집중이 clustering 실패를 직접 일으켰다고 검증한 것은 아니다.** 다만 그만큼 γ에서 찾은 패턴과 실제 학습을 주도하는 질량의 관계를 별도로 봐야 한다.

![Posterior와 teacher assignment 및 candidate 유사도](figures/posterior_vs_teacher.png)

왼쪽은 grouping의 입력 γ, 중앙은 기존 그림의 R, 오른쪽은 1−Hellinger distance다. 모든 열 순서는 같은 responsibility 순서다. 2D double에서는 큰 두 블록이 보이고, 17D 대표 상태에서는 더 복잡한 국소 담당 관계가 보인다. Posterior는 actor component의 상대 density만 반영하므로, Q를 바꾸어도 같은 actor와 같은 candidate에서는 γ가 변하지 않는다. 따라서 **γ만으로 모든 Q mode를 식별할 수 있다고 기대할 수는 없다.**

## 5. 선택 규칙과 재현성

![K별 점수 전체](figures/selection_scores.png)

원은 최소 크기를 통과한 cut, ×는 작은 그룹이 있어 탈락한 cut, 테두리 별은 채택한 K다. 점선은 사전에 둔 silhouette 0.25 기준이다. 이진 분할이든 K=3이든 강제로 지정하지 않았다.

이번 결과가 지지하는 해석은 **“분리된 담당 구조가 이미 형성되어 있으면 responsibility만으로 이를 드러낼 수 있다”**다. 2D double이 그 예다. 반면 actor의 담당 구조가 target mode와 어긋나거나 표본이 적으면 mode들을 합치거나 쪼갤 수 있다. 이 partition을 gradient masking에 바로 사용하면 잘못 합쳐진 mode끼리의 간섭은 남고, 잘못 나눈 mode에서는 필요한 gradient까지 제거할 수 있다.

현재 단계에서는 이 방법을 **담당 구조를 관찰하는 진단 도구**로 두는 것이 타당하다. 학습용 자동 selection으로 쓰려면 새 latent·candidate 재표집에 대한 안정성, 작은 고립 그룹 처리, w가 작은 후보의 영향부터 검증해야 한다. 이번 결과만으로 MuJoCo 성능 향상을 주장하지 않는다.

## 6. 검증·코드·보관

- 입력 36개 파일의 SHA256을 manifest와 비교했다.
- γ의 column sum, R의 row sum, J의 teacher marginal을 검사했다.
- 동일한 student만 있는 인공 예제는 K=1, 분리된 두 responsibility block은 K=2로 나오는지 검사했다.
- ARI identity/permutation과 student-row 순열에 대한 candidate 거리 불변성을 검사했다.
- 2D 실제 32,768 samples에서 다시 센 histogram이 저장된 8개 histogram 모두와 일치했다.
- 17D posterior 재구성의 최대 절대 오차는 **1.12e-6 미만**, R의 최대 오차는 **5.27e-5 미만**이다. Float64 재계산과 원래 float32 경로의 차이를 포함한다.
- 핵심 clustering 분석 commit: `__ANALYSIS_COMMIT__` (계산 전에 heejoon push).
- 시각화·리포트 생성 commit: `__REPORT_COMMIT__`.
- 2D 원래 학습 commit: `a08517ef9ed5fb8743252132997638b00feb5d75`.
- 17D 원래 학습 commit: `52ae6c4dd2502cb3384a410bf4197ad589306492`.
- [수치 결과](summary.json), [검증 기록](VALIDATION.json), [입력 manifest](INPUT_MANIFEST.json), [사전 분석 프로토콜](PROTOCOL.md), [계산 코드](analyze.py), [리포트 생성 코드](finalize.py).

로컬: `/Users/heejoon/Documents/ChatGPT/OptiQ/reports/20260921_responsibility_grouping/`.

중앙 리포트: `dildata:/data1/heejoonorm/OptiQ/reports/20260921_responsibility_grouping/`.

원본 입력·분석 배열: `dildata:/data1/heejoonorm/OptiQ/studies/20260921_responsibility_grouping/`.

Markdown에는 LaTeX 원문을 보존했다. HTML에는 그림과 수식을 모두 내장하여 인터넷 없이 열 수 있다. 배열 NPZ는 별도 보관한다.
'''
for key,val in {'__TOYTABLE__':toytable,'__RLTABLE__':rltable,'__MASSTABLE__':'\n'.join(massrows),'__ANALYSIS_COMMIT__':s['analysis_commit'],'__REPORT_COMMIT__':a.commit}.items():text=text.replace(key,val)
(O/'report.md').write_text(text)
def mathimage(m):
 expr=m[1].replace('\\operatorname','\\mathrm');fig=plt.figure(figsize=(.1,.1));fig.text(0,0,'$'+expr+'$',fontsize=15);b=io.BytesIO();fig.savefig(b,format='png',dpi=160,bbox_inches='tight',pad_inches=.1,transparent=True);plt.close(fig)
 return '<div class="math"><img alt="'+html.escape(m[1],quote=True)+'" src="data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()+'"></div>'
body=markdown.markdown(re.sub(r'\$\$(.*?)\$\$',mathimage,text,flags=re.S),extensions=['tables','fenced_code'])
def embed(m):
 path=O/m[1]
 return 'src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'"' if path.exists() else m[0]
body=re.sub(r'src="([^"<>]+\.png)"',embed,body)
css='body{max-width:1500px;margin:36px auto;padding:0 24px;font:16px/1.8 system-ui;color:#263448}h1{line-height:1.3}h2{margin-top:42px;border-top:1px solid #dce3eb;padding-top:20px}h3{margin-top:25px}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;border-bottom:1px solid #dce3eb;text-align:left}th{background:#f1f5f9}code{background:#f1f5f9;overflow-wrap:anywhere}.math{text-align:center;overflow:auto}a{color:#087b99}'
(O/'report.html').write_text('<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Responsibility grouping: 2D & 17D</title><style>'+css+'</style></head><body>'+body+'</body></html>')
for name in ['PROTOCOL.md','INPUT_MANIFEST.json','analyze.py','finalize.py']:shutil.copy2(a.source/name,O/name)
provenance=dict(analysis_commit=s['analysis_commit'],report_commit=a.commit,training_updates=0,input_count=36,generated_at=datetime.datetime.now().astimezone().isoformat())
(O/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2))
print(json.dumps({'report':str(O/'report.html'),'html_bytes':(O/'report.html').stat().st_size,'figures':len(list((O/'figures').glob('*.png')))}))
