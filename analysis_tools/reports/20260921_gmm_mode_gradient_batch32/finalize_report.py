import argparse,base64,html,io,json,math,re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();O=a.output
s=json.loads((O/'summary.json').read_text());rec={(r['batch'],r['n'],r['m'],r['method'],r['seed']):r for r in s['runs']}
methods=['baseline','mode_only','mode_confidence'];labels=['Direct GMM','Mode selection','Selection + confidence'];colors=['#07889b','#8554ae','#e36b35']
sizes=[(16,16),(64,64),(128,128),(256,256),(1024,1024),(2048,2048),(64,4096),(2048,4096)]
common={size:[seed for seed in range(4) if all((b,*size,m,seed) in rec for b in [1,32] for m in methods)] for size in sizes};common={k:v for k,v in common.items() if v}
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
def load(b,n,m,method,seed,name):
 with np.load(Path(rec[b,n,m,method,seed]['run_path'])/name) as f:return {k:f[k] for k in f.files}
def target(x):
 norm=sum((math.erf((1-c)/(.1*np.sqrt(2)))-math.erf((-1-c)/(.1*np.sqrt(2))))/6 for c in [-.6,0,.6])
 return sum(np.exp(-.5*((x-c)/.1)**2) for c in [-.6,0,.6])/(3*.1*np.sqrt(2*np.pi)*norm)
for group,start in [('small',0),('larger',3)]:
 subset=list(common)[start:start+3]
 if not subset:continue
 fig,axs=plt.subplots(2,len(subset),figsize=(4.7*len(subset),7.5),squeeze=False,sharey=True,layout='constrained')
 for col,(n,m) in enumerate(subset):
  seeds=common[n,m]
  for row,b in enumerate([1,32]):
   ax=axs[row,col];x=np.linspace(-1,1,1000);ax.plot(x,target(x),'--',c='#24334a',lw=1.8,label='Exact target')
   for method,label,c in zip(methods,labels,colors):
    ee=[load(b,n,m,method,seed,'eval_020000.npz') for seed in seeds];edges=ee[0]['edges'];density=np.array([e['histogram']/np.diff(edges) for e in ee])
    ax.stairs(density.mean(0),edges,color=c,lw=1.2,label=label)
   ax.set(title=f'Batch {b} | {n} x {m}\n20K; paired seeds {seeds}',xlabel='Action',ylabel='Density',xlim=(-1,1));ax.grid(alpha=.18)
 axs[0,0].legend(fontsize=9);fig.suptitle('32,768 sampled actions/run; 256-bin histograms; no smoothing',fontsize=14)
 fig.savefig(O/'figures'/f'density_{group}.png',dpi=150);plt.close(fig)
r=s['representative'];n,m,seed=r['n'],r['m'],r['seed']
for sort in [False,True]:
 fig,axs=plt.subplots(2,3,figsize=(14,6.8),layout='constrained')
 for row,b in enumerate([1,32]):
  for col,(method,label) in enumerate(zip(methods,labels)):
   d=load(b,n,m,method,seed,'diagnostic_020000.npz');order=np.argsort(d['training_z'][:,0]) if sort else np.arange(n);ax=axs[row,col]
   im=ax.imshow(d['H'][order].T,aspect='auto',origin='lower',interpolation='nearest',vmin=0,vmax=1,cmap='viridis')
   ax.set(title=f'{label} | B={b}',xlabel='Latent z rank' if sort else 'Original sample index',yticks=[0,1,2],yticklabels=['Left','Center','Right'])
 fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.9,label='Teacher-weighted mode fraction H');fig.suptitle(f'{n} x {m}, seed {seed}, 20K | B32 displays representative group 0')
 fig.savefig(O/'figures'/('responsibility_sorted.png' if sort else 'responsibility_raw.png'),dpi=150);plt.close(fig)
# Figures have now been regenerated from the fixed snapshot, not newer run results.
status='최종 보고서' if s['final'] else '중간 보고서 — 완료된 run만 비교'
template=r'''# 3-mode Direct GMM: batch 1 vs batch 32

**STATUS**

분석 기준: **STAMP**. Batch 1은 **96/96 완료**, batch 32는 **COUNT/96 완료**. 본문에서 비교하는 모든 결과는 **20,000 optimizer updates를 완료한 checkpoint**이며, 양쪽에 존재하는 동일 설정·seed끼리만 비교한다. 현재 paired comparison은 seed 0 중심이다. 아직 4-seed 평균이나 최종 승자를 주장하지 않는다.

Histogram TV는 actor histogram과 정답의 bin별 확률 차이를 합한 뒤 2로 나눈 값으로, 작을수록 정확하다. 예를 들어 두 bin의 확률이 (0.6,0.4)와 (0.5,0.5)이면 TV=0.1이다.

## 먼저 볼 결과

1. **Batch 32는 이번 toy의 분포 복구를 크게 바꿨다.** 기존 Direct GMM은 seed 0의 128×128에서 histogram TV가 0.3734 → 0.0506, 256×256에서는 0.3739 → 0.0450으로 개선됐다. 넓은 하나의 분포에 머물던 actor가 세 peak를 복구했다.
2. **작은 그룹에서는 mode-gradient 선택이 여전히 유리하다.** Batch 32의 16×16에서는 기존 Direct GMM TV 0.3737, mode 선택 0.0846; 64×64에서는 각각 0.3736, 0.0576이다. Batch를 늘리는 것만으로 모든 작은-N 문제를 해결하지는 못했다.
3. **Confidence 곱의 추가 이점은 일관되지 않다.** Batch 32에서 16×16, 64×64는 mode 선택만 한 쪽이 더 좋았고, 128×128·256×256에서는 차이가 작다. Seed 0만으로 근소한 순위를 일반화하지 않는다.
4. **잘 복구한 actor에도 서로 반대 방향인 mode gradient가 남아 있다.** 그러므로 이번 개선을 “gradient conflict를 제거했기 때문”이라고 결론낼 수 없다. 표본 평균화·latent 분화·gradient 크기·학습 경로를 함께 봐야 한다.

TV와 heatmap의 의미는 아래에서 먼저 정의한다. 이후 그림은 그 정의로 읽으면 된다.

## 0. 실험 설정과 그림 읽는 법

| 항목 | 설정 |
|---|---|
| 환경 | 상태 하나, 고정된 1D Q; critic 및 환경 상호작용 없음 |
| Action | [−1,1] |
| 목표 mode | 중심 −0.6, 0, 0.6; 폭 0.1; 동일 nominal mixture mass |
| Temperature | 0.25 |
| Actor | 원래 toy의 squashed conditional Gaussian; 256×2 GELU |
| Latent | 1D standard Gaussian; 매 update 독립 재표집 |
| Sigma | 초기 0.5, log sigma 범위 [−5,1] |
| Optimizer | Adam, learning rate 3×10⁻⁴ |
| Proposal | 현재 actor의 conditional mixture; teacher sigma floor 0.05 |
| Batch | 1 또는 32개의 독립적인 latent/candidate 그룹 |
| 그룹당 N×M | 16×16 / 64×64 / 128×128 / 256×256 / 1024×1024 / 2048×2048 / 64×4096 / 2048×4096 |
| Seeds | 0,1,2,3; 실행 계획은 batch별 96 runs |
| 학습 길이 | 20,000 optimizer updates |
| 평가 | 새로운 latent와 conditional noise로 action 32,768개; 256-bin histogram |
| 그림 | Target만 해석적 density, actor는 실제 sample histogram; KDE/smoothing 없음 |

$$f(a)=\frac{1}{3}\sum_{c\in\{-0.6,0,0.6\}}\mathcal{N}(a;c,0.1^2),\qquad Q(a)=0.25\log f(a).$$

정답 Boltzmann 분포는 f를 [−1,1]에서 정규화한 것이다. Actor는 아래와 같다.

$$a=\tanh(\mu_\theta(z)+\sigma_\theta(z)\epsilon),\qquad z,\epsilon\sim\mathcal{N}(0,1).$$

N은 독립적으로 학습되는 N개의 고정 Gaussian parameter가 아니다. **한 neural network에 매번 새 latent N개를 넣어 얻는 conditional Gaussian 수**다. 그림의 고정 evaluation latent 2,048개는 학습 경로를 관찰하기 위한 probe다.

### Batch 32의 정확한 뜻

같은 update 직전 actor에서 32개의 그룹을 독립적으로 뽑는다. 각 그룹마다 N개 student, M개 candidate, 별도의 importance weight와 responsibility가 있다. 32개의 mixture를 하나의 32N-component mixture로 합치지 않는다.

$$g_B(\theta)=\frac{1}{32}\sum_{b=1}^{32}g_b(\theta).$$

이 gradient 평균으로 **Adam을 한 번** 갱신한다. 32번 연속 갱신하는 방식이 아니다. 메모리를 위해 계산을 나누어도 모든 그룹은 동일한 update 직전 parameter를 사용하며, 수식 검증에서 평균 gradient와 한 번의 Adam step이 일치함을 확인했다.

같은 20K updates라도 batch 32는 그룹·candidate 소비량이 32배다. 따라서 이 보고서는 같은 update budget 비교이며, **같은 표본 예산이나 같은 GPU 시간의 비교는 아니다.** 초기 actor는 seed별로 동일하지만, batch가 달라 RNG 진행 경로는 다르다.

### 세 방법과 H heatmap

Teacher weight와 student responsibility는 다음과 같다. 이 실험에서는 OT solver를 사용하지 않는다.

$$w_j=\mathrm{softmax}_j(Q(b_j)/0.25-\log q_F(b_j)),\qquad \gamma_{ij}=\frac{k_i(b_j)}{\sum_l k_l(b_j)}.$$

각 candidate에 대한 effective joint mass는 J이며, mode별 비율 H를 component별로 정규화한다.

$$J_{ij}=w_j\gamma_{ij},\qquad H_{mi}=\frac{\sum_{j\in\mathcal{B}_m}J_{ij}}{\sum_jJ_{ij}}.$$

Basin 경계는 −0.3, 0.3으로 고정했다. 각 heatmap 열은 한 sampled latent이고, 세 mode의 H를 합하면 1이다. 예를 들어 (0.1,0.8,0.1)이면 그 component가 받은 teacher 질량 중 80%가 가운데 basin에서 왔다는 뜻이다. **Conditional Gaussian 자체가 가운데 basin에 80%의 확률을 둔다는 뜻은 아니다.**

| 방법 | 실제 update |
|---|---|
| Direct GMM | 원래 marginal mixture NLL gradient |
| Mode selection | 각 sampled latent에서 H가 가장 큰 mode의 output gradient만 유지 |
| Selection + confidence | 선택한 output gradient에 max H를 추가로 곱함 |

$$m_i=\arg\max_m H_{mi},\qquad c_i=\max_m H_{mi},\qquad \tilde{g}_i=c_i g_{i,m_i}.$$

Gradient 선택은 component의 mu·log sigma output에서 수행한 다음 공유 network에 역전파한다. Teacher, posterior, mode 선택과 confidence에는 gradient를 흘리지 않는다. Mode selection에서는 c=1이다. 추가 gradient-norm 보정은 하지 않았다. 이 update는 원래 marginal NLL의 정확한 gradient가 아니다.

### 지표

**Histogram TV**는 256개 bin의 actor sample mass와 exact target mass를 비교한다.

$$\mathrm{TV}_{\mathrm{hist}}=\frac{1}{2}\sum_{q=1}^{256}|\hat{p}_q-p_q^\star|.$$

예를 들어 두 bin에서 actor mass가 (0.6,0.4), target이 (0.5,0.5)이면 TV=0.1이다. 작을수록 target에 가깝다. 단지 세 basin의 총질량만 비교하는 basin-mass TV와 다르며, 여기서는 **bin 단위 histogram TV**로 통일한다. Sample 수가 유한하므로 정답 분포에서 샘플링해도 정확히 0은 아니다.

**Specialist fraction**은 고정된 evaluation latent 중 자기 conditional Gaussian 확률의 80% 이상을 한 basin에 두는 latent 비율이다. 예를 들어 2,048개 중 1,024개가 이 조건을 만족하면 50%다. 이 보조 지표는 conditional CDF로 계산하며, density 그림을 그리는 데는 사용하지 않는다. Heatmap H의 argmax와도 다르다.

**Gradient cosine**은 원래 marginal NLL을 L/C/R teacher basin별로 나눈 parameter gradient 사이의 cosine이다. −1은 반대 방향, 0은 직교, 1은 같은 방향이다. Batch 32에서는 32개 그룹의 mode gradient를 각각 평균한 뒤 cosine을 계산한다.

## 1. 실제 분포가 어떻게 바뀌었는가

아래 위쪽은 batch 1, 아래쪽은 batch 32다. 각 열은 동일 N×M, 동일 seed, 동일 20K update다. 각 run은 32,768 action을 실제로 샘플링했고 256-bin histogram을 그렸다. 여러 seed가 있는 경우 동일한 완료 seed의 histogram만 평균하며, 사용 seed를 panel에 표시한다. **현재 주요 비교 panel은 seed 0이다.**

![작은 N×M: batch 1과 32의 실제 histogram](figures/density_small.png)

16×16과 64×64에서는 batch를 늘려도 기존 Direct GMM은 넓은 분포에 머문다. 반면 mode-gradient 선택 방법은 batch 1에서 약했던 **가운데 peak까지 복구**했다. 128×128에서는 기존 Direct GMM도 batch 32에서 세 peak를 복구하여 세 방법의 최종 결과가 가까워졌다.

![더 큰 N×M: batch 1과 32의 실제 histogram](figures/density_larger.png)

256×256과 1024×1024에서도 seed 0의 기존 Direct GMM이 batch 32에서 크게 개선됐다. 64×4096도 마찬가지다. 따라서 작은 N 자체만으로 실패를 설명하기는 어렵고, candidate 수 M과 batch도 함께 관여한다. 다만 N=64, M=64와 M=4096 비교는 proposal/teacher 정확도 등 여러 효과를 함께 바꾸므로 원인을 하나로 확정하지 않는다.

### 같은 seed끼리 비교한 전체 완료 수치

아래는 양쪽 batch 모두 완료된 조합만 포함한다. TV는 낮을수록 좋고, 마지막 열이 음수면 batch 32가 개선된 것이다. 여러 seed가 완료되면 같은 seed 집합의 평균을 비교한다.

PAIRED_TABLE

## 2. Responsibility와 실제 latent 분화는 같은 이야기를 하는가

아래는 **128×128, seed 0, 20K**의 예시다. 설정은 작은-N과 큰-N 사이의 변화를 보여주기 위해 미리 정한 128×128을 사용했다. Batch 32의 H는 32개 그룹 중 **group 0 하나**를 보여준다. 모든 그룹의 H는 원본 진단에 저장되어 있다.

![Responsibility: z 정렬](figures/responsibility_sorted.png)

![동일 responsibility 행렬: 원본 sampling 순서](figures/responsibility_raw.png)

두 그림은 같은 H 행렬을 다른 열 순서로 나타낸 것이며, 동일한 0–1 색 범위를 쓴다. 정렬 기준은 action이 아니라 **latent z**다. 따라서 가로축의 순서를 행동 위치로 읽으면 안 된다. 원본 training_z와 H를 함께 저장했으므로 정렬을 그대로 재현할 수 있다.

Batch 1의 기존 Direct GMM은 latent별 분업이 거의 보이지 않는다. Batch 32에서는 기존 Direct GMM도 L/C/R을 담당하는 구간으로 나뉜다. 다만 batch 1의 mode-gradient 방법도 H상 분업은 보이는데 가운데 peak를 제대로 복구하지는 못했다. **H의 선명한 분화만으로 density fitting의 성공을 판정하면 안 된다.**

![고정 evaluation latent의 mean 및 sigma](figures/latent_specialization.png)

같은 2,048개 evaluation latent에서 본 함수다. 위쪽은 tanh(mu), 아래쪽은 **pre-tanh sigma**이며 action의 실제 평균·표준편차와 같지 않다. Batch 1은 점선, batch 32는 실선이다. 기존 Direct GMM은 batch 1에서 거의 일정한 mean과 넓은 sigma를 내놓지만, batch 32에서는 latent에 따라 다른 mode로 mean이 나뉜다.

128×128 seed 0의 기존 Direct GMM은 specialist fraction이 **0% → 93.9%**, 평균 pre-tanh sigma가 **0.602 → 0.124**로 바뀌었다. 이 변화는 단순히 histogram의 우연한 noise가 아니라 actor 함수 자체가 달라졌음을 보여준다. 다만 mode 선택 방법은 평균 sigma가 batch 32에서 약간 더 커져도 density가 개선됐으므로, **sigma를 작게 만드는 것 자체가 정답은 아니다.**

## 3. Gradient interference를 어떻게 해석해야 하는가

![Mode별 원래 marginal-NLL parameter gradient cosine](figures/gradient_cosine.png)

이 그림은 각 방법의 학습된 actor에서 **원래 marginal NLL**을 mode별로 분해하여 측정한 진단이다. Mode selection 방법의 최종 routed update 그 자체를 mode별로 분해한 그림은 아니다. Batch 1은 한 그룹, batch 32는 32그룹 평균 gradient이며, 서로 다른 panel은 학습된 actor와 teacher도 다르다.

분포를 잘 복구한 batch 32에서도 기존 Direct GMM의 C–R cosine은 약 −0.83이고, mode selection의 L–C cosine은 약 −0.98이다. 즉, **큰 음의 cosine이 곧 mode collapse를 뜻하지 않는다.** 서로 다른 mode의 목적은 최적점 근처에서도 상쇄될 수 있으며, cosine만으로 gradient의 크기나 실제 update 영향을 판단할 수도 없다.

지금 결과가 보여주는 것은 batch와 gradient routing이 실제 분포 복구를 바꾼다는 사실이다. 아직 “어떤 특정 mode 간 충돌이 실패의 직접 원인이었다”거나 “batch averaging이 충돌을 제거했다”는 인과 결론까지 뒷받침하지는 않는다. 같은 actor·같은 teacher에서 한 mode update만 적용한 counterfactual과 실제 학습 경로를 함께 분석해야 한다. 관련 counterfactual 원본은 저장되어 있다.

## 4. 수렴 경로와 시간

![동일 완료 seed의 update 수 및 기록된 training 시간에 따른 TV](figures/tracking_steps_time.png)

작은 설정의 mode-gradient 방법은 초기 두 peak를 형성한 뒤 긴 구간을 지나 가운데 peak를 추가로 복구하는 경로를 보인다. 예를 들어 16×16 batch 32는 20K 근처에서 오차가 크게 내려간다. 따라서 마지막 시점 하나만 보고 “초기부터 항상 빠르다”고 말하지 않는다.

오른쪽 시간은 코드가 기록한 **training block 누적 시간**이며 JIT compilation을 포함한다. 진단·평가·checkpoint I/O·GPU 배정 대기는 제외된다. 전체 run 경과시간은 이보다 길다. GPU 모델과 동시 사용 부하가 run별로 다를 수 있어, 이를 동일 GPU의 정밀 속도 benchmark로 해석하지 않는다. 원본 표에는 diagnostic 시간과 실행 노드도 별도로 보관했다.

동일한 update 수에서는 batch 32가 유리한 사례가 많지만, 계산량·표본 예산까지 같게 맞춘 우위는 아직 실험하지 않았다.

## 5. 현재 가져갈 수 있는 해석

**첫째, 이번에 관찰한 분포 복구 실패를 actor의 표현력 부족만으로 설명하기 어렵다.** 같은 구조·같은 초기 seed에서도 batch를 늘리자 기존 Direct GMM이 세 mode를 복구했다.

**둘째, 그룹 내부의 N·M과 그룹 평균 개수 B를 분리해서 봐야 한다.** B=32는 표본 변동을 평균하지만, 각 그룹이 사용하는 N-component mixture 및 M-candidate teacher는 그대로다. B를 늘린다고 한 그룹의 mixture가 32N개 component로 바뀌지는 않는다. 표본 변동 감소가 개선에 기여했을 가능성은 있으나, 이번 실험은 gradient variance 감소량을 직접 추정한 실험은 아니다.

**셋째, mode-gradient 선택은 작은 그룹에서 유용한 개입으로 보인다.** 하지만 알려진 basin label을 사용하는 toy 개입이며 일반 RL에서 mode를 찾아내는 문제는 해결하지 않았다. 큰 설정에서는 기존 Direct GMM도 비슷한 수준까지 도달하므로 모든 크기에서 선택 방식이 필요하다는 결론은 아니다.

**넷째, confidence는 별도 검토가 필요하다.** 현재 seed 0에서는 confidence를 곱하지 않은 쪽이 더 좋은 작은 설정도 있다. Confidence는 assignment의 명확도뿐 아니라 gradient 크기와 component별 상대 가중치도 바꾸기 때문에, 추가 이점을 단순하게 가정해서는 안 된다.

전체 4 seeds가 끝나면 이 초기 패턴이 반복되는지와 수렴 실패 빈도를 최종 비교해야 한다. 이번 snapshot에 포함되지 않은 run을 실패 또는 미복구로 집계하지 않았다.

## 6. Batch 1 전체 결과와 원본 수치

Batch 1은 8크기×3방법×4seeds가 모두 완료됐다. 아래에는 불리한 seed를 제외하지 않고 각 seed의 histogram을 가는 선으로 모두 표시했다. **큰 N에서 기존 Direct GMM이 복구하는 사례도 있으므로, 앞서 보여준 작은 설정의 실패를 전체 실험의 일반적 결론으로 확대하면 안 된다.**

![Batch 1: 전체 96 runs의 최종 histogram](figures/batch1_all_seeds.png)

전체 완료 run별 수치는 [전체 수치표](all_runs_table.md), 분석에 사용한 구조화 데이터는 [summary.json](summary.json)에 있다. 본문 paired table 밖의 완료 run도 이 파일에는 모두 포함한다.

## 7. 재현 및 보관 위치

- Batch 1 source commit: `COMMIT1`
- Batch 32 source commit: `COMMIT32`
- 분석용 표본 RNG: latent probe `77000+seed`, action draw `88000+seed`; action 평가는 latent·conditional noise를 각각 새로 생성한다.
- 수치 검증: 32개 독립 gradient의 직접 평균과 구현의 일치, microbatch 1/4/32 일치, optimizer 1회 갱신, checkpoint+Adam+RNG 재개, H 정규화와 mode-gradient 합 일치를 통과했다.
- 보고서 작성 시 각 완료 run의 32,768 samples에서 histogram과 TV를 다시 계산했고, 저장된 값과의 일치를 확인했다. Exact target bin mass의 합도 검증했다.
- [실험 계획](PROTOCOL.md), [분석·그림 생성 코드](build_report.py), [보고서 조립 코드](finalize_report.py), [원본 파일 SHA256](ARTIFACT_MANIFEST.json).

중앙 보고서 폴더:

`dildata:/data1/heejoonorm/OptiQ/reports/20260921_gmm_mode_gradient_batch32/`

원본 batch 32 checkpoint·평가 sample·진단:

`dildata:/data1/heejoonorm/OptiQ/studies/20260921_gmm_mode_gradient_batch32/campaign/runtime/runs/`

원본 batch 1:

`dildata:/data1/heejoonorm/OptiQ/studies/20260921_gmm_mode_gradient/campaign/runtime/runs/`

대표 파일은 각 run의 `checkpoint.msgpack`, `eval_020000.npz`, `diagnostic_020000.npz`다. 각 파일의 hash와 정확한 경로를 ARTIFACT_MANIFEST에 기록했다. HTML은 그림과 수식 이미지를 모두 내부에 넣어 인터넷 없이 열 수 있다. Markdown은 수식 원문을 유지한다.
'''
text=template.replace('STATUS',status).replace('STAMP',s['snapshot']).replace('COUNT',str(s['completed']['32'])).replace('PAIRED_TABLE',(O/'paired_table.md').read_text().strip()).replace('COMMIT1',s['commits']['1']).replace('COMMIT32',s['commits']['32'])
(O/'report.md').write_text(text)
# Small offline renderer for this report: mathtext formulas and plots are data URIs.
def uri(path):return 'data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()
def inline(t):
 t=html.escape(t);t=re.sub(r'`([^`]+)`',r'<code>\1</code>',t);t=re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',t)
 t=re.sub(r'\[([^]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',t);return t
parts=[];lines=text.splitlines();i=0
while i<len(lines):
 line=lines[i].strip()
 if not line:i+=1;continue
 if line.startswith('$$'):
  expr=line.strip('$');fig=plt.figure(figsize=(.1,.1));fig.text(0,0,'$'+expr+'$',fontsize=16);buf=io.BytesIO();fig.savefig(buf,format='png',dpi=170,bbox_inches='tight',pad_inches=.12,transparent=True);plt.close(fig)
  parts.append('<div class="equation"><img alt="'+html.escape(expr,quote=True)+'" src="data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()+'"></div>');i+=1;continue
 if line.startswith('!['):
  q=re.match(r'!\[([^]]*)\]\(([^)]+)\)',line);parts.append('<figure><img alt="'+html.escape(q[1])+'" src="'+uri(O/q[2])+'"><figcaption>'+html.escape(q[1])+'</figcaption></figure>');i+=1;continue
 if line.startswith('#'):
  n=len(line)-len(line.lstrip('#'));parts.append(f'<h{n}>'+inline(line[n:].strip())+f'</h{n}>');i+=1;continue
 if line.startswith('|'):
  table=[]
  while i<len(lines) and lines[i].strip().startswith('|'):table.append([v.strip() for v in lines[i].strip().strip('|').split('|')]);i+=1
  parts.append('<div class="table"><table><thead><tr>'+''.join('<th>'+inline(c)+'</th>' for c in table[0])+'</tr></thead><tbody>')
  for row in table[2:]:parts.append('<tr>'+''.join('<td>'+inline(c)+'</td>' for c in row)+'</tr>')
  parts.append('</tbody></table></div>');continue
 if line.startswith('- ') or re.match(r'^\d+\. ',line):
  ordered=bool(re.match(r'^\d+\. ',line));tag='ol' if ordered else 'ul';parts.append('<'+tag+'>')
  while i<len(lines) and ((lines[i].strip().startswith('- ')) if not ordered else bool(re.match(r'^\d+\. ',lines[i].strip()))):
   val=re.sub(r'^(?:- |\d+\. )','',lines[i].strip());parts.append('<li>'+inline(val)+'</li>');i+=1
  parts.append('</'+tag+'>');continue
 parts.append('<p>'+inline(line)+'</p>');i+=1
style='body{max-width:1440px;margin:36px auto;padding:0 28px;font:16px/1.75 system-ui;color:#223049;background:#fff}h1{font-size:32px}h2{margin-top:46px;border-top:1px solid #dfe5eb;padding-top:22px}h3{margin-top:28px}figure{margin:24px 0}figure img{width:100%;height:auto}figcaption{font-size:13px;color:#657083}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px 10px;border-bottom:1px solid #e3e9ef;text-align:left}th{background:#eff4f8}.table{overflow:auto}code{background:#f2f5f8;padding:2px 5px;overflow-wrap:anywhere}a{color:#176f9d}.equation{overflow:auto;text-align:center;padding:14px}.equation img{max-width:100%;height:auto}li{margin:7px 0}'
(O/'report.html').write_text('<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Mode-gradient batch 32: completed-run comparison</title><style>'+style+'</style></head><body>'+''.join(parts)+'</body></html>')
print('Rendered',status,s['snapshot'],s['completed'])
