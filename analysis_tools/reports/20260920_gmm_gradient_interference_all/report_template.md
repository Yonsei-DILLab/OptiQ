# Direct GMM: mode별 gradient 간섭과 latent 분업

**최종 통합 보고서 · 32/32 run 완료 · 8개 N×M · seed 0–3 · 모든 결과 20,000 updates**

이번 실험은 **“큰 N에서 latent가 mode별로 나뉘는가, 나뉜 뒤에도 한 mode의 update가 다른 담당 latent를 움직이는가?”**를 확인한다. 관찰 결과, N=2048에서는 세 mode를 담당하는 latent들이 분화했다. 그러나 분화 이후에도 공유 network를 통한 상호 영향은 남았다. 작은 크기에서 실패한 run들은 다른 양상이다. 대부분 latent의 출력이 비슷해져, 전체가 하나의 넓은 conditional Gaussian처럼 동작했다.

Gradient orthogonalization은 아직 적용하지 않았다. 아래 결과는 **수정하지 않은 Direct GMM의 학습과, 저장된 actor를 복사해 수행한 한 번의 update 진단**이다.

## 0. 실험 설정과 그림 읽는 법

### 환경과 목표 분포

State는 항상 0이고 action은 −1에서 1 사이다. Critic을 학습하거나 Q를 변경하지 않는다. 고정된 세 봉우리의 Q를 사용한다.

$$
f(a)=\frac{1}{3}\sum_{c\in\{-0.6,0,0.6\}}\mathcal{N}(a;c,0.1^2)
$$

$$
Q(a)=0.25\log f(a),\qquad \pi^*(a)=\frac{f(a)}{\int_{-1}^{1}f(x)\,dx}
$$

Temperature는 0.25다. 따라서 정답은 위 세 Gaussian mixture를 action 범위에서 정규화한 분포다. 봉우리 사이 거리는 0.6, 각 봉우리의 폭은 0.1이다. 보상·환경 feedback으로 붕괴가 생기는 실험은 아니다.

### Actor와 teacher

$$
a=\tanh\!\left(\mu_\theta(z)+\sigma_\theta(z)\epsilon\right),\qquad z,\epsilon\sim\mathcal{N}(0,1)
$$

**N은 매 update 새로 뽑는 latent의 수이며, 고정된 N개의 독립 network나 영구적인 component ID가 아니다.** 모든 latent가 같은 actor parameter를 공유한다. 매번 이 N개의 conditional Gaussian mixture에서 총 M개의 후보를 IID로 뽑는다. N=M이라고 해서 각 latent에서 정확히 한 후보를 뽑는 방식은 아니다.

후보를 뽑은 mixture의 density를 q_F라 하면 teacher weight는 다음과 같다.

$$
w_j=\operatorname{softmax}_j\!\left(Q(b_j)/0.25-\log q_F(b_j)\right)
$$

$$
q_N(b)=\frac{1}{N}\sum_{i=1}^N k_\theta(b\mid z_i),\qquad L=-\sum_{j=1}^M w_j\log q_N(b_j)
$$

Teacher 후보·weight·latent는 gradient를 끊는다. 원본 학습은 pre-tanh NLL로 구현되어 있고, 보고서의 공통 likelihood 진단은 parameter와 무관한 tanh Jacobian을 포함한다. 두 표현의 actor gradient는 같다. **OT와 Sinkhorn은 사용하지 않는다.**

| 항목 | 설정 |
|---|---|
| 원본 경로 | 기존 0917 non-stationary toy의 3-mode 초기 구간과 v5 Direct GMM update |
| N×M | 16×16, 64×64, 128×128, 256×256, 1024×1024, 2048×2048, 64×4096, 2048×4096 |
| Seeds / 학습 길이 | 각 0,1,2,3 / random initialization부터 20K updates |
| Actor | 256×2 GELU, state 1D·latent 1D, state batch 1 |
| Variance | Learned sigma, 초기 0.5, log sigma 범위 [−5,1] |
| Teacher proposal | Conditional squashed Gaussian mixture, proposal에만 sigma floor 0.05 |
| Optimizer | Adam 3×10⁻⁴, 원본 update, 추가 clipping·EMA·gradient projection 없음 |
| Precision | 원본 JAX float32/default matmul |
| Density 평가 | 새 latent·noise로 생성한 **32,768개 action의 256-bin histogram** |
| 진단용 latent | Seed별 고정 2,048개, training RNG와 분리 |
| 상세 진단 시점 | 0,1,10,100,500,1K,2K,5K,10K,15K,20K — 총 352 snapshots |

Actor density에는 KDE smoothing이나 별도의 density 적분을 쓰지 않는다. 검은 점선은 해석적으로 계산한 exact target density다. 평가용 Gaussian CDF는 아래 latent 역할과 target bin mass 계산에만 사용한다. 같은 seed의 평가 RNG는 크기와 step 사이에 동일하게 유지하며, 학습 RNG는 진단 때문에 바뀌지 않는다.

### 지표 정의

- **Histogram TV**: 각 action bin의 actor 확률과 정답 확률 차이의 절댓값 합에 1/2을 곱한 값. 0이면 bin 단위 분포가 같다. 봉우리 내부의 모양 차이까지 반영한다.
- **Basin-mass TV**: action을 L=[−1,−0.3), C=[−0.3,0.3), R=[0.3,1]의 세 영역으로 나눈 뒤 같은 TV를 계산한다. 세 영역의 질량만 비슷하면 작아질 수 있어, mode 내부 복구의 근거로 단독 사용하지 않는다.
- **Specialist latent**: 한 conditional Gaussian이 특정 basin에 확률 0.8 이상을 두는 경우 그 basin의 담당 latent로 분류한다. 나머지는 ambiguous다. Mean만 가장 가까운 mode에 강제 배정하지 않는다.
- **Teacher ESS**: 1/Σw². ESS/M은 후보 weight가 얼마나 균등한지 보여준다.
- **Component ESS**: GMM responsibility로 component별 할당 질량 α를 구해 1/Σα²을 계산한다. ESS/N이 1에 가까워도 latent들이 서로 다른 mode를 담당한다는 뜻은 아니다.

표의 ±는 **4 seed의 표본 표준편차**이며 confidence interval이 아니다. `TV<0.1`은 결과를 읽기 위해 적용한 사후 요약 기준이고, 사전에 정한 성공 판정이나 통계적 검정은 아니다. 이 보고서는 속도 우열을 주장하지 않는다. 자동 timing 필드는 진단·컴파일·프로세스 시작을 포함하는 범위가 달라 별도 timing 실험 없이 비교하지 않았다.

## 1. 학습이 끝난 뒤 실제로 어떤 분포가 나왔는가?

아래 그림은 8개 크기의 **모든 seed**를 같은 action·density 축에 겹친 것이다. 성공한 seed와 실패한 seed가 섞인 설정에서 평균 density 하나만 그리면 차이를 감출 수 있으므로 각각 그렸다.

![8개 크기와 4 seed의 최종 action histogram](figures/01_all_histograms.png)

{{RESULT_TABLE}}

- **16×16,64×64,128×128,256×256:** 모두 TV가 약 0.366이다. 세 봉우리의 질량이 들어갈 영역을 넓게 덮지만 봉우리 모양은 복구하지 못한다.
- **1024×1024:** seed 3만 TV 0.0517로 복구했다. 다른 세 seed는 작은 크기와 비슷한 넓은 분포에 남았다.
- **64×4096:** seed 2만 TV 0.0691로 복구했다. M만 크게 늘리는 것으로 모든 seed의 문제를 해결하지 못했다.
- **2048×2048 및 2048×4096:** 각 4 seed 모두 TV<0.1이고, 약 91.8%의 고정 latent가 한 basin의 specialist가 됐다.

**이 실험에서 N=2048은 잘 작동했다. N=2048이면 항상 충분하다거나 1024와 2048 사이에 일반적인 임계값이 있다는 뜻은 아니다.** 네 seed와 현재 initialization·Q·학습 길이에서 관찰한 차이다.

![크기별 seed별 학습 곡선](figures/03_learning_curves.png)

2048×2048은 최초 TV<0.1이 8K–9K, 2048×4096은 8K–10K에서 관찰됐다. 반면 64×4096의 성공 seed는 19K에서 처음 이 기준에 도달했다. 최초 도달 시점은 이후 계속 유지된다는 뜻이 아니며, 위 곡선과 최종 결과를 함께 봐야 한다.

N=M의 여섯 설정은 N과 M이 함께 변한다. 분리해서 볼 수 있는 비교는 **64×64 ↔ 64×4096**, **2048×2048 ↔ 2048×4096**처럼 N이 같은 쌍이다. 후자의 평균 TV 차이는 0.0615 대 0.0541로 작고 seed 변동도 있으므로, M=4096이 유의하게 낫다고 결론내리지 않는다.

## 2. 큰 N에서는 정말 담당 latent가 나뉘는가?

**N=2048에서는 그렇다.** 다음 두 그림에서 x축은 매번 동일하게 유지한 평가 latent z다. 하나의 선은 한 seed의 actor가 z에 따라 내는 대표 위치 또는 sigma다. Tanh(mu)는 conditional의 대표 위치이며 squashed action의 정확한 평균은 아니다.

![모든 크기·seed의 고정 latent 대표 위치](figures/04_latent_means.png)

![모든 크기·seed의 고정 latent sigma](figures/05_latent_sigmas.png)

잘 안 된 run에서는 서로 다른 z가 거의 같은 mean을 내고 sigma는 약 0.6이다. **큰 Gaussian 하나로 덮는 것과 유사한 출력**이 된다. 잘 된 run에서는 z 영역별로 mean이 세 mode 쪽으로 나뉘고 평균 sigma는 약 0.09까지 작아졌다.

아래 heatmap의 각 seed마다 L/C/R 세 줄은 해당 fixed latent의 conditional이 각 basin에 놓는 확률이다. 하나의 latent 열에서 특정 줄만 밝다면 해당 mode를 주로 담당한다. 세 줄이 동시에 비슷한 밝기라면 여러 basin을 넓게 덮는 conditional이다.

![32개 run의 fixed latent 담당 확률](figures/06_latent_roles_all_seeds.png)

![학습 중 specialist 비율이 형성되는 과정](figures/07_role_emergence.png)

분업이 나타나기 전부터 있던 담당 latent가 사라졌다고 해석하면 안 된다. 이 초기화 실험의 실패 run은 상당 부분 **분업 자체가 형성되지 않은 채 비슷한 출력으로 남는 현상**이다. 모든 mode가 이미 학습된 공통 actor에서 출발해 붕괴만 측정한 실험은 아니다.

같은 64개 fixed latent를 z 분위수 기준으로 선택해 시간에 따라 추적했다. 1024×1024의 성공·실패 차이와 2048×2048의 네 seed를 모두 보여준다. 현재 학습 batch의 row 번호를 영구적인 latent ID로 사용하지 않는다.

![동일 latent의 학습 중 대표 위치 궤적](figures/17_fixed_latent_trajectories.png)

## 3. Teacher가 나빠서 못 배우는가? 일부 component만 살아남는가?

![Final teacher와 actor의 basin 질량](figures/08_teacher_actor_basin_mass.png)

{{TEACHER_TABLE}}

Teacher basin TV·ESS·component ESS는 **20K에서 다음 update가 사용할 teacher 한 번씩, 총 4 seed**의 값이다. Mode 누락 횟수는 전체 11진단×4seeds에서 한 basin이라도 원시 후보가 0개인 snapshot 수다. 따라서 서로 다른 집계 범위임을 구분해야 한다. 작은 M의 teacher는 한 번의 표본에 따른 변동이 크다.

**64×4096은 teacher의 mode 질량이 상당히 정확해도 세 seed가 실패했다.** 최종 teacher basin TV 평균은 0.0073인데 actor histogram TV 평균은 0.2927이다. 이것은 최종 teacher에서 mode가 통째로 누락되는 것만으로 실패를 설명하기 어렵다는 근거다. 다만 이 coarse mass 진단만으로 teacher의 전체 density나 학습 중 모든 step이 정확했다고 주장하지 않는다.

작은 크기의 실패 run에서 **component ESS/N은 거의 1**이다. 이는 일부 component가 모든 responsibility를 독점한 `dead-component` 설명과 맞지 않는다. 모든 component가 거의 같은 Gaussian이 되어 각 후보를 비슷하게 나눠 가지면, 사용량은 균등하면서 mode 분업은 없을 수 있다.

GMM에는 OT plan이 없다. 아래 heatmap은 posterior responsibility를 이용해 그린다.

$$
\gamma_{ij}=\frac{k_i(b_j)}{\sum_\ell k_\ell(b_j)},\qquad P^{\mathrm{GMM}}_{ij}=w_j\gamma_{ij},\qquad \alpha_i=\sum_jP^{\mathrm{GMM}}_{ij}
$$

각 training latent i에 대해 그 할당 질량 중 L/C/R basin이 차지하는 비율을 표시했다. 즉 row별로 Σ_{j∈basin}Pᵢⱼ / αᵢ를 계산하고 latent z 순으로 정렬했다. 전체 N×M OT 행렬이나 학습에 추가한 constraint가 아니다.

![GMM responsibility의 mode별 분업](figures/16_gmm_assignment.png)

**균등한 row supervision만으로 mode 분업이 보장되지는 않는다.** 이번 실패 결과에서는 이미 사용량이 거의 균등했기 때문이다. 이것은 OT의 geometry-aware assignment가 효과 없다는 검증도 아니다. 이번 비교에는 OT 학습 조건이 없어서 그 효과는 판정할 수 없다.

## 4. Mode별 gradient가 실제로 충돌하는가?

동일한 teacher를 세 basin으로 나누어 다음 loss를 정의한다. 각 basin 내부에서 weight를 다시 정규화하지 않고 원래 w를 유지한다.

$$
L_b=-\sum_{j:b_j\in\mathcal{B}_b}w_j\log q_N(b_j),\qquad g_b=\nabla_\theta L_b,\qquad \nabla_\theta L=\sum_b g_b
$$

$$
\operatorname{cos}(g_b,g_c)=\frac{g_b^\top g_c}{\|g_b\|\,\|g_c\|}
$$

음의 cosine은 −g_b 방향의 아주 작은 **plain SGD** step이 L_c를 1차 근사에서 증가시킨다는 뜻이다. Adam은 momentum과 coordinate별 preconditioning을 쓰므로 cosine만으로 실제 update 결과를 예측할 수 없다.

![Final mode별 full-parameter gradient cosine](figures/09_gradient_cosine.png)

작은 N의 넓은 분포에서는 mode별 gradient 충돌이 강하다. 하지만 성공한 큰 N에서도 음의 쌍이 남는다. **충돌이 존재하는 것과 학습에 실패하는 것은 같은 명제가 아니다.** 이 그림은 각 seed의 cosine을 계산한 뒤 평균한 것이며, 먼저 gradient를 평균한 뒤 cosine을 계산하지 않았다.

아래는 같은 비교를 공유 trunk, mean head, sigma head로 나눈 것이다. 해당 block의 norm이 0이면 cosine은 정의되지 않아 N/A로 표시한다.

![공유 trunk와 mean·sigma head의 gradient](figures/10_gradient_blocks.png)

## 5. 한 mode만 업데이트하면 다른 mode가 얼마나 영향을 받는가?

각 저장 시점의 **같은 actor·Adam state**를 복사하고, 다음 teacher·training latent를 고정한 채 여덟 branch를 만든다.

| Branch | 적용한 한 번의 update | 역할 |
|---|---|---|
| Full Adam | 원본 Direct GMM loss 전체 | 실제 알고리즘이 다음에 할 update의 대조 |
| Momentum only | 새 gradient=0, Adam state 유지 | 과거 gradient history의 영향 |
| L/C/R-only Adam | 한 basin loss만 입력 | 한 mode objective가 다른 mode에 미치는 영향 |
| L/C/R-only SGD | 해당 gradient 방향, parameter 이동 norm을 Full Adam과 일치 | Adam과 구분한 방향성 대조 |

모든 branch는 측정 후 폐기한다. 원본 training parameter, optimizer, RNG는 바뀌지 않는다. 따라서 **gradient 조작 알고리즘을 학습해 비교한 결과가 아니다.** 아래는 20K checkpoint에서 생성한 다음 한 step의 진단이다.

Likelihood는 두 가지로 평가한다.

1. 같은 teacher 후보와 training latent: 현재 유한 표본 objective의 변화.
2. 별도 target reference와 fixed 2048 evaluation latent: 새 latent로 평가한 목표 분포 fitting 변화. Target reference는 1024개 bin의 exact CDF 질량을 midpoint에 둔 quadrature이며, likelihood의 완전한 해석적 적분은 아니다.

![Mode-only Adam의 다른 mode NLL 변화](figures/11_mode_adam_reference.png)

![Mode-only norm-matched SGD의 다른 mode NLL 변화](figures/12_mode_sgd_reference.png)

행은 update에 사용한 mode, 열은 평가한 mode다. **양수는 악화**, 음수는 개선이다. 값과 colorbar는 NLL 변화에 1,000을 곱해 표시했다. 네 seed의 평균이므로 표의 비율 및 개별 결과와 함께 읽어야 한다.

{{GRADIENT_TABLE}}

음의 쌍 비율은 20K의 3쌍×4seeds, 다른 mode 악화 비율은 6개 off-diagonal×4seeds에서 계산했다. 악화는 ΔNLL>10⁻⁶으로 집계한 서술 통계다. 24개 항목을 독립 seed처럼 취급한 유의성 검정은 하지 않았다.

**큰 N에서도 다른 mode에 영향을 주는 현상은 있다.** 2048×2048에서 mode-only Adam의 다른 mode NLL은 83.3%에서 증가했고, norm-matched SGD에서는 79.2%였다. 2048×4096에서도 각각 62.5%,75.0%다. 그러나 이런 단독 update는 원래 다른 두 mode의 loss를 빼고 수행하므로, 이 비율 자체가 full objective 학습의 비효율을 증명하지는 않는다.

### 이미 담당 mode가 나뉜 latent도 함께 움직이는가?

Grouping은 update 전의 specialist 판정을 유지한다. Update 뒤 새 담당 mode로 재분류하지 않는다. 빈칸은 담당 latent가 없어서 측정할 수 없는 경우다. 이를 “간섭 0”으로 읽으면 안 된다.

![한 mode update가 기존 담당 latent들의 대표 위치를 움직인 크기](figures/14_cross_mode_movement.png)

![기존 담당 latent의 자기 basin 확률 변화](figures/15_cross_mode_mass.png)

2048×2048과 2048×4096에서 **다른 mode 담당 latent의 대표 action RMS 이동은 평균 약 0.002**였다. 해당 mode 담당 latent 이동과의 비율은 seed별 계산 후 평균하면 각각 약 0.91,0.90이다. 즉 공유 network를 통해 다른 담당 latent도 같은 order로 움직인다. 비율만 보면 커 보이지만, 절대 이동은 mode 중심 간 거리 0.6보다 훨씬 작다. 이 한 step이 mode collapse를 만들었다는 관찰은 아니다.

자기 basin 확률 변화는 부호가 섞여 있고 대체로 작은 규모다. **다른 latent가 움직였다는 것과 mode coverage가 무너졌다는 것은 구별해야 한다.**

### Full update와 Adam history를 보면 해석이 달라지는가?

![Full Adam과 momentum-only의 전체 likelihood 변화](figures/13_full_vs_momentum.png)

원은 full Adam, x는 새 gradient를 0으로 준 momentum-only다. 색은 seed다. 왼쪽은 같은 teacher·training latent, 오른쪽은 별도 target·evaluation latent 평가다. 작은 변화와 큰 변화를 함께 보기 위해 y축은 0 근처가 선형인 symlog다.

예를 들어 **64×4096 seed 2**의 20K 진단에서 같은 teacher의 전체 NLL은 **0.00766 감소**하지만, 별도 reference NLL은 **0.00346 증가**했다. 이 차이는 현재 유한 표본 objective에서의 개선과 전체 target fitting의 개선이 일치하지 않을 수 있음을 보여준다. 다른 후보·latent 평가와 optimizer history가 함께 작용하므로 이를 gradient 간섭 하나의 영향으로 분리해서 해석할 수는 없다.

Momentum-only에서도 loss와 출력이 변한다. Mode-only Adam에서 보이는 변화를 모두 이번 mode gradient에 귀속해서는 안 된다. 두 결과의 단순 차감도 Adam의 비선형 preconditioning 때문에 엄밀한 순수 gradient 효과가 되지는 않는다.

## 6. 질문에 대한 답과 다음 분석 방향

**질문 1: N이 크면 여러 latent가 mode별로 나뉘는가?**

이 설정의 N=2048에서는 네 seed 모두 분화했다. N=1024에서도 가능한 seed는 있었지만 세 seed는 분화하지 못했다. 따라서 충분한 표현 가능성과 실제 optimization에서 분업이 형성되는지는 별개의 문제다.

**질문 2: 분업이 생기면 mode별 update가 서로 독립적인가?**

아니다. 한 mode만으로 업데이트할 때 다른 담당 latent들의 mean·sigma·basin probability도 바뀐다. Shared parameter를 통해 출력 변화가 전달되는 현상을 직접 측정했다. 큰 N이 이런 상호작용을 자동으로 제거하지는 않는다.

**질문 3: Gradient orthogonalization을 하면 성능이 좋아질까?**

이번 실험은 이를 아직 답하지 않는다. 후속 intervention을 한다면 두 상황을 나누는 편이 좋다.

- **분업 전의 넓은 출력:** 서로 다른 latent가 역할을 얻는지부터 확인해야 한다. 거의 동일한 conditional들의 mean gradient만 직교화해서는 분업이 자동으로 생긴다고 보장할 수 없다.
- **분업이 이미 생긴 출력:** 동일 teacher·optimizer에서 conflict projection을 적용한 branch와 원본 full update를 비교하여, 다른 담당 latent의 fitting을 보호하면서 전체 target NLL·density를 개선하는지 확인할 수 있다.

이번 결과가 직접 지지하는 문장은 다음과 같다.

> **큰 N에서는 mode별 latent 분업을 학습할 수 있지만, 공유 parameter 때문에 분업 이후에도 mode별 update의 상호 영향이 남는다. 작은 N의 주요 실패 양상은 일부 component의 사용 중단보다, 여러 latent가 비슷한 넓은 conditional로 수렴하는 것이다.**

이 결과로 “모든 실패의 원인이 gradient 간섭이다”, “orthogonalization이 해결한다”, “OT가 반드시 필요하다”, “RL에서도 같은 현상이 지배적이다”까지 주장하지는 않는다. Q가 고정된 1D 진단이며, critic 오류·탐색·data collection을 포함하지 않는다.

## 7. 전체 seed별 결과

{{RUN_TABLE}}

![32개 run의 개별 histogram](figures/02_individual_histograms.png)

## 8. 검증, 원본 위치와 재현

- 총32개 COMPLETE.json과20K 상태 확인. 352개 상세 진단과 evaluation histogram을 읽었다.
- 저장된32768 samples로 histogram을 재계산해 일치 확인. Target bin mass는 독립적인 Gaussian CDF 계산과 대조했다.
- Teacher weight 합·responsibility 질량 보존·fixed latent identity·branch 이름과 크기를 검사했다. 집계에서는 float32 배열을 float64로 합산해 불필요한 누적 roundoff를 피했다.
- 실행 전 각 크기에서 원본 gradient와 mode gradient 합, 원본 update 경로, 유한차분, teacher detach, checkpoint 재개, 진단 후 actor·Adam·RNG 불변 검증이 통과했다.
- Failed run이나 미완료 run을 평균에 숨겨 넣지 않았다. 이전 source snapshot은 변경하지 않았다.

| 실행 snapshot | 포함 크기 | 학습 source commit |
|---|---|---|
| 1b1c6947d06b | 64×4096,2048×4096 | `1b1c6947d06b11d85df3400d16941ab111e56cef` |
| f3aebce92a3e | 16×16,64×64 | `f3aebce92a3ea84ff6c01fce5b2e6910ee98beee` |
| 32bc71599d06 | 128×128,256×256,1024×1024,2048×2048 | `32bc71599d066601ac9b4b1e413ff476bb4f0f6f` |

분석 source commit: `{{ANALYSIS_COMMIT}}`. 분석 코드는 `Yonsei-DILLab/OptiQ`의 `heejoon`, `analysis_tools/reports/20260920_gmm_gradient_interference_all/`에 있다. 이 SHA는 **분석 코드**이며, 위 학습 코드의 SHA와 구분한다.

원본 서버: `heejoonorm@31.148.50.247:11717`, `/home/heejoonorm/OptiQ/legacy_monge/gradient_interference/{snapshot}/runs/{run}/`.

중앙 보관: `dildata:/data1/heejoonorm/OptiQ/studies/20260919_legacy_monge/remote_3114850247/gradient_interference/{snapshot}/`.

보고서 폴더의 `INPUT_MANIFEST.json`은 분석에 사용한 파일별 SHA256과 검증 기록을, `per_run.json`과 `summary.json`은 표와 한-step 진단의 수치 원본을 담고 있다. HTML은 그림과 수식을 이미지로 내장해 인터넷 없이 열 수 있고, **report.md에는 LaTeX 수식 원문**을 유지했다.
