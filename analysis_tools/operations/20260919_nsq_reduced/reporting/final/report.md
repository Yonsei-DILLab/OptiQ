# Non-stationary Q: 축소 전 완료된 1,823개 비교 실험

**집계 시점: 2026-09-19 20:51 KST.** 완료된 실험만 분석했다. 새 크기256×1024·512×512 결과는 아직 포함하지 않는다. 이 문서는 이전13:40 보고서에 큰 크기의 완료 결과와 수정된 legacy closed-loop 결과를 추가한 보고서다.

**지표 읽는 법:** histogram TV는 첫 action 좌표의 512개 bin에 배분된 확률이 목표와 얼마나 다른지, basin-mass TV는 각 mode 주변 영역에 배분된 총확률이 얼마나 다른지를 나타낸다. 둘 다 0이면 일치하고 1이면 완전히 분리된다. Mode의 질량만 맞고 폭·형태는 틀릴 수 있다.

## 먼저 볼 결과

**큰 크기에서 GMM의 최종 분포는 더 정확했지만, 질량 반전에 대한 즉각적인 반응은 Sinkhorn ε=0.1이 훨씬 빨랐다.** 두-mode 실험의 256×16384·1D에서 최종 histogram TV는 GMM **0.065**, Exact learned-σ **0.165**, Sinkhorn ε=0.1 **0.210**이었다. 반면 변화 뒤 첫 1,000 updates의 평균 basin-mass TV는 각각 **0.194, 0.080, 0.011**이었다. 모두 4 seeds 평균이다.

즉, 이번 완료 결과는 **최종 fitting 정확도와 재적응 속도 사이의 차이**를 보여준다. 256×16384의 2D에서도 GMM 최종 TV **0.064** 대 Sinkhorn **0.185**, 변화 직후 mass 오차 **0.219** 대 **0.011**로 같은 경향이 나타났다. 2D Exact는 미완료라 이 비교에 포함하지 않았다.

16×64의 분리·병합에서는 Exact OT가 유리한 조건이 나타난다. 따라서 OT의 역할은 “언제나 더 정확하다”보다 **어떤 목표 변화에서 질량 배분과 복구 속도를 개선하는가**로 나눠 보는 편이 이 결과에 잘 맞는다.

## 0. 완료 범위와 이번 변경

| N×M | 두-mode 질량 변화 | 세-mode 질량 변화 | 3→6→3 | 공통 learned-Q 재생 | 자기 critic 학습 | 합계 |
|---|---:|---:|---:|---:|---:|---:|
| 16×64 | 304 | 304 | 304 | 304 | 304 | **1,520** |
| 256×16384 | 150 | 0 | 0 | 130 | 16 | **296** |
| 1024×4096 | 0 | 0 | 0 | 0 | 7 | **7** |
| 2048×2048 | 0 | 0 | 0 | 0 | 0 | **0** |
| 전체 | 454 | 304 | 304 | 434 | 327 | **1,823** |

별도로 완료된 초기 20K prefix758개와 공통 critic16개가 있다. 이 준비 실행을 비교 실험 수에 중복해서 세지 않았다. 실행 중이던 제외 조건20개는 checkpoint 저장 후 중단했고, 중간 결과를35K 최종 결과로 취급하지 않는다.

**완료 범위가 불균형하다.** 16×64는19방법×4차원×4seeds가 모두 완료됐다. 256×16384의 두-mode 실험은1D 전체76개,2D64개(다차원 Exact 미완료),4D10개다. 큰 크기의 closed-loop 결과는 기존 row-argmax만 완료됐다. 따라서 모든 크기·차원에서 방법의 전체 순위를 매길 수 없다. 표의n은 완료 seed 수이고, ‘—’는 완료 결과가 없거나 해당 지표가 유효하지 않다는 뜻이다.

기존 row-argmax의 첫 TD update에서 발생한 zero-std truncated Gaussian 연결 오류는 수정 후 재검증했다. 이번 보고서는 수정된 source의 유효한 완료 결과만 사용한다. 이전의 오류 결과16개를 성능 저하나 collapse로 세지 않는다.

새 실행 범위는16×64 결과를 재사용하고, **256×1024·512×512**, learned-σ Sinkhorn **ε=1e−4,1e−3,1e−2,1e−1,1,10**을 사용한다. 사용자 목록에 적힌 값은6개다. 방법 13개,4차원,4seeds,5종류이며 새 비교 2,080개+prefix 832개를 실행한다. 축소 범위 전체 3,120개 중1,040개는 기존16×64 완료 결과다. 보고서는 제외된 epsilon·크기에서 이미 완료된 결과도 보존한다.

## 1. 실험 설정

### 1.1 비교 방법과 hyperparameter

| 항목 | 설정 |
|---|---|
| 완료 결과의 크기 | 16×64 / 256×16384 / 일부 1024×4096 |
| 차원·seed | D=1,2,4,8 · seed 0,1,2,3 |
| Temperature · actor optimizer | τ=0.25 · Adam 3×10⁻⁴ |
| NLL actor | v5 conditional squashed Gaussian, hidden 256×2, GELU |
| Learned σ | log σ 범위 [-5,1], 초기 σ=0.5 |
| Fixed σ 비교 | σ=0.1 또는 0.5; sampling과 NLL 모두 고정 |
| NLL proposal | conditional Gaussian mixture, pre-tanh σ floor 0.05, IID 후보, anchor 없음 |
| Importance weight | softmax(Q/τ − log q_F), proposal density correction 사용 |
| OT source 위치 | **tanh μᵢ**. 실제 actor 평가의 tanh(μᵢ+σᵢε) 샘플과 구분 |
| NLL OT cost | action 공간의 squared Euclidean distance 합, cost 정규화 없음 |
| Sinkhorn | log-domain, **100 iterations**, 이후 row normalization |
| Learned-σ Sinkhorn ε | 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50 |
| Fixed-σ Sinkhorn ε | 0.1 |
| Exact OT | 1D: 단조 정렬 coupling. D>1: 전체 좌표의 cost를 이용한 network-simplex/POT, CPU float64 |
| 연산 precision | GPU matmul `highest`; network/gradient는 float32 |
| 관측 주기 | 기본 200 updates, 변화 경계 근처 20 updates + 지정 직후 snapshot |
| Actor density 평가 | **실제 action 32,768개**, 좌표별 512 bins, KDE smoothing 없음 |
| 2D joint density | 실제 샘플의 64×64 histogram |

방법은 총 19개다. Direct GMM 1개, Exact OT의 learned/fixed σ 3개, fixed-σ Sinkhorn 2개, learned-σ Sinkhorn의 ε 12개, 기존 rowwise-argmax 1개를 비교한다.

기존 rowwise-argmax는 v5 Gaussian mean에 MSE를 붙인 버전이 아니다. **과거의 deterministic implicit actor** G(s,z), hidden 256×3를 사용하고 action을 [-1,1]로 clip한다. KDE std=0.2, local clip=0.5, stratified 후보, anchor 없음, mean-normalized OT cost, ε=0.05, 30 iterations, rowwise argmax MSE다. **NLL 방법과는 architecture·proposal·cost 설정도 다르므로, 순수한 loss-only ablation으로 해석하지 않는다.**

NLL 방법 간에는 다음 목적함수를 비교한다. R은 각 row가 합 1이 되도록 정규화한 plan이다.

$$
\mathcal L_{\rm OT}=-\frac1N\sum_{i,j}R_{ij}\log k_\theta(b_j\mid z_i),\qquad
\mathcal L_{\rm GMM}=-\sum_jw_j\log\left[\frac1N\sum_i k_\theta(b_j\mid z_i)\right].
$$

Fixed σ는 proposal의 범위와 conditional fitting 능력도 함께 바꾼다. 따라서 fixed/learned 비교를 “variance gradient만 켜고 끈 결과”로 축약할 수 없다.

### 1.2 직접 지정한 Q

Action은 [-1,1]ᴰ. 첫 좌표에만 여러 mode를 두고, 나머지는 표준편차 0.35인 독립 Gaussian으로 둔다. **고차원에서도 전체 joint mode 수가 2,3,6으로 유지된다.** 임의의 상관관계를 가진 고차원 target을 대표하지는 않는다.

$$
f_t(a)=\left[\sum_k\rho_{k,t}\mathcal N(a_1;c_{k,t},h^2)\right]
\prod_{d=2}^D\mathcal N(a_d;0,0.35^2),\qquad
Q_t(a)=0.25\log f_t(a),\qquad
\pi_t^*(a)=\frac{f_t(a)}{\int_{[-1,1]^D}f_t(x)\,dx}.
$$

| 환경 | 0–20K | 20K 이후 변화 |
|---|---|---|
| 두 mode | 중심 (-0.65,+0.65), h=0.12, 질량 50:50 | 20,001부터 80:20, 25,001부터 20:80, 30,001부터 50:50; 35K 종료 |
| 세 mode | 중심 (-0.6,0,0.6), h=0.1, 동일 질량 | (0.6,0.2,0.2) → (0.2,0.2,0.6) → 동일 질량; 경계는 위와 동일 |
| 3→6→3 | 같은 세-mode 초기 학습 | 중심 cₖ를 cₖ±d로 분리. d:20–22K 동안 0→0.15, 25K까지 유지, 25–27K에 0으로 복귀 |

각 방법·seed가 자기 actor를 처음부터 20K 학습한다. 세-mode의 질량 변화와 분리·병합은 **동일한 20K actor·Adam·RNG checkpoint에서 분기**한다. 초기 fitting이 나쁜 방법도 제외하지 않는다.

![Q와 정답 분포](N16_M64/figures/design_targets.png)

### 1.3 실제로 학습되는 Q

상태와 행동 모두 [-1,1]ᴰ이고 다음 상태는 s′=a다. 보상은 세 좋은 위치에 머물도록 하되 이동 비용을 둔다.

$$
g(a)=\sum_{c\in\{-0.6,0,0.6\}}e^{-(a_1-c)^2/(2\cdot0.1^2)}
\prod_{d=2}^De^{-a_d^2/(2\cdot0.35^2)},\qquad
r(s,a)=g(a)-\frac{0.25}{D}\|a-s\|_2^2.
$$

Reward-only twin critic, γ=0.99, target update 0.005, critic batch 256, Adam 3×10⁻⁴. Uniform exploration 5K steps 뒤 actor/critic 각각 step당 1회, 35K updates. Episode는 200 steps에서 reset하되 TD에서 terminal로 처리하지 않는다.

$$
y=r+0.99\min_{\ell=1,2}\bar Q_\ell(s',a'),\quad a'\sim\pi_\theta(\cdot\mid s'),\qquad
Q_{\rm extract}=\tfrac12(Q_1+Q_2).
$$

- **공통 Q replay:** seed·차원별로 learned-σ Sinkhorn ε=0.1, N16/M64, actor state batch 256으로 Q 궤적을 만든다. 매 update의 live critic 전체 parameter를 저장하고, 비교 actor 모두 같은 Qₜ(0,a)를 같은 속도로 받는다. 시간·action interpolation은 없다.
- **Closed-loop:** 각 방법이 자기 데이터를 수집하고 자기 critic도 학습한다. Actor state batch 1, critic batch 256. 이는 1D/다차원 진단 실험이며 MuJoCo 기본 학습 설정 전체의 재현은 아니다.


### 1.4 그림과 지표

Actor 곡선은 실제 32,768개 action을 뽑은 histogram이다. 정답 analytic density만 적분으로 bin 확률을 구하며, actor에는 별도 적분이나 KDE smoothing을 사용하지 않는다.

$$
\mathrm{TV}_{\rm hist}=\frac12\sum_b|\hat p_b-p_b^*|,\qquad
\mathrm{TV}_{\rm basin}=\frac12\sum_k|\hat m_k-m_k^*|.
$$

첫 식은 512개 bin별 확률을, 둘째는 target의 인접 peak 사이 골짜기로 나눈 영역별 총확률을 비교한다. 예를 들어 두 mode의 목표 질량이80:20인데 actor가 50:50이면 basin-mass TV는 0.30이다. 이 값만으로 mode의 폭이나 정확한 모양을 판단할 수 없다. ‘Mode TV’와 ‘basin-mass TV’를 별개의 지표처럼 혼용하지 않고 이 보고서에서는 후자로 통일했다.

4D·8D는 첫 좌표 marginal을 주 지표로 쓰고, 나머지 좌표의 marginal을 따로 표시한다. 2D surface는64×64 joint histogram의 높이가 density인 그림이다. 이는 4D·8D의 전체 joint TV와 다르다. 최종 atlas는 seed0를 예시로 표시하며, 추적 curve는 완료된 seed의 평균±표준편차다.

## 2. 16×64: 질량 이동과 분포 복구를 구분해야 한다

두 mode가 계속 존재하는 상태에서 질량 비율만 바꿨다. 아래 첫 표는 변화 뒤 1,000 updates 동안 **첫 좌표 histogram TV의 평균**을 세 번의 변화에 대해 평균한 값이다. 낮을수록 새로운 density를 더 빨리, 더 정확하게 따라간다. 초기 fitting이 나쁜 방법은 속도와 초기 오차가 함께 반영되므로, 순수한 반응 시간 하나로 해석하지 않는다.

| 방법 | 1D | 2D | 4D | 8D |
|---|---:|---:|---:|---:|
| Direct GMM | 0.273 (n=4) | 0.276 (n=4) | 0.272 (n=4) | 0.287 (n=4) |
| Exact OT · learned σ | 0.393 (n=4) | 0.391 (n=4) | 0.385 (n=4) | 0.407 (n=4) |
| Sinkhorn ε=0.001 | 0.372 (n=4) | 0.316 (n=4) | 0.513 (n=4) | 0.513 (n=4) |
| Sinkhorn ε=0.01 | 0.305 (n=4) | 0.323 (n=4) | 0.335 (n=4) | 0.371 (n=4) |
| Sinkhorn ε=0.1 | 0.350 (n=4) | 0.364 (n=4) | 0.380 (n=4) | 0.402 (n=4) |
| 기존 row-argmax | 0.506 (n=4) | 0.646 (n=4) | 0.771 (n=4) | 0.708 (n=4) |

같은 구간의 **basin-mass TV**는 다음과 같다. mode의 질량 비율이 맞아도 각 mode의 폭·위치는 틀릴 수 있으므로 위 표와 함께 본다.

| 방법 | 1D | 2D | 4D | 8D |
|---|---:|---:|---:|---:|
| Direct GMM | 0.110 (n=4) | 0.125 (n=4) | 0.123 (n=4) | 0.142 (n=4) |
| Exact OT · learned σ | 0.052 (n=4) | 0.043 (n=4) | 0.039 (n=4) | 0.039 (n=4) |
| Sinkhorn ε=0.001 | 0.336 (n=4) | 0.258 (n=4) | 0.500 (n=4) | 0.500 (n=4) |
| Sinkhorn ε=0.01 | 0.037 (n=4) | 0.034 (n=4) | 0.044 (n=4) | 0.054 (n=4) |
| Sinkhorn ε=0.1 | 0.025 (n=4) | 0.025 (n=4) | 0.034 (n=4) | 0.034 (n=4) |
| 기존 row-argmax | 0.326 (n=4) | 0.498 (n=4) | 0.499 (n=4) | 0.498 (n=4) |

![두 mode 추적과 질량 오차](N16_M64/figures/tracking_double_mass.png)

![좌우 질량의 시간 변화](N16_M64/figures/massflow_double_mass.png)

세 mode가 여섯 개로 갈라졌다가 합쳐지는 실험은 다른 결과를 보인다. 아래 값은 20,001–35,000updates의 평균 histogram TV다.

| 방법 | 1D | 2D | 4D | 8D |
|---|---:|---:|---:|---:|
| Direct GMM | 0.279 (n=4) | 0.279 (n=4) | 0.280 (n=4) | 0.280 (n=4) |
| Exact OT · learned σ | 0.229 (n=4) | 0.229 (n=4) | 0.245 (n=4) | 0.253 (n=4) |
| Sinkhorn ε=0.001 | 0.217 (n=4) | 0.205 (n=4) | 0.224 (n=4) | 0.245 (n=4) |
| Sinkhorn ε=0.01 | 0.221 (n=4) | 0.226 (n=4) | 0.245 (n=4) | 0.252 (n=4) |
| Sinkhorn ε=0.1 | 0.245 (n=4) | 0.248 (n=4) | 0.253 (n=4) | 0.255 (n=4) |
| 기존 row-argmax | 0.303 (n=4) | 0.296 (n=4) | 0.347 (n=4) | 0.404 (n=4) |

![분리와 병합 추적](N16_M64/figures/tracking_tri_split.png)

## 3. 256×16384: 완료된 조건에서의 변화

N과M을 함께 늘렸으므로 latent 수 효과와 teacher 후보 수 효과를 분리한 ablation은 아니다. 같은 target·seed·schedule에서 완료된 조건끼리 비교한다. 특히2D Exact가 비어 있는 것은 알고리즘 성능 때문이 아니라 계산이 끝나지 않았기 때문이다.

**두-mode 질량 변화 뒤 1,000 updates의 평균 histogram TV:**

| 방법 | 1D | 2D | 4D | 8D |
|---|---:|---:|---:|---:|
| Direct GMM | 0.235 (n=4) | 0.244 (n=4) | 0.198 (n=1) | — |
| Exact OT · learned σ | 0.384 (n=4) | — | — | — |
| Sinkhorn ε=0.001 | 0.319 (n=4) | 0.316 (n=4) | 0.272 (n=1) | — |
| Sinkhorn ε=0.01 | 0.195 (n=4) | 0.220 (n=4) | 0.222 (n=1) | — |
| Sinkhorn ε=0.1 | 0.220 (n=4) | 0.222 (n=4) | — | — |
| 기존 row-argmax | 0.478 (n=4) | 0.480 (n=4) | 0.461 (n=1) | — |

**동일 구간의 basin-mass TV:**

| 방법 | 1D | 2D | 4D | 8D |
|---|---:|---:|---:|---:|
| Direct GMM | 0.194 (n=4) | 0.219 (n=4) | 0.178 (n=1) | — |
| Exact OT · learned σ | 0.080 (n=4) | — | — | — |
| Sinkhorn ε=0.001 | 0.288 (n=4) | 0.293 (n=4) | 0.241 (n=1) | — |
| Sinkhorn ε=0.01 | 0.021 (n=4) | 0.019 (n=4) | 0.033 (n=1) | — |
| Sinkhorn ε=0.1 | 0.011 (n=4) | 0.011 (n=4) | — | — |
| 기존 row-argmax | 0.012 (n=4) | 0.010 (n=4) | 0.016 (n=1) | — |

![큰 크기의 변화 추적](N256_M16384/figures/tracking_double_mass.png)

![큰 크기의1D 최종 실제 histogram](N256_M16384/figures/atlas_double_mass_D1.png)

![큰 크기의2D 최종 실제 marginal histogram](N256_M16384/figures/atlas_double_mass_D2.png)

## 4. Critic이 학습될 때

공통 learned-Q 재생은 모든 비교 actor에 같은 live-critic 궤적을 제공한다. 자기 critic 학습에서는 수집 데이터와 critic 자체가 방법별로 달라진다. 두 실험을 동일한 인과 비교로 취급하지 않는다.

**공통 learned-Q 재생의 유효 reference 시점에서 평균 histogram TV:**

| 방법 | 1D | 2D | 4D | 8D |
|---|---:|---:|---:|---:|
| Direct GMM | 0.424 (n=4) | 0.295 (n=4) | 0.058 (n=4) | 0.060 (n=4) |
| Exact OT · learned σ | 0.410 (n=4) | 0.265 (n=4) | 0.089 (n=4) | 0.088 (n=4) |
| Sinkhorn ε=0.001 | 0.361 (n=4) | 0.239 (n=4) | 0.092 (n=4) | 0.081 (n=4) |
| Sinkhorn ε=0.01 | 0.407 (n=4) | 0.263 (n=4) | 0.089 (n=4) | 0.085 (n=4) |
| Sinkhorn ε=0.1 | 0.431 (n=4) | 0.273 (n=4) | 0.090 (n=4) | 0.089 (n=4) |
| 기존 row-argmax | 0.454 (n=4) | 0.317 (n=4) | 0.250 (n=4) | 0.634 (n=4) |

| 방법 | 1D | 2D | 4D | 8D |
|---|---:|---:|---:|---:|
| Direct GMM | 0.170 (n=4) | 0.082 (n=4) | — | — |
| Exact OT · learned σ | 0.372 (n=4) | — | — | — |
| Sinkhorn ε=0.001 | 0.184 (n=4) | 0.148 (n=3) | — | — |
| Sinkhorn ε=0.01 | 0.353 (n=4) | 0.267 (n=3) | — | — |
| Sinkhorn ε=0.1 | 0.411 (n=4) | 0.260 (n=4) | — | — |
| 기존 row-argmax | 0.472 (n=4) | 0.285 (n=3) | — | — |

위 두 표는 순서대로 16×64,256×16384다. D>1에서 Sobol 적분의 ESS와 반쪽 간 오차를 검증해 신뢰성이 낮은 시점은 평균에서 제외했다. Q가 거의 평평해도 TV는 낮아질 수 있으므로 낮은 TV를 높은 제어 성능으로 읽지 않는다.

![16×64 실제 actor-critic 결과](N16_M64/figures/tracking_tri_closed.png)

**16×64의 최종200-step stochastic return:**

| 방법 | 1D | 2D | 4D | 8D |
|---|---:|---:|---:|---:|
| Direct GMM | 122.689 (n=4) | 48.327 (n=4) | -10.157 (n=4) | -29.671 (n=4) |
| Exact OT · learned σ | 87.373 (n=4) | 55.958 (n=4) | -16.788 (n=4) | -30.828 (n=4) |
| Sinkhorn ε=0.001 | 113.183 (n=4) | 75.959 (n=4) | -8.647 (n=4) | -27.356 (n=4) |
| Sinkhorn ε=0.01 | 88.041 (n=4) | 58.828 (n=4) | -15.026 (n=4) | -29.855 (n=4) |
| Sinkhorn ε=0.1 | 78.037 (n=4) | 53.047 (n=4) | -16.246 (n=4) | -30.578 (n=4) |
| 기존 row-argmax | 112.126 (n=4) | 84.491 (n=4) | 52.980 (n=4) | 84.209 (n=4) |

256×16384 closed-loop16개와1024×4096 closed-loop7개는 모두 legacy row-argmax다. 다른 방법이 완료되지 않았으므로 큰 크기에서 GMM·OT와의return 비교 근거로 사용할 수 없다. 해당 수치는 아래 전체 데이터 표와 부록에 보존했다.

## 5. Teacher·assignment와 epsilon 진단

GMM 그림은 OT plan이 아니다. candidate j에 대한 component posterior responsibility γᵢⱼ를 계산한 뒤 wⱼγᵢⱼ를 그린 **effective assignment**다. 열의 합은 teacher weight wⱼ이고 row의 합은 1/N으로 고정되지 않는다. OT는 raw P를 표시한다. 정렬은 첫 action 좌표로 그림의 행·열만 재배열한 것이며, 다차원 OT solver를 1D로 바꾸지 않는다.

![원본과 action 정렬 assignment](N16_M64/figures/assignment_original_sorted.png)

![Proposal, weighted teacher, actor](N16_M64/figures/teacher_actor_histograms.png)

이 teacher 그림은 16×64의 한 번에 뽑은 64개 후보를 보여준다. fine-bin TV가 큰 데에는 유한 표본 효과도 있으므로, 이를 전부 proposal 오류로 해석하지 않는다. 비교 actor는 독립적인 32,768개 sample histogram이다. 그림은 64 bins로 rebin했고 smoothing하지 않았다.

![Epsilon별 정확도와 solver residual](N16_M64/figures/epsilon_accuracy_residual.png)

작은 epsilon에서 100회 Sinkhorn이 충분히 수렴하지 않을 수 있다. Row normalization 이후의 유효 column mass도 함께 확인해야 한다. 따라서 epsilon별 결과는 **고정100iterations의 구현 결과**이며, 수렴한 entropic OT의 이론적 성능으로 일반화하지 않는다.

## 6. 해석의 범위

- 분포 전체 오차, mode별 질량 오차, adaptation speed는 서로 다른 지표다. 빠른 질량 이동만으로 정확한Boltzmann extraction을 입증할 수 없다.
- 4D·8D 그림은 첫 좌표와 나머지 좌표의 marginal이다. 모든 joint mode 구조나 상관관계 복구를 보장하지 않는다. 본 analytic target도 첫 좌표에만 mode가 있고 다른 좌표는 독립인 통제된 문제다.
- 큰 크기의 완료 결과가 일부 방법에 치우친 것은 실행 시간과 queue 순서 때문이다. 미완료 조건을 0점으로 놓지 않았고, 완료 seed를 성능에 따라 고르지 않았다.
- GPU 종류가 섞였으므로 mixed-hardware wall clock을 알고리즘 고유의 speedup으로 주장하지 않는다. 주요 추적 비교는 update 수 기준이다.
- 초기 20K와 변화 후 35K의 개선에는추가 15K 학습 효과도 들어간다. stationary 35K 대조군이 없어 목표 변화 자체가 개선을 유발했다고 단정하지 않는다.

## 7. 전체 그림과 수치

모든 density atlas는 seed0에서 독립적으로 뽑은32,768개 action의 histogram이다. seed0가 완료되지 않았으면 빈 패널로 남겼다. 평균·SD와 각 seed의 값은 CSV에 있다.

### N=16, M=64

- [method_summary.csv](N16_M64/method_summary.csv)
- [run_summary.csv](N16_M64/run_summary.csv)
- [shift_events.csv](N16_M64/shift_events.csv)

<details>
<summary>N16_M64 · assignment_original_sorted</summary>

![assignment_original_sorted](N16_M64/figures/assignment_original_sorted.png)

</details>

<details>
<summary>N16_M64 · atlas_double_mass_D1</summary>

![atlas_double_mass_D1](N16_M64/figures/atlas_double_mass_D1.png)

</details>

<details>
<summary>N16_M64 · atlas_double_mass_D2</summary>

![atlas_double_mass_D2](N16_M64/figures/atlas_double_mass_D2.png)

</details>

<details>
<summary>N16_M64 · atlas_double_mass_D4</summary>

![atlas_double_mass_D4](N16_M64/figures/atlas_double_mass_D4.png)

</details>

<details>
<summary>N16_M64 · atlas_double_mass_D8</summary>

![atlas_double_mass_D8](N16_M64/figures/atlas_double_mass_D8.png)

</details>

<details>
<summary>N16_M64 · atlas_tri_closed_D1</summary>

![atlas_tri_closed_D1](N16_M64/figures/atlas_tri_closed_D1.png)

</details>

<details>
<summary>N16_M64 · atlas_tri_closed_D2</summary>

![atlas_tri_closed_D2](N16_M64/figures/atlas_tri_closed_D2.png)

</details>

<details>
<summary>N16_M64 · atlas_tri_closed_D4</summary>

![atlas_tri_closed_D4](N16_M64/figures/atlas_tri_closed_D4.png)

</details>

<details>
<summary>N16_M64 · atlas_tri_closed_D8</summary>

![atlas_tri_closed_D8](N16_M64/figures/atlas_tri_closed_D8.png)

</details>

<details>
<summary>N16_M64 · atlas_tri_mass_D1</summary>

![atlas_tri_mass_D1](N16_M64/figures/atlas_tri_mass_D1.png)

</details>

<details>
<summary>N16_M64 · atlas_tri_mass_D2</summary>

![atlas_tri_mass_D2](N16_M64/figures/atlas_tri_mass_D2.png)

</details>

<details>
<summary>N16_M64 · atlas_tri_mass_D4</summary>

![atlas_tri_mass_D4](N16_M64/figures/atlas_tri_mass_D4.png)

</details>

<details>
<summary>N16_M64 · atlas_tri_mass_D8</summary>

![atlas_tri_mass_D8](N16_M64/figures/atlas_tri_mass_D8.png)

</details>

<details>
<summary>N16_M64 · atlas_tri_replay_D1</summary>

![atlas_tri_replay_D1](N16_M64/figures/atlas_tri_replay_D1.png)

</details>

<details>
<summary>N16_M64 · atlas_tri_replay_D2</summary>

![atlas_tri_replay_D2](N16_M64/figures/atlas_tri_replay_D2.png)

</details>

<details>
<summary>N16_M64 · atlas_tri_replay_D4</summary>

![atlas_tri_replay_D4](N16_M64/figures/atlas_tri_replay_D4.png)

</details>

<details>
<summary>N16_M64 · atlas_tri_replay_D8</summary>

![atlas_tri_replay_D8](N16_M64/figures/atlas_tri_replay_D8.png)

</details>

<details>
<summary>N16_M64 · atlas_tri_split_D1</summary>

![atlas_tri_split_D1](N16_M64/figures/atlas_tri_split_D1.png)

</details>

<details>
<summary>N16_M64 · atlas_tri_split_D2</summary>

![atlas_tri_split_D2](N16_M64/figures/atlas_tri_split_D2.png)

</details>

<details>
<summary>N16_M64 · atlas_tri_split_D4</summary>

![atlas_tri_split_D4](N16_M64/figures/atlas_tri_split_D4.png)

</details>

<details>
<summary>N16_M64 · atlas_tri_split_D8</summary>

![atlas_tri_split_D8](N16_M64/figures/atlas_tri_split_D8.png)

</details>

<details>
<summary>N16_M64 · auc_double_mass</summary>

![auc_double_mass](N16_M64/figures/auc_double_mass.png)

</details>

<details>
<summary>N16_M64 · auc_tri_mass</summary>

![auc_tri_mass](N16_M64/figures/auc_tri_mass.png)

</details>

<details>
<summary>N16_M64 · auc_tri_split</summary>

![auc_tri_split](N16_M64/figures/auc_tri_split.png)

</details>

<details>
<summary>N16_M64 · design_targets</summary>

![design_targets](N16_M64/figures/design_targets.png)

</details>

<details>
<summary>N16_M64 · double_before_after</summary>

![double_before_after](N16_M64/figures/double_before_after.png)

</details>

<details>
<summary>N16_M64 · epsilon_accuracy_residual</summary>

![epsilon_accuracy_residual](N16_M64/figures/epsilon_accuracy_residual.png)

</details>

<details>
<summary>N16_M64 · massflow_double_mass</summary>

![massflow_double_mass](N16_M64/figures/massflow_double_mass.png)

</details>

<details>
<summary>N16_M64 · massflow_tri_mass</summary>

![massflow_tri_mass](N16_M64/figures/massflow_tri_mass.png)

</details>

<details>
<summary>N16_M64 · surface_D2_025200</summary>

![surface_D2_025200](N16_M64/figures/surface_D2_025200.png)

</details>

<details>
<summary>N16_M64 · surface_D2_final_independent</summary>

![surface_D2_final_independent](N16_M64/figures/surface_D2_final_independent.png)

</details>

<details>
<summary>N16_M64 · teacher_actor_histograms</summary>

![teacher_actor_histograms](N16_M64/figures/teacher_actor_histograms.png)

</details>

<details>
<summary>N16_M64 · tracking_double_mass</summary>

![tracking_double_mass](N16_M64/figures/tracking_double_mass.png)

</details>

<details>
<summary>N16_M64 · tracking_tri_closed</summary>

![tracking_tri_closed](N16_M64/figures/tracking_tri_closed.png)

</details>

<details>
<summary>N16_M64 · tracking_tri_mass</summary>

![tracking_tri_mass](N16_M64/figures/tracking_tri_mass.png)

</details>

<details>
<summary>N16_M64 · tracking_tri_replay</summary>

![tracking_tri_replay](N16_M64/figures/tracking_tri_replay.png)

</details>

<details>
<summary>N16_M64 · tracking_tri_split</summary>

![tracking_tri_split](N16_M64/figures/tracking_tri_split.png)

</details>

### N=256, M=16384

- [method_summary.csv](N256_M16384/method_summary.csv)
- [run_summary.csv](N256_M16384/run_summary.csv)
- [shift_events.csv](N256_M16384/shift_events.csv)

<details>
<summary>N256_M16384 · atlas_double_mass_D1</summary>

![atlas_double_mass_D1](N256_M16384/figures/atlas_double_mass_D1.png)

</details>

<details>
<summary>N256_M16384 · atlas_double_mass_D2</summary>

![atlas_double_mass_D2](N256_M16384/figures/atlas_double_mass_D2.png)

</details>

<details>
<summary>N256_M16384 · atlas_double_mass_D4</summary>

![atlas_double_mass_D4](N256_M16384/figures/atlas_double_mass_D4.png)

</details>

<details>
<summary>N256_M16384 · atlas_tri_closed_D1</summary>

![atlas_tri_closed_D1](N256_M16384/figures/atlas_tri_closed_D1.png)

</details>

<details>
<summary>N256_M16384 · atlas_tri_closed_D2</summary>

![atlas_tri_closed_D2](N256_M16384/figures/atlas_tri_closed_D2.png)

</details>

<details>
<summary>N256_M16384 · atlas_tri_closed_D4</summary>

![atlas_tri_closed_D4](N256_M16384/figures/atlas_tri_closed_D4.png)

</details>

<details>
<summary>N256_M16384 · atlas_tri_closed_D8</summary>

![atlas_tri_closed_D8](N256_M16384/figures/atlas_tri_closed_D8.png)

</details>

<details>
<summary>N256_M16384 · atlas_tri_replay_D1</summary>

![atlas_tri_replay_D1](N256_M16384/figures/atlas_tri_replay_D1.png)

</details>

<details>
<summary>N256_M16384 · atlas_tri_replay_D2</summary>

![atlas_tri_replay_D2](N256_M16384/figures/atlas_tri_replay_D2.png)

</details>

<details>
<summary>N256_M16384 · tracking_double_mass</summary>

![tracking_double_mass](N256_M16384/figures/tracking_double_mass.png)

</details>

<details>
<summary>N256_M16384 · tracking_tri_closed</summary>

![tracking_tri_closed](N256_M16384/figures/tracking_tri_closed.png)

</details>

<details>
<summary>N256_M16384 · tracking_tri_replay</summary>

![tracking_tri_replay](N256_M16384/figures/tracking_tri_replay.png)

</details>

### N=1024, M=4096

- [method_summary.csv](N1024_M4096/method_summary.csv)
- [run_summary.csv](N1024_M4096/run_summary.csv)
- [shift_events.csv](N1024_M4096/shift_events.csv)

<details>
<summary>N1024_M4096 · atlas_tri_closed_D1</summary>

![atlas_tri_closed_D1](N1024_M4096/figures/atlas_tri_closed_D1.png)

</details>

<details>
<summary>N1024_M4096 · atlas_tri_closed_D2</summary>

![atlas_tri_closed_D2](N1024_M4096/figures/atlas_tri_closed_D2.png)

</details>

<details>
<summary>N1024_M4096 · tracking_tri_closed</summary>

![tracking_tri_closed](N1024_M4096/figures/tracking_tri_closed.png)

</details>

## 8. 재현과 보관

원래 수치 source: `cc11f537af330e23e1cc77cb94a9426660b55ebb`. Legacy closed-loop 수정 source: `91cb9c3d32f9d3c7f43ff886d34b6199080be811`. 새 축소 설정 commit: `a08517ef9ed5fb8743252132997638b00feb5d75` (이 보고서의기존 실험을 새 commit으로 재표기하지 않음).

원본 checkpoint·metric·sample은 `dildata:/data1/heejoonorm/OptiQ/studies/20260918_nonstationary_nd/`의 각 revision에 보관한다. 이 보고서는 `dildata:/data1/heejoonorm/OptiQ/reports/20260919_nonstationary_nd_completed/`에 저장한다. HTML은 그림과 수식을 내장하므로 인터넷 없이 열린다. Markdown은 같은 폴더의 그림·CSV를 참조한다.

[집계 시점 전체 inventory](inventory.json)
