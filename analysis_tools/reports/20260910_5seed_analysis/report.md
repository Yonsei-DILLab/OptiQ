# OptiQ Boltzmann backup — 5-seed 분석 리포트

**2026-09-10 · seeds 0–4 확정 · 수정 control 검증 완료 · 추가 seed 확장 중단**

이번 분석의 질문은 두 가지다. **OptiQ의 actor가 local sampling보다 Boltzmann backup을 정확하게 추정하는가? 그리고 정확한 Boltzmann extraction이 SD2·SD3에서 논의한 학습·landscape 장점으로 이어지는가?** 완료된 다섯 seed를 모두 사용해 평균, 분포, 실제 backup 구현, landscape를 나누어 확인했다. SAC 비교와 대규모 benchmark는 포함하지 않았다.

## 1. 이번 결과가 말해주는 것

| 질문 | 관측 결과 | 현재 판단 |
|---|---|---|
| 같은 learned Q에서 backup이 더 정확한가? | Raw OptiQ RMSE **0.0606**, local no-IS **0.1413**. 다섯 seed 모두 raw가 우세 | 이 설정에서는 actor sampling의 이점이 뚜렷하다 |
| Production 방식의 TD action noise를 더하면? | 같은 online actor에 noise를 넣으면 RMSE **0.2485** | Noise의 영향이 크다. Target actor까지 포함한 실제 target 전체 오차와는 구분한다 |
| 복잡한 분포도 정확하게 유지하는가? | 4D·8D에서 넓은 coverage 초기화도 첫 100 OT updates 안에 한 mode로 집중 | 표현할 수 있는 분포와 현재 update로 유지되는 분포를 구분해야 한다 |
| 어느 단계에서 목표 분포와 차이가 생기는가? | 가중 후보 → row-argmax target에서 Q 평균 변화. Sinkhorn 300회에서도 지속 | Candidate weighting, 유한 Sinkhorn, hard target 선택을 분리해 볼 근거를 확보했다 |
| Landscape가 더 smooth한가? | 수정 control과 같은 경로의 cross-critic 비교에서 일관된 우위 없음 | 이번 다섯 seed로 smoothing 주장을 재현했다고 쓰기는 어렵다 |

**가장 중요한 새 발견은 OT coupling의 marginal을 맞추는 것과, 그 coupling에서 row-argmax로 뽑은 target의 분포를 맞추는 것이 다르다는 점이다.** 이 차이는 실제 저장된 production actor에서 관측됐고, Sinkhorn 반복 수만 늘리는 진단으로도 사라지지 않았다. 다만 이것이 장기 mode collapse의 유일한 원인이라는 인과 결론까지 내린 것은 아니다.

## 2. 실행 범위와 수치 검증

사용자 요청에 따라 확장 array `2193747`과 후속 gate `2193748`을 취소했다. 분석 seed는 **0, 1, 2, 3, 4**로 고정했고 자동 확장도 비활성화했다. 이미 존재하는 추가 seed 파일은 보존했지만 집계에 넣지 않았다. 보고서 작성 중 최종 scheduler 조회에서 실행·대기 작업은 없었다.

| 완료 결과 | 개수 | 구성 |
|---|---:|---|
| Frozen analytic Q | 130 | 13개 문제 × default/coverage 초기화 × 5 seeds |
| MoveCar 학습 | 25 | OptiQ, DDPG, SD2, TD3, SD3 × 5 seeds; 각 1M steps |
| 수정 fixed-replay control | 20 | 네 가지 backup × 5 seeds; 각 20k updates |
| 각 방법 자신의 critic landscape | 25 | 다섯 방법 × 5 seeds |
| Learned-Q backup 진단 | 5 | seed당 32개 상태; 공통 Q에서 네 추정기 비교 |
| 동일 actor 경로의 cross-critic landscape | 15 | 세 가지 경로 × 5 seeds |
| **완료 디렉터리 합계** | **220** | 학습과 후처리 디렉터리를 모두 포함 |

추가로 **기존 actor checkpoint만 재평가하는 OT audit**을 수행했다. 4개 Q 문제 × 2개 초기화 × 2개 시점 × 5 seeds = 80조건이다. 조건마다 256개 batch group을 4회 평가했다. 이 반복은 새 학습 seed가 아니다. Actor 파라미터를 새로 학습하거나 저장하지 않았으며, 300 Sinkhorn 반복은 coupling 진단에만 사용했다.

### Control의 적분 문제는 해결됐다

기존 257점 control을 float64로 다시 검사했을 때 13/20에서 실제 격자 오차가 1e-3보다 컸다. 따라서 단순히 허용치를 완화하지 않고, 네 arm을 공통으로 **2056점 학습 격자 + float32 highest matmul precision**으로 다시 학습했다. 최종 기준값은 저장 가중치를 float64로 평가하고, 고정 계산 chunk를 사용해 격자를 두 배씩 늘렸을 때 연속 두 번 변화량이 1e-4 아래로 내려가는지 확인했다.

수정 control **20/20 통과**, 학습 격자와 최종 reference의 최대 차이는 **3.8491×10⁻⁵**다. Gate 허용치는 기존 **1e-3**을 유지했다. 이 검증은 지정된 probe 상태에서 최종 Q의 적분 정확도를 확인한다. 학습 과정 전체의 함수 근사 오차나 모든 상태에 대한 전역 오차 보장은 아니다. 이번 보고서는 수정 control만 사용하고, 이전 보고서는 과거 스냅샷으로 보존한다.

## 3. 정확히 무엇을 비교했는가?

상태를 고정한 continuous-action Boltzmann distribution과 그 backup 값은 다음과 같다. 모든 analytic 문제의 action 범위는 [−1,1]ᵈ이고, τ=0.25다.

```text
πQτ(a|s) = exp(Q(s,a)/τ) / ∫ exp(Q(s,b)/τ) db
BτQ(s)  = ∫ πQτ(a|s) Q(s,a) da
TBQ(s,a) = r(s,a) + γ E[BτQ(s′)]

Raw actor: B̂raw = (1/K) Σ Q(s, gθ(s,zₖ))
TD sampler: 먼저 aₖ ~ actor, 그다음 ãₖ ~ Pnoise(·|aₖ)
            B̂TD = (1/K) Σ Q(s,ãₖ)
Bias = E[B̂] − BτQ,    RMSE² = Bias² + Var(B̂)
```

`Pnoise`는 production의 action 경계 및 perturbation 범위를 반영한 truncated Gaussian sampler다. 이를 추가한 추정기는 raw actor가 생성하는 분포와 다른 분포를 적분한다. RMSE가 낮다는 것은 **Q 평균**을 잘 맞춘다는 뜻이며, mode coverage 또는 전체 확률밀도의 일치를 자동으로 뜻하지 않는다.

### Production OptiQ를 사용했는지 점검

Frozen-Q fitting은 analytic Q를 두 개의 동일한 critic 출력으로 제공하고, **production `OptiQDIME.update_actor`를 직접 호출**한다. MoveCar는 production OptiQ 학습 경로를 사용한다. 기본값은 256×3 GELU implicit actor, latent 차원=action 차원, policy sample 16개 × sample당 candidate 4개, KDE std 0.2 / clip 0.5, density correction 1, Sinkhorn ε=0.05 / 30 iterations, **row-argmax target + MSE**다. 학습률 3e-4, batch 256, γ=0.99, UTD 1, warm-up 5k를 사용했다.

저장 checkpoint의 입력 차원과 Jacobian을 확인했다. 2D·8D 문제에 1D latent를 잘못 넣은 문제는 발견하지 못했다. OT audit 재계산과 실제 production update의 actor loss 및 ESS도 네 대표 문제에서 차이 0으로 일치했다. 이 검사는 호출 경로와 진단의 일치를 확인하는 것이며, 구현 전체가 이상적인 Boltzmann extraction을 보장한다는 뜻은 아니다.

실제 production에서는 **actor가 live twin-mean Q를 추종하고, critic backup은 target actor·target twin-min Q에 TD action noise까지 사용**한다. 따라서 이상적인 단일-Q Boltzmann operator와의 차이는 actor 표현력 하나만으로 설명되지 않는다.

Fixed-replay control은 이를 분리하기 위해 만든 **수정 알고리즘**이다. 동일한 65,536개 uniform replay, critic 초기화, implicit actor 초기화를 사용하고 backup만 grid max / grid Boltzmann / actor sampling / local no-IS로 바꾼다. 단일 scalar Q를 actor interface에 동일 twin으로 제공하고, actor extraction과 backup에 같은 target Q를 사용하며 TD noise는 제거했다. 이 결과를 기본 OptiQ의 성능으로 표기하지 않는다.

DDPG·SD2·TD3·SD3는 같은 256×3 GELU 규모에 맞춘 JAX 구현이다. Local 후보 K=50, β=4 등을 사용했으며 **공식 코드와 모든 세부 설정이 동일한 재현은 아니다.** 아래 frozen local sampler는 전체 SD2·SD3 학습 알고리즘과 구분한다.

## 4. Backup 평균: learned Q에서는 이점, 복잡한 frozen Q에서는 한계

### 4.1 Learned MoveCar Q: raw actor의 이점은 다섯 seed 모두에서 나타난다

각 seed의 저장된 **같은 target twin-min Q**를 모든 comparator에 제공했다. 상태 32개, K=50, Monte Carlo 반복 2,000회다. Q는 모든 추정기에서 저장 가중치를 동일하게 float64로 평가했다. Local sampler의 중심에는 reference 격자에서 찾은 global argmax를 제공했으므로, 중심 선택은 local 방법에 유리한 조건이다. Raw actor는 학습 중 live twin-mean을 추종한 **online actor**다. TD 진단도 같은 online actor에 production 방식의 noise를 추가한다. 실제 critic update의 target actor와 online actor 사이 차이는 이 실험에서 별도로 평가하지 않았으므로, 아래 RMSE를 실제 production target 전체의 오차와 동일시하지 않는다.

| 추정기 | S0 | S1 | S2 | S3 | S4 | 평균 RMSE | seed SD |
|---|---|---|---|---|---|---|---|
| OptiQ raw actor | 0.05486 | 0.09740 | 0.05199 | 0.04693 | 0.05178 | 0.06059 | 0.02077 |
| OptiQ + TD action noise | 0.26139 | 0.27985 | 0.23202 | 0.24749 | 0.22179 | 0.24851 | 0.02310 |
| Local no-IS | 0.13412 | 0.14285 | 0.14246 | 0.14484 | 0.14214 | 0.14128 | 0.00414 |
| Local official-style IS | 0.13789 | 0.14353 | 0.14163 | 0.14702 | 0.14000 | 0.14201 | 0.00348 |


![그림 1. 동일 learned Q에서의 backup 비교. 왼쪽은 seed당 32개 상태 RMSE의 평균, 오른쪽은 상태별 bias다. 연결선 하나가 독립 학습 seed 하나다.](figures/01_learned_backup.png)

*그림 1. 동일 learned Q에서의 backup 비교. 왼쪽은 seed당 32개 상태 RMSE의 평균, 오른쪽은 상태별 bias다. 연결선 하나가 독립 학습 seed 하나다.*


Raw actor의 seed-평균 RMSE는 local no-IS보다 **약 57% 작다**. 다섯 seed 모두 같은 방향이며, 상태별로는 **139/160**에서 raw RMSE가 더 작았다. 상태 160개는 다섯 학습 결과 안에 속한 반복 관측이므로 160개의 독립 실험처럼 유의성을 계산하지 않았다.

TD noise를 더하면 RMSE가 raw의 **약 4.1배**가 된다. Noise가 만든 추정 평균 변화는 **160개 상태 모두 음수**이고, seed와 상태에 걸친 평균 변화는 **−0.2153**이다. 이를 다음처럼 분해할 수 있다.

```text
eTD = eraw + Eactor[ Enoise[Q(s,ã)|a] − Q(s,a) ]
      추출 오차          noise가 backup 평균을 바꾸는 항
```

이 데이터에서는 noise 항이 큰 음의 방향으로 작용한다. Q의 peak 근방에서 action을 퍼뜨리면 평균 Q가 내려가는 해석과 맞지만, 모든 Q에서 항상 음수가 되는 성질은 아니다. **“TD noise를 없애면 학습 return도 개선된다”는 결론은 아직 검증하지 않았다.** 지금 확인한 것은 저장 Q·actor에 대한 추정 오차다.

### 4.2 Frozen Q: 추정 평균을 직접 비교

다음은 default actor 20k updates, K=50, seed 0–4 결과다. Local 범위는 각각의 알려진 mode 중심에서 얻은 **seed-평균의 최소–최대**다. 가장 나쁜 중심만 골라 비교하지 않았고, 이 범위를 신뢰구간으로 해석하지 않는다.

| 고정 Q | 참값 BτQ | Raw 추정 평균 | TD 추정 평균 | Local 중심별 평균 범위 |
|---|---|---|---|---|
| 1D single | 0.3037 | 0.3348 | 0.1946 | 0.3588–0.3588 |
| 1D asymmetric | 0.2686 | 0.3394 | 0.1804 | 0.1194–0.3445 |
| 2D / 4 modes | 0.5363 | 0.5196 | -0.1393 | 0.3546–0.7013 |
| 2D / 8 modes | 0.3873 | 0.4082 | 0.0319 | 0.1824–0.5431 |
| 4D separable | 0.7061 | 1.0668 | 0.7204 | 0.5295–0.9953 |
| 8D separable | 1.4121 | 2.2651 | 1.5266 | 0.9661–1.9869 |

| 고정 Q | Raw RMSE ± seed SD | TD RMSE | Local 중심별 RMSE 범위 |
|---|---|---|---|
| 1D single | 0.0342 ± 0.0012 | 0.1174 | 0.0561–0.0561 |
| 1D asymmetric | 0.0721 ± 0.0031 | 0.0996 | 0.0769–0.1506 |
| 2D / 4 modes | 0.1224 ± 0.0048 | 0.6862 | 0.0552–0.1847 |
| 2D / 8 modes | 0.0412 ± 0.0024 | 0.3644 | 0.0287–0.2278 |
| 4D separable | 0.3608 ± 0.0020 | 0.0519 | 0.1909–0.2905 |
| 8D separable | 0.8530 ± 0.0024 | 0.1313 | 0.4998–0.5778 |


![그림 2. 참값과 추정 평균, 그리고 RMSE. 오차막대는 학습 seed SD다. Local 선은 중심 선택에 따른 범위다.](figures/02_frozen_summary.png)

*그림 2. 참값과 추정 평균, 그리고 RMSE. 오차막대는 학습 seed SD다. Local 선은 중심 선택에 따른 범위다.*


1D 및 일부 2D 문제에서는 raw actor가 좋은 평균을 제공하지만, **4D·8D에서는 raw actor도 참값보다 높은 평균에 치우친다.** 반대로 이 두 문제에서는 TD noise가 scalar RMSE를 낮춘다. 따라서 “noise는 항상 나쁘다”도, “raw actor는 local estimator보다 항상 좋다”도 현재 결과에 맞지 않는다. 같은 방법의 분포가 잘못되어 있어도 추가 noise가 Q 평균 오차를 상쇄할 수 있다.


![그림 3. K=1부터 1,024까지 backup RMSE. Actor와 reference의 음영은 seed 최소–최대, 주황색 음영은 local 중심별 seed-평균 범위다. 모든 곡선은 같은 고정 Q를 사용한다.](figures/03_rmse_vs_k.png)

*그림 3. K=1부터 1,024까지 backup RMSE. Actor와 reference의 음영은 seed 최소–최대, 주황색 음영은 local 중심별 seed-평균 범위다. 모든 곡선은 같은 고정 Q를 사용한다.*


K를 늘리면 reference sampler의 Monte Carlo 오차는 줄지만, actor 또는 local sampler의 편향은 남는 경우가 많다. 저장된 추정치에서 `RMSE² = Bias² + Var` 항등식도 수치 오차 약 4.6×10⁻¹⁴ 이내로 확인했다. **추정당 action 수 K를 늘리는 것과, actor update의 candidate budget을 늘려 분포 자체를 바꾸는 것은 다른 조치다.**

### 4.3 Local importance sampling은 support 밖의 질량을 복구하지 못한다

올바른 proposal density q로 self-normalized importance sampling을 하더라도, q의 support가 S로 제한되면 다음 극한을 갖는다.

```text
B̂IS = Σ [exp(Q(aₖ)/τ)/q(aₖ)] Q(aₖ) / Σ [exp(Q(aₖ)/τ)/q(aₖ)]

K → ∞일 때 B̂IS → EπQτ[Q | a∈S] = BS
BS − B = PπQτ(Sᶜ) · (E[Q|S] − E[Q|Sᶜ])
```

위 극한은 **올바르게 정규화된 truncated proposal IS**에 해당한다. No-IS나 noise clipping 이전의 Gaussian density를 사용하는 official-style 구현에 같은 식을 그대로 적용하지 않았다.

Support 질량은 이 analytic Gaussian mixture에서 CDF로 계산했다. 조건부 평균은 support 경계에 맞춘 float64 midpoint grid 256→512→1024로 재검증했으며, 마지막 refinement 차이의 최대는 **1.96×10⁻⁶**이다.


![그림 4. Local sampler의 K=1,024 추정 평균과 올바른 IS의 support-조건부 극한. 가로축 백분율은 proposal support 안에 들어오는 목표 질량이다. Official-style IS와 correct truncated IS는 다른 구현이다.](figures/07_local_support.png)

*그림 4. Local sampler의 K=1,024 추정 평균과 올바른 IS의 support-조건부 극한. 가로축 백분율은 proposal support 안에 들어오는 목표 질량이다. Official-style IS와 correct truncated IS는 다른 구현이다.*


2D 8-mode에서 각 중심의 support는 전체 목표 질량의 **약 13–43%**만 포함한다. 다만 center 4는 약 **32%**만 포함해도 조건부 평균 0.3778이 전체 참값 0.3873과 가깝다. **평균이 비슷하다는 이유만으로 여러 mode를 복구했다고 판단하면 안 되는 구체적인 예**다. 8D의 all-positive 중심은 약 **4.98%** 질량만 포함하고 조건부 평균은 1.7098, 전체 참값은 1.4121이다.

이 support 문제는 OptiQ에도 관련된다. OptiQ는 여러 actor sample 주변의 KDE를 사용하므로 한 deterministic 중심보다 넓게 후보를 공급할 수 있지만, 현재 actor가 한 mode에 집중하고 KDE perturbation 범위가 제한되면 잃은 mode를 다시 찾기 어려울 수 있다. **Expressive network의 잠재적 표현력만으로 실제 candidate support가 보장되지는 않는다.**

## 5. Mode coverage와 OT update를 분리해서 추적

### 5.1 넓은 초기 분포도 첫 100 updates 안에 무너지는 경우가 있다

Coverage 초기화는 reference를 이용한 inverse-CDF 대응을 2,000회 supervised warm-up한 진단 조건이다. 기본 OptiQ 성능에는 포함하지 않는다. 각 checkpoint에서 actor 및 reference 표본 32,768개를 사용했다.

Mode-bin discrepancy는 mode별 질량의 차이 `½Σ|pⱼ−pⱼ*|`다. 전체 continuous density의 TV distance가 아니다. 유효 mode 수는 `1/Σpⱼ²`로 정의했다. Reference의 bin 질량도 유한 표본이므로 작은 차이에는 sampling 오차가 포함된다.

| Coverage 초기화 | 업데이트 0 | 100 | 20k | 최종 actor 유효 mode 수 |
|---|---|---|---|---|
| 2D / 4 modes | 0.0329 | 0.1021 | 0.1008 | 2.76 |
| 2D / 8 modes | 0.0414 | 0.0331 | 0.0426 | 6.26 |
| 4D separable | 0.0206 | 0.7705 | 0.7733 | 1.00 |
| 8D separable | 0.0564 | 0.9479 | 0.9484 | 1.00 |


![그림 5. Default와 coverage 초기화의 mode-bin 차이. 가는 선은 다섯 seed, 굵은 선은 평균이다. 첫 두 checkpoint 사이에 발생한 변화를 보이기 위해 가로축 간격은 범주형이다.](figures/04_mode_history.png)

*그림 5. Default와 coverage 초기화의 mode-bin 차이. 가는 선은 다섯 seed, 굵은 선은 평균이다. 첫 두 checkpoint 사이에 발생한 변화를 보이기 위해 가로축 간격은 범주형이다.*


4D·8D의 coverage 초기화는 실제로 넓었다. 초기 평균 action 표준편차는 약 0.61이고, bin discrepancy도 각각 0.0206, 0.0564였다. 그런데 **100번의 기존 OT update 뒤에는 한 mode로 집중**했고, 20k까지 회복하지 않았다. 따라서 이 결과를 단순히 “처음부터 coverage 초기화를 못 했다”로 설명할 수는 없다. 8D는 256개의 조합 mode를 가지며, 최종 actor가 남긴 한 mode의 목표 질량은 약 5%다.

Default 2D 8-mode는 최종 bin discrepancy **0.0324**로 mode 질량은 잘 맞는다. 반면 2D 4-mode의 default 평균은 **0.1794**, coverage 초기화 후 최종 평균은 **0.1008**로 더 큰 차이가 남는다. 1D single-mode에서도 coverage 초기 표준편차 약 0.1801이 최종 0.1576으로 줄어 목표보다 좁아졌다.

고차원 separable Q는 좌표별 Q의 합으로 만들었다. 차원이 증가하면 전체 Q 범위도 커지므로, 이 결과를 Q scale을 고정한 순수한 차원 효과로 해석하지는 않는다. 또한 default 초기 출력이 원점 주변에 모여도 mode 경계 분할 때문에 bin 수가 많아 보일 수 있어, 초기 coverage 판단에는 action spread도 함께 확인했다.


![그림 6. Seed 0의 2D target와 actor 밀도 예시. 각 행은 공통 색상 스케일이며 밝기는 bin 질량을 나타낸다. 정량 결론은 모든 seed를 사용하고, 이 그림은 mode 내부의 좁아짐을 보여주는 예시다.](figures/05_density.png)

*그림 6. Seed 0의 2D target와 actor 밀도 예시. 각 행은 공통 색상 스케일이며 밝기는 bin 질량을 나타낸다. 정량 결론은 모든 seed를 사용하고, 이 그림은 mode 내부의 좁아짐을 보여주는 예시다.*


### 5.2 Coupling이 맞춘 질량이 row-argmax target까지 보존되지는 않는다

Production update를 다음 단계로 나눠 저장 actor에 다시 적용했다.

```text
Actor actions → KDE candidates → critic/density weights w
              → Sinkhorn coupling T → row-argmax targets → MSE regression

이상적인 marginal: Σⱼ Tᵢⱼ = 1/N,   Σᵢ Tᵢⱼ = wⱼ
Row sampling의 기대 질량: w̃ⱼ = (1/N) Σᵢ Tᵢⱼ / ΣₗTᵢₗ
Row-argmax target 질량:   ŵⱼ = (1/N) Σᵢ 1[j = argmaxₗ Tᵢₗ]
```

Marginal이 정확하면 row sampling의 기대 질량 w̃는 w와 같다. 그러나 **row-argmax가 만드는 ŵ는 일반적으로 w와 같지 않다.** 유한 Sinkhorn 반복에서는 row marginal도 아직 정확하지 않을 수 있어 row normalization 단계의 오차도 따로 측정했다.

아래는 default 20k actor에서 단계별 target Q 평균이다. ‘Row 기대’는 실제로 actor를 새로 학습한 결과가 아니라, 해당 coupling에서 row별로 sampling한다고 할 때의 기대값이다. 300회 결과도 같은 actor·후보·가중치에서 coupling 계산만 길게 한 진단이다.

| Q | 참값 | Actor | 가중 후보 | Row 기대 30 | Argmax 30 | Row 기대 300 | Argmax 300 |
|---|---|---|---|---|---|---|---|
| 1D single | 0.30371 | 0.33495 | 0.30267 | 0.30318 | 0.31748 | 0.30267 | 0.31686 |
| 2D / 4 modes | 0.53628 | 0.51960 | 0.56693 | 0.56513 | 0.69137 | 0.56693 | 0.68985 |
| 2D / 8 modes | 0.38734 | 0.40943 | 0.38962 | 0.38901 | 0.49841 | 0.38963 | 0.49625 |
| 8D separable | 1.41213 | 2.26512 | 1.66708 | 1.66711 | 1.73852 | 1.66708 | 1.73852 |


![그림 7. OT target 선택 진단. 왼쪽: 2D 8-mode의 단계별 Q 평균. 가운데: 30→300 반복에서 row marginal 오차 감소. 오른쪽: 같은 2D actor의 Jacobian 최소/최대 특이값 비율. 각 점은 독립 학습 seed에서 계산한 진단 요약이다.](figures/06_ot_audit.png)

*그림 7. OT target 선택 진단. 왼쪽: 2D 8-mode의 단계별 Q 평균. 가운데: 30→300 반복에서 row marginal 오차 감소. 오른쪽: 같은 2D actor의 Jacobian 최소/최대 특이값 비율. 각 점은 독립 학습 seed에서 계산한 진단 요약이다.*


**2D 8-mode가 가장 명확하다.** 참값은 0.38734, 가중 후보 평균은 0.38962, 30회 row 기대값은 0.38901이다. 그런데 row-argmax 회귀 target 평균은 **0.49841**로 높아진다. Sinkhorn을 300회 돌리면 row 기대값은 0.38963으로 가중 후보 평균에 가까워지지만, argmax target 평균은 여전히 **0.49625**다. 이 조건에서 **반복 수만 늘리는 것으로 target 선택의 평균 변화가 해결되지는 않았다.**

2D 8-mode의 평균 row marginal L1 오차는 **0.07962→0.000108**, 2D 4-mode는 **0.12570→0.000741**로 감소했다. 유한 Sinkhorn 수렴 오차는 존재하지만, 그것과 hard selection의 영향은 구별된다. Candidate 64개에 target 16개만 대응하므로 hard histogram과 w의 TV가 큰 것 자체에는 양자화 효과도 있다. 따라서 그 지표 하나로 문제를 단정하지 않고, 여러 batch에 걸친 **체계적인 Q 평균 변화가 300회에서도 유지되는 점**을 근거로 삼았다.

2D 8-mode에서 actor Jacobian의 최소/최대 특이값 비율은 default 초기 평균 약 **0.308**에서 최종 **0.00229**로 작아졌다. 이는 시각적으로 나타난 분포의 좁아짐과 맞는 국소적 진단이다. Actor의 latent는 실제로 2D이며, architecture 자체의 차원을 잘못 구성한 결과는 아니다. 다만 이 비율은 전역 rank나 전체 밀도의 정확한 오차 척도는 아니다.

### 5.3 8D에서는 candidate weighting 단계도 이미 어렵다

8D coverage 초기화의 post-hoc bin discrepancy는 actor **0.0624**, unweighted candidates **0.0633**이지만, 가중 후보에서는 **0.2792**, argmax target에서는 **0.2930**으로 커진다. 이때 candidate ESS는 평균 **6.41/64**다. 이후 한 mode로 집중한 최종 actor에서는 ESS가 오히려 **약 50.96/64**로 높아진다. 즉 **ESS가 높아도 전체 mode coverage는 나쁠 수 있다.**

또한 8D의 최종 가중 후보 Q 평균은 1.6671로, 이미 전체 참값 1.4121보다 높다. 여기서는 row-argmax 하나만으로 모든 오차를 설명할 수 없다. Finite candidate weighting, 현재 actor에 의존하는 proposal support, target 선택, MSE regression이 함께 관련될 가능성이 있다.

지금 확보한 것은 **오차가 나타나는 단계를 분리한 증거**다. Row sampling으로 target을 바꾸기만 하면 mode collapse가 해결된다고 주장하지 않는다. Random target을 사용해도 MSE가 조건부 평균을 학습하면서 분포를 좁힐 수 있으므로, target marginal과 최종 actor 분포를 모두 측정하는 후속 ablation이 필요하다.

## 6. SD2·SD3의 landscape 주장은 어디까지 재현됐는가?

### 6.1 원 논문의 핵심 실패 상황은 이번 MoveCar에서 나타나지 않았다

[Softmax Deep Double Deterministic Policy Gradients, §4.1](https://papers.nips.cc/paper/2020/file/884d247c6f65a96a7da4d1105d584ddd-Paper.pdf)는 MoveCar에서 나쁜 DDPG policy와 좋은 SD2 policy를 놓고 critic이 유도하는 value objective의 perturbation과 두 policy 사이 경로를 비교했다. 우리 설정에서는 DDPG·SD2·TD3·SD3가 모두 최종 return 188에 도달했다. **따라서 나쁜 local optimum에서 벗어나는 원 논문의 상황 자체를 재현하지 못했다.**

이하 landscape는 `LQ(θ)=−E[Q(s,gθ(s,z))]` 또는 deterministic actor에 대응하는 objective다. **OptiQ가 실제로 최소화하는 OT regression loss의 landscape는 아니다.** Probe 상태와 latent를 고정했고, 경로는 두 actor의 파라미터 선형 보간이다. Barrier는 `maxα L(α) − max(L(0),L(1))`로 정의했다. 특정 경로의 barrier나 국소 perturbation만으로 전역 smoothness나 contraction을 증명할 수 없다.

### 6.2 수치가 검증된 fixed-replay control에서도 seed별 결과가 다르다

| Backup | S0 barrier | S1 barrier | S2 barrier | S3 barrier | S4 barrier | 평균 barrier | 국소 2차 차분 |
|---|---|---|---|---|---|---|---|
| Grid max | 0.05049 | 0.00000 | 0.16850 | 0.00000 | 0.00000 | 0.04380 | 0.01679 |
| Grid Boltzmann | 0.07281 | 0.00000 | 0.09593 | 0.00000 | 0.00000 | 0.03375 | 0.02133 |
| Actor K=1 | 0.08841 | 0.00000 | 0.03131 | 0.00000 | 0.00000 | 0.02394 | 0.02494 |
| Local K=50 | 0.02899 | 0.00000 | 0.11824 | 0.00000 | 0.00000 | 0.02945 | 0.01937 |


![그림 8. 수정 control의 seed별 barrier, 국소 2차 차분, 최종 적분 검증 오차. 동일 seed끼리 replay와 초기화를 공유한다. 최종 critic과 actor endpoint는 학습 arm에 따라 달라진다.](figures/08_control.png)

*그림 8. 수정 control의 seed별 barrier, 국소 2차 차분, 최종 적분 검증 오차. 동일 seed끼리 replay와 초기화를 공유한다. 최종 critic과 actor endpoint는 학습 arm에 따라 달라진다.*


Grid Boltzmann의 평균 barrier는 0.03375로 grid max의 0.04380보다 작지만, seed별로는 **1개 개선, 1개 악화, 3개 동률**이다. 평균 차이는 seed 2의 영향이 크다. 국소 2차 차분도 Boltzmann이 더 작지 않아, 현재 데이터에서 일관된 smoothing 효과가 있다고 주장하기 어렵다.

Actor arm의 평균 barrier는 더 낮지만 모든 seed에서 우세한 것은 아니다. 또한 마지막 1,000 updates의 기록된 critic loss 평균은 max 0.1057, Boltzmann 0.0868, actor 0.1462, local 0.1024다. Actor arm은 **K=1 stochastic backup**이므로 loss를 그대로 critic 정확도로 비교할 수 없다.

```text
E[(Q−Y)² | s,a] = (Q−E[Y|s,a])² + Var(Y|s,a)
```

Actor sampling에는 target 분산이 추가된다. 따라서 loss가 높다는 사실만으로 reward-only evaluation 또는 actor extraction이 더 불안정하다고 결론 내리지 않았다.

### 6.3 같은 actor 경로를 여러 critic으로 평가해도 차이는 제한적이다


![그림 9. DDPG→SD2와 TD3→SD3의 같은 actor 경로를 각각 두 critic으로 평가했다. Seed마다 두 endpoint 모두 최종 return 188이다. 세로축은 각 critic의 시작 objective를 뺀 값이다.](figures/09_cross_landscape.png)

*그림 9. DDPG→SD2와 TD3→SD3의 같은 actor 경로를 각각 두 critic으로 평가했다. Seed마다 두 endpoint 모두 최종 return 188이다. 세로축은 각 critic의 시작 objective를 뺀 값이다.*


DDPG→SD2 경로에서 SD2 critic의 barrier가 더 낮은 seed는 **3/5**이며, 곡선은 대체로 비슷하다. TD3→SD3 경로에서는 SD3 쪽 barrier가 **5/5에서 더 높다**. 그러나 이는 두 성공한 policy 사이의 parameter interpolation 결과이므로 SD3의 제어 성능이 더 나쁘다는 뜻도, 원 논문의 local optimum 탈출 설명을 반증한다는 뜻도 아니다.


![그림 10. 동일 OptiQ early→final actor 경로를 다섯 critic으로 평가했다. Seed 1의 barrier 등 경로의 큰 모양이 여러 critic에서 함께 나타난다.](figures/11_optiq_cross.png)

*그림 10. 동일 OptiQ early→final actor 경로를 다섯 critic으로 평가했다. Seed 1의 barrier 등 경로의 큰 모양이 여러 critic에서 함께 나타난다.*


이 비교에서도 OptiQ critic이 유독 더 smooth한 경로를 만든다는 뚜렷한 패턴은 확인되지 않았다. **이번 landscape 실험의 적절한 결론은 “추출·비교는 완료했지만, smoothing 효과의 결정적 재현은 아직 아니다”**다.

Cross-landscape 배열은 기존 float32 계산이며 float64로 다시 추출한 결과가 아니다. 특히 1e-6 threshold로 정의한 flat/decreasing-direction 비율은 계산 정밀도에 민감할 수 있어 핵심 근거로 사용하지 않았다. 위 barrier plot도 정밀도 검증을 끝낸 control 적분과 동일한 수준의 수치 보장을 주장하지 않는다.

## 7. MoveCar return과 188의 의미

| 방법 | S0 | S1 | S2 | S3 | S4 | 평균 return |
|---|---|---|---|---|---|---|
| DDPG | 188.0 | 188.0 | 188.0 | 188.0 | 188.0 | 188.00 |
| SD2 | 188.0 | 188.0 | 188.0 | 188.0 | 188.0 | 188.00 |
| TD3 | 188.0 | 188.0 | 188.0 | 188.0 | 188.0 | 188.00 |
| SD3 | 188.0 | 188.0 | 188.0 | 188.0 | 188.0 | 188.00 |
| OPTIQ | 186.2 | 181.8 | 183.6 | 186.0 | 180.2 | 183.56 |


![그림 11. MoveCar 학습 곡선과 최종 return. 왼쪽 음영은 seed 최소–최대, 오른쪽은 seed별 최종 평가 평균이다. OptiQ는 stochastic policy로 평가했다.](figures/10_movecar.png)

*그림 11. MoveCar 학습 곡선과 최종 return. 왼쪽 음영은 seed 최소–최대, 오른쪽은 seed별 최종 평가 평균이다. OptiQ는 stochastic policy로 평가했다.*


환경은 x∈[0,10], 시작 위치 x=8, action∈[−1,1]이고 다음 상태가 [0.5,1.5]에 있으면 reward 2, [8.5,9.5]에 있으면 reward 1이다. 100-step undiscounted 평가에서 높은 reward 구간에 처음 도달할 수 있는 시점은 7번째 step이므로, 그곳에 머물면 **94×2=188**이다. 네 baseline이 모두 같은 188인 것은 이 평가의 최댓값에 도달했기 때문이다.

OptiQ의 평균은 183.56이다. 이전에 수행한 **seeds 0,1,3만의** 1,000-episode 고정 actor 재평가에서도 이 차이가 남았고, 높은 reward 구간 도달 지연과 일부 seed의 이탈로 나뉘었다. 이 진단을 다섯 seed 결과처럼 합치지는 않았다. Stochastic action이 이 평가에서 return 손실을 만들 수 있지만, 현재의 모든 격차를 τ=0.25의 불가피한 손실로 단정할 수는 없다.

학습 로그의 Q-minus-behavior-return 값도 별도로 저장되어 있으나, behavior policy와 critic backup policy의 차이까지 포함한다. 이를 순수한 critic approximation error 또는 같은 Q에 대한 Boltzmann 적분 오차와 동일시하지 않았다. 또한 이번 학습 시간에는 진단 비용·서로 다른 실행 환경이 섞여 있으므로, 이 기록만으로 OptiQ의 inference latency나 training speed 우위를 평가하지 않는다.

## 8. 논문 주장과 다음 실험의 우선순위

이번 다섯 seed에서 바로 가져갈 수 있는 근거는 다음과 같다.

1. **같은 learned Q에 대한 raw actor backup이 local reweighting보다 정확한 조건이 존재한다.** MoveCar에서는 다섯 seed 모두 확인됐다. 이 결과를 분포·온도·차원 전체로 일반화하지 않는다.
2. **Boltzmann extraction 정확도와 실제 backup 정확도를 따로 측정해야 한다.** 현재 production의 TD action noise가 learned-Q에서 정확도를 크게 낮춘다.
3. **Mode coverage, scalar backup 오차, candidate ESS는 서로 다른 지표다.** 같은 평균 또는 높은 ESS가 정확한 분포를 보장하지 않는 사례를 실제로 확보했다.
4. **현재 OT update에서는 coupling의 marginal 제약이 최종 회귀 target 분포까지 그대로 전달되지 않는다.** 30→300 Sinkhorn 진단으로 유한 반복 오차와 row-argmax의 영향을 분리했다.

반면 **“OptiQ가 복잡한 모든 Boltzmann 분포를 정확히 복구한다”, “SD2·SD3의 smoothing 장점을 OptiQ에서도 재현했다”, “TD noise를 제거하면 return까지 오른다”**는 현재 결과만으로 쓰지 않는다.

다음에 실험을 이어간다면 seed를 더 늘리는 것보다 아래의 **원인별 비교를 동일한 다섯 seed에서 하는 것이 우선**이다. 아래 항목은 이번 분석에서 도출한 제안이며, 새 학습을 자동으로 시작하지 않았다.

| 우선순위 | 비교 | 반드시 같이 볼 지표 |
|---|---|---|
| 1 | 기존 argmax target / marginal을 보존하는 assignment 또는 sampling target; 30/300 Sinkhorn을 구분 | Candidate·coupling·회귀 target·최종 actor 각각의 질량, Q 평균, mode 내부 폭. MSE의 조건부 평균 효과도 점검 |
| 2 | 기본 candidate budget과 증가한 budget, support를 넓히는 proposal | Coverage 초기화 직후부터 1/10/100 updates의 mode 유지, ESS, 가중 후보 편향. 총 Q scale을 고정한 차원 비교 |
| 3 | 기존 TD action noise / noise 제거, 나머지 설정 동일 | Raw와 실제 critic target의 오차, 학습 return, Q 평가의 일관성 |
| 4 | 원 논문의 실패 조건을 먼저 재현한 MoveCar landscape | 나쁜/좋은 policy의 확인, 같은 actor 경로와 같은 probe, 정밀도에 맞는 perturbation 지표 |

당장의 핵심은 **“더 많은 seed로 우열을 확정”하는 것보다 “현재 actor update가 의도한 Boltzmann 분포와 어디에서 갈라지는가”를 해결하는 것**이다. 그 과정이 정리되어야 operator의 성질과 expressive architecture의 장점을 연결하는 논문 주장도 더 명확해진다.

## 부록 A. 13개 frozen 문제 전체 결과

모든 행은 seed 0–4, actor 20k updates, K=50의 RMSE 평균이다. Coverage는 reference-informed warm start이며 기본 성능과 구분했다. Main text는 단일 mode, 비대칭 두 mode, 2D multimodal, separable 고차원을 대표하는 여섯 문제를 상세히 보였고, 전체 문제는 아래에 공개한다.

| Case | 참값 | Default raw | Default TD | Coverage raw | Coverage TD | Local no-IS 범위 |
|---|---|---|---|---|---|---|
| constant | 0.7000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000–0.0000 |
| unimodal | 0.3037 | 0.0342 | 0.1174 | 0.0325 | 0.1199 | 0.0561–0.0561 |
| symmetric_1.3_0.12 | 0.2338 | 0.0689 | 0.2625 | 0.0682 | 0.2610 | 0.0327–0.0330 |
| asymmetric_0.4_0.08 | 0.2855 | 0.0268 | 0.1624 | 0.0261 | 0.1648 | 0.0584–0.0625 |
| asymmetric_0.4_0.16 | 0.1716 | 0.0218 | 0.0434 | 0.0217 | 0.0437 | 0.0290–0.0644 |
| asymmetric_0.8_0.08 | 0.2639 | 0.0408 | 0.2392 | 0.0395 | 0.2423 | 0.0828–0.1512 |
| asymmetric_0.8_0.16 | 0.1216 | 0.0179 | 0.0332 | 0.0178 | 0.0337 | 0.0932–0.1126 |
| asymmetric_1.3_0.08 | 0.2686 | 0.0721 | 0.0996 | 0.0547 | 0.2059 | 0.0769–0.1506 |
| asymmetric_1.3_0.16 | 0.1162 | 0.0325 | 0.0581 | 0.0327 | 0.0582 | 0.0988–0.1401 |
| modes2d_4 | 0.5363 | 0.1224 | 0.6862 | 0.1252 | 0.7146 | 0.0552–0.1847 |
| modes2d_8 | 0.3873 | 0.0412 | 0.3644 | 0.0430 | 0.3715 | 0.0287–0.2278 |
| separable_4 | 0.7061 | 0.3608 | 0.0519 | 0.3611 | 0.0517 | 0.1909–0.2905 |
| separable_8 | 1.4121 | 0.8530 | 0.1313 | 0.8532 | 0.1303 | 0.4998–0.5778 |


Analytic Q는 τ=0.25에서 Gaussian mixture가 Boltzmann 목표 분포가 되도록 구성했다. 2D mode 중심은 반경 0.65 위에 배치했고, mode별 폭과 질량을 달리했다. Separable 문제는 각 좌표의 중심 −0.65/+0.65, 폭 0.12/0.2, mixture weight 0.3/0.7을 사용하고 Q를 좌표별로 합산한다. 정확한 정의는 [problems.py](source/problems.py)에 있다.

## 부록 B. 재현 자료와 파일

- [실험 protocol](protocol.json), [5-seed 범위 확정](scope_5seeds_20260910.json), [그림·완료 수 검증](figure_manifest.json)
- [전체 집계](analysis_summary.json), [후속 분석 JSON](deep_analysis.json), [Frozen 요약 CSV](frozen_summary.csv)
- [Learned-Q 상태별 결과](learned.csv), [Mode history](mode_history.csv), [수정 control](controls.csv), [Cross 경로 지표](cross_paths.csv)
- [Local support 정밀 계산](local_support_refined.csv), [OT 단계별 audit](ot_audit_extended.json)
- [집계 코드](deep_analyze.py), [Support 계산 코드](refine_local_support.py), [그림 생성 코드](build_figures.py), [보고서 생성 코드](build_report.py)
- [OT audit 코드](source/ot_checkpoint_audit.py), [실제 actor update snapshot](source/optiq_dime/algorithm.py), [transport snapshot](source/optiq_dime/transport.py)
- [기존 3-seed 추가 평가](eval_audit_v1/summary.json): seeds 0,1,3 전용. 5-seed 학습 결과와 구분

추정치·분포 표본·landscape 배열과 실행 metadata는 이 폴더의 `runs/`에 있으며, 모든 PNG와 vector PDF는 `figures/`에 있다. 전체 학습 checkpoint를 포함하는 서버 원본은 `login4:/lustre/hobbit9882/OptiQ/outputs/boltzmann_analysis/20260909_v1/`, 기준 repository commit은 `7e2da67d2f0988f6f211b635f7311d32a0c8e8c6`이다. `source/`는 관련 코드를 보존한 것이며 학습을 재실행하려면 원본 repository의 dependencies와 configs도 필요하다. 이번 보고서의 결과 때문에 production OptiQ 코드를 변경하지 않았다.
