# 현재 검증 중인 OptiQ v2 알고리즘

대상은 `mujoco_v2_checked`로 시작한 Humanoid 4시드 비교 실험이다.
실행 시작 코드 `8cb4f23`과 각 실행의 `config.json`을 기준으로 확인했다.
이 문서는 진행 중인 후보의 설명이며, 최종 성능 검증은 아직 끝나지 않았다.

## 실행 정책과 teacher 분포

환경에서 실행하는 정규화 행동은 다음과 같다.

\[
(\mu_\theta(s,z),\log\sigma_\theta(s,z))=G_\theta(s,z),\qquad
a=\tanh(\mu_\theta+\sigma_\theta\odot\epsilon),\qquad z,\epsilon\sim\mathcal N(0,I).
\]

한 번의 네트워크 forward로 조건부 Gaussian의 평균과 표준편차를 얻는다.
그 뒤 Gaussian noise와 tanh를 적용하고 실제 환경 행동 범위로 변환한다.
Humanoid는 행동 17차원이며, 정책 밀도는 정규화 행동 좌표에서 계산한다.
log sigma 범위는 [-5, 1], 초기 sigma는 0.5다.
5K warmup 뒤에는 이 정책을 그대로 실행하며 추가 uniform 행동 교체는 없다.

Actor 업데이트에서는 상태마다 latent 16개와 각각의 Gaussian noise를 뽑아
student 행동 16개를 만든다. 같은 latent에서 얻은 평균·표준편차로 teacher
proposal을 먼저 정의한다.

\[
q_F(u\mid s)=\frac1{16}\sum_{i=1}^{16}
\mathcal N\!\left(u;\mu_i,\operatorname{diag}(\tilde\sigma_i^2)\right),\qquad
\tilde\sigma_i=\max(\sigma_i,0.05).
\]

이후 그 혼합분포에서 후보 64개를 뽑고 tanh를 적용한다. `exact` 모드이므로
각 후보의 component를 독립적으로 선택한다. R=4는 총 후보 수 16×4를 정하며,
각 component에서 반드시 4개씩 뽑는 stratified 방식은 아니다. Anchor는 없다.
Student는 실제 정책에서 나오고, sigma floor는 teacher proposal에만 적용한다.

현재 teacher는 **학습된 조건부 Gaussian의 혼합**이다. 초기 v2처럼 실현된
student u를 중심으로 고정 bandwidth 0.8의 KDE를 만드는 경로와 구별된다.
후보를 먼저 뽑아 그 후보들에 KDE를 다시 맞추는 방식도 사용하지 않는다.
샘플링과 밀도 계산 모두 같은 teacher 평균·표준편차를 사용한다.

## Importance correction과 OT distillation

후보 행동의 가중치는 다음과 같다.

\[
w_j=\operatorname{softmax}_j\left(\frac{(Q_1+Q_2)(s,a_j)}{2T}
-\beta\log q_F(a_j\mid s)\right),\qquad T=0.1,\quad\beta=1.
\]

밀도는 행동 차원을 먼저 합산한 **joint Gaussian density**들을 혼합하고,
tanh Jacobian도 반영한다. 생성 때의 pre-tanh u를 보존하여 계산한다.
T와 beta는 고정이며, ESS가 낮아져도 자동으로 바뀌지 않는다.

OT의 행은 student 16개에 각각 질량 1/16, 열은 teacher 후보 64개에 질량 w를
갖는다. 비용은 정규화 행동 좌표에서의 제곱거리 합이다.
비용 행렬을 평균으로 나누는 추가 정규화는 꺼져 있다.
Sinkhorn epsilon은 **0.25**, 반복 횟수는 **100**이다.
이 epsilon은 OT 결합의 regularization이며, Boltzmann 목표 온도 T=0.1과 다르다.

Actor는 각 OT 행의 전체 분포에 대한 조건부 Gaussian negative log likelihood를
최소화한다. Teacher u의 가중 평균뿐 아니라 분산도 반영하여 mu와 sigma를
학습한다. Q, teacher, importance weight, OT 결합은 stop-gradient 처리한다.
Actor에는 이 지도학습 gradient를 사용하며, Q의 행동 미분이나 별도의 actor
entropy-gradient 손실은 추가하지 않는다.

## Soft critic backup과 엔트로피

다음 상태에서는 현재 actor로 실제 정책 행동을 뽑는다. 그 행동을 생성한
component 1개와 추가 component 15개로 밀도를 추정한다. 이 계산은 teacher의
sigma floor를 쓰지 않는다.

\[
y=r+\gamma(1-d)\left[\min_{k=1,2}Q_{\bar\phi,k}(s',a')
-T\log\hat\pi_\theta(a'\mid s')\right].
\]

Critic은 scalar twin-Q의 TD MSE를 학습한다. TD 행동에 추가 smoothing noise는
없다. T=0.1은 teacher 가중치와 soft backup에 공통으로 쓰이며, SAC식 자동
온도 조정은 하지 않는다. IDAC entropy lower bound는 component와 행동에 대한
**기댓값에서의 bound**다. 개별 표본이 항상 참 엔트로피보다 작다는 뜻은 아니다.

## 업데이트 승인 검사와 이론의 범위

제안된 actor 업데이트는 독립 replay 상태 32개에서 검사한다. 상태마다 행동
8회, 밀도 component 16개를 사용한다. 같은 고정된 minimum twin-Q 아래에서
새 정책의 lower soft score와 기존 정책의 upper soft score 차이를 추정한다.
유한한 평균 차이가 양수이면 승인한다. 현재 추가 표준오차 margin은 0이다.
거절하면 actor 파라미터와 Adam 상태·step을 모두 복원한다. Critic 학습은 계속된다.

정확한 기존 정책 Q에 대해 모든 관련 상태에서

\[
\mathbb E_{\pi_{new}}Q^{\pi_{old}}+T H(\pi_{new})
\ge \mathbb E_{\pi_{old}}Q^{\pi_{old}}+T H(\pi_{old})
\]

가 성립하면, 명시된 Bellman 가정 아래 soft policy improvement를 보장할 수 있다.
정확한 Boltzmann 추출과 분포를 보존하는 정확한 distillation도 이 조건을 만족한다.
현재 replay 평균의 유한 표본 검사는 그 **상태별 정확한 조건의 인증이 아니다**.
W2 감소나 NLL 감소만으로 위 부등식이 따라오지도 않는다.
자세한 가정과 반례는 [이론 문서](V2_SOFT_POLICY_IMPROVEMENT.md)에 있다.
실제 환경 보상 우위와 장기 안정성은 별도의 실험으로 검증한다.

## 공통 설정과 기준선 차이

두 방법 모두 Humanoid-v4, seed 0/1/2/3, actor·critic 256×3 GELU, scalar twin-Q,
Adam 3e-4와 betas=(0.9,0.999), gradient norm clipping=2를 사용한다.
Batch=256, replay=1M, warmup=5K, UTD=1, gamma=0.99, critic tau=0.005,
policy tau=1이며 BN/LN/dropout은 없다. 각 실행은 1M cap,
5K마다 stochastic 평가 10회, diagnostic 5K, checkpoint 50K다.

| 항목 | 현재 v2 후보 | OptiQ 기준선 |
|---|---|---|
| 실행 정책 | tanh 조건부 Gaussian의 implicit mixture | clip(g(s,z)) |
| Teacher 후보 | 64, anchor 없음, IID mixture | 80, 중심마다 무작위 4개+anchor 1개 |
| Teacher 온도 / beta | 0.1 / 1 | 0.25 / 1 |
| OT 비용 평균 정규화 | 끔 | 켬 |
| Sinkhorn epsilon / 횟수 | 0.25 / 100 | 0.05 / 30 |
| Actor distillation | 전체 OT 조건부 NLL | argmax pointwise MSE |
| TD entropy | IDAC 추정, T=0.1 | 없음 |
| TD smoothing sigma / clip | 0 / 0 | 0.2 / 0.5 |
| Soft-score 승인 검사 | 사용 | 사용하지 않음 |

따라서 두 방법의 비교는 전체 알고리즘 비교다. 특정 요소 하나의 효과로
성능 차이를 모두 해석할 수 없으며, 별도 진단·screen 결과를 함께 봐야 한다.

현재 경로에서는 남아 있는 `proposal_clip=0.5`로 teacher를 잘라내지 않는다.
`transport_target_mode=argmax`의 선택점은 진단 등에 계산되지만 NLL 손실은
전체 OT 행을 쓴다. `source_reference=uniform_action`은 현재 actor 업데이트가
읽는 키가 아니며, student가 uniform 행동이라는 뜻이 아니다.
`v_min/v_max`는 scalar critic의 Q clipping 범위로 적용되지 않는다.

```bash
OPTIQ_CONFIG=mujoco_v2_checked scripts/run_v2.sh 0 --check
```

이는 설정 확인 명령이다. 원형 v2와 현재 후보의 구분 및 실행 안내는
[README](../README.md), 실험 근거는 [조사 기록](V2_IMPROVEMENT.md)을 따른다.
구현 근거: [actor·critic 업데이트](../optiq_dime/algorithm.py),
[정책](../optiq_dime/policy.py), [밀도·proposal](../optiq_dime/semi_implicit.py),
[NLL](../optiq_dime/distillation.py), [승인 검사](../optiq_dime/soft_improvement.py).
