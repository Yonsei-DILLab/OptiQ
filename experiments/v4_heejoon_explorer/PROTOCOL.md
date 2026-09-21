# v4_heejoon_explorer: v2 min–mean + exp(A)

## 변경과 정확한 수식

v2의 target twin-min / evaluator MC-mean advantage를 그대로 사용한다.
V3와 달리 evaluator 한 행동의 twin-max가 아니라, **독립적으로 뽑은 evaluator
행동 16개 각각의 twin-min을 구한 다음 행동 표본 간 평균**을 낸다.

\[
x_b=\operatorname{clip}(G_{exp}(s_b,z_b),-1,1),\qquad
\tilde a_{bk}=\operatorname{clip}(G_{eval}(s_b,\tilde z_{bk}),-1,1),
\]
\[
A_b=\min_{i=1,2}Q_i^-(s_b,x_b)
-\frac1{16}\sum_{k=1}^{16}\min_{i=1,2}Q_i^-(s_b,\tilde a_{bk}).
\]

Explorer latent z와 baseline의 16개 latent는 독립이다. 아래 MSE prediction에는
explorer target을 생성한 **동일한 z**를 사용한다.

\[
m_b=\mathbf1\{A_b>0\},\qquad w_b=\exp(A_b),
\]
\[
L_{eval}=\frac{\sum_bm_bw_b\|
\operatorname{clip}(G_{eval}(s_b,z_b),-1,1)-\operatorname{stopgrad}(x_b)
\|_2^2}{\max(\sum_bm_b,1)}.
\]

- 분모는 v2/v3와 같은 accepted count이다. Weight sum으로 정규화하지 않는다.
- 지수에 별도의 temperature를 넣거나 A를 clip/normalize하지 않는다.
  Explorer Boltzmann temperature 0.25와 이 지수 가중치는 별개다.
- 거절한 표본의 구현상 weight는 0이다. Target/action/Q/A/mask/weight는 stop-gradient.
- Evaluator는 별도 Adam 3e-4로 학습한다. EMA 없이, 전부 거절한 batch는 Adam
  momentum과 step도 그대로 둔다.
- Legacy rowwise-argmax explorer, explorer의 환경 collection 및 OT source/KDE 중심,
  evaluator의 evaluation 및 TD action 역할을 유지한다.
- TD target은 reward-only target twin-min, extraction Q는 live twin-mean 그대로다.

## 실험 설정

| 항목 | 설정 |
|---|---|
| 환경 | Ant-v4 / Humanoid-v4 / HalfCheetah-v4 |
| Seed / 예산 | 0,1,2,3 / 각 1M environment steps, 총 12개 새 run |
| 초기화 | Random initialization, v2/v3 checkpoint를 사용하지 않음 |
| Explorer | a=G(s,z), GELU 256×3, N16 / M64, rowwise-argmax MSE |
| Proposal / OT | Truncated Gaussian KDE, beta1, epsilon0.05, 30 iterations |
| Boltzmann temperature | 0.25 |
| Critic / batch | Scalar twins, replay256, gamma0.99, target tau0.005 |
| Optimizer | Actor/critic/evaluator Adam3e-4, UTD1, delay1, 5K uniform warmup |
| Evaluation | Evaluator, step1 및5K마다10episodes |
| Checkpoint | Full-state 50K마다 및 종료/중단, evaluator Adam 포함 |
| W&B | OptiQ/legacy_explorer, v4_heejoon_explorer_legacy_N16_M64_T025_MinMean_ExpA_K16 |
| GPU | Ant0 / Humanoid1 / HalfCheetah2, 최대2개/GPU; GPU3 별도 유지 |

V2와는 exp(A) 가중치만 다르다. V3와는 가중치와 advantage baseline을 함께 변경한다.
따라서 v3 대비 차이를 가중치 하나의 효과로 해석하지 않는다.

## 진단과 검증

Acceptance fraction/count, A mean/min/max, baseline MC std/SE, evaluator loss와
parameter gradient norm, weight sum/max/accepted mean/ESS 및 유한성 여부를 기록한다.
ESS만 log-space rescaling을 사용하며, 실제 loss는 raw exp(A) 그대로다.

실행 전 toy 검증: 동일 RNG에서 v2 teacher와 일치, min 후 행동 평균 순서,
A=0 거절, raw exp(A)의 analytic loss/gradient와 scale, Q/explorer gradient 차단,
all-reject Adam 불변을 확인한다. 세 실제 환경에서는 routing, batch256,
full-state checkpoint 및 다음 action/simulator/training update 재현을 검사한다.
CPU 검증 후, 각 환경 GPU slot이 비면 GPU 검증을 통과한 뒤 학습을 시작한다.
W&B는 실제 committed history의 원격 조회로 온라인 기록을 확인하며 300초까지 재시도한다.

## 실행과 보관

Source/config/protocol/runner/validation을 heejoon에 commit/push한 후 검증/실행한다.
Numerical source와 scheduler SHA를 별도로 기록하고 immutable manifest로 검증한다.
기존 학습은 유지하며, 대기 순서는 v2 → v3 → v4 → 남은 v1이다. 환경 간 dependency 없음.

서버: heejoonorm@31.148.50.247:11717,
/home/heejoonorm/OptiQ/legacy_monge/explorer_v4/COMMIT/.
중앙 인덱스: dildata:/data1/heejoonorm/OptiQ/studies/20260920_v4_heejoon_explorer/.
기존 collector가 source/runtime/checkpoint를 회수한다. 인증 정보는 Git/백업에서 제외한다.
