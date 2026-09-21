# v2_heejoon_explorer: positive-advantage evaluator distillation

## 목적

v1의 evaluator parameter EMA를 제거하고, 현재 evaluator보다 target critic에서
좋게 평가되는 explorer 행동만 선택적으로 학습한다. Explorer의 빠른 legacy
OT 학습과 환경 수집 역할은 유지한다. Gate는 학습된 Q와 유한 MC 평균에 기반하므로
실제 return의 단조 개선이나 critic 오류의 완전한 차단을 가정하지 않는다.

## 정확한 업데이트

Replay state batch \(s_b\), 크기256에서 explorer를 기존 legacy update로 갱신한 뒤,
상태당 explorer 행동 한 개를 새로 뽑는다. Evaluator는 이 update 직전 상태다.

\[
z_b\sim\mathcal N(0,I),\quad
x_b=\operatorname{clip}(G_{\theta_{exp}}(s_b,z_b),-1,1).
\]

각 상태에서 독립적인 evaluator latent 16개로 기준값을 추정한다.

\[
\widehat V^-_{eval}(s_b)=\frac1{16}\sum_{j=1}^{16}
\min_{k=1,2}Q_{\bar\phi_k}\left(s_b,
\operatorname{clip}(G_{\theta_{eval}}(s_b,\tilde z_{bj}),-1,1)\right),
\]
\[
A_b=\min_{k=1,2}Q_{\bar\phi_k}(s_b,x_b)-\widehat V^-_{eval}(s_b),
\qquad m_b=\mathbf1\{A_b>0\}.
\]

**Min을 취한 뒤 sample 평균**을 낸다. Live critic, max critic, 평균 critic이나
\(\min_k \mathbb E Q_k\)로 바꾸지 않는다. 두 항 모두 동일 target critic을 사용한다.
Parent update 순서를 유지하므로 critic gradient update와 target critic Polyak
update가 끝난 뒤의 target parameters를 gate에 사용한다.

Evaluator 목적함수는 실제 normalized action-space MSE다.

\[
L_{eval}=\frac{\sum_b m_b\left\|
\operatorname{clip}(G_{\theta_{eval}}(s_b,z_b),-1,1)
-\operatorname{stopgrad}(x_b)\right\|_2^2}
{\max(\sum_b m_b,1)}.
\]

- Explorer와 evaluator 예측은 **같은 latent \(z_b\)**를 사용한다. 두 policy의 임의
  샘플을 무관하게 짝지어 여러 mode를 평균내도록 학습시키지 않는다.
- Target 행동, advantage, mask, evaluator 기준 Q를 모두 stop-gradient한다.
  **Gradient는 evaluator의 MSE prediction에만** 흐른다.
- Positive advantage는 binary filter다. Advantage 크기로 가중하지 않고,
  별도 margin이나 confidence threshold도 넣지 않는다. A=0은 제외한다.
- 통과한 sample 수로 loss를 정규화한다. 한 batch에 통과 sample이0개이면
  **optimizer 호출 자체를 건너뛰고 params, Adam momentum, optimizer step 모두 유지**한다.
- Evaluator는 gradient update 한 번/환경 step, Adam lr3e-4, (0.9,0.999).
  Explorer와 같은 architecture와 초기 params로 시작하지만 optimizer state는 별도다.
- **Evaluator EMA는 전혀 없다.** policy_tau=0, parent EMA hook은 identity.
- Gate sampling RNG는 기존 model key에서 fold_in으로 분리하며, 기존 explorer/critic
  RNG stream을 추가로 소비하지 않는다. Evaluation RNG도 기존처럼 별도로 유지한다.

TD target은 v1과 동일하게 evaluator가 만든 다음 행동을 사용한다.

\[
a'_b\sim\pi_{eval}(\cdot\mid s'_b),\qquad
y_b=\operatorname{stopgrad}\left[r_b+0.99(1-d_b)
\min_k Q_{\bar\phi_k}(s'_b,a'_b)\right].
\]

| 역할 | v2 사용 정책/값 |
|---|---|
| 환경 data collection | Explorer (5K uniform warmup 유지) |
| OT source / KDE / candidate / legacy actor update | Explorer |
| Explorer extraction Q | Live twin critic mean, 기존 그대로 |
| Evaluator acceptance gate | Target twin critic min과 evaluator MC 평균 |
| Evaluator 학습 | Positive-A action-space MSE, same latent |
| Evaluation / TD next action | Evaluator |

## 실행 설정 — v1과 같은 12 runs

| 항목 | 설정 |
|---|---|
| 환경 | Ant-v4, Humanoid-v4, HalfCheetah-v4 |
| Seed | 0,1,2,3 |
| 길이 | 각각1M environment steps |
| Actor 및 critic | 기존 legacy GELU256×3; scalar twin critic |
| Explorer N/M | 16/64, random4 per center, anchor 없음 |
| Explorer KDE | Truncated Gaussian std0.2, local clip0.5 |
| Temperature / density beta | 0.25 /1 |
| Explorer OT | Mean-normalized squared cost, epsilon0.05,30 iterations |
| Explorer loss | 기존 raw actor output 대 rowwise-argmax target MSE |
| Critic batch / actor state batch | 256/256 |
| UTD / actor delay | 1/1 |
| Optimizer | Actor, evaluator, critic Adam3e-4 |
| Target critic coefficient | 0.005 |
| Evaluation | Step1 및5K마다 evaluator stochastic policy10episodes |
| Checkpoint | 50K마다 및 종료/정상 중단 시 full resume |
| W&B | OptiQ/legacy_explorer, 별도 v2 group/algorithm_name |
| GPU | Ant0, Humanoid1, HalfCheetah2; GPU3 GMM40 유지 |

수치 precision도 기존 v1의 GPU 기본 설정을 유지한다. 별도 GMM40의 highest
precision 환경변수를 가져오지 않는다. 추가 TD noise, entropy backup, KL penalty,
새 loss weighting 등은 넣지 않는다.

## 진단 및 검증

W&B에 evaluator acceptance fraction/count, advantage 범위/평균, accepted advantage,
explorer target Q, evaluator baseline Q, MC std/standard error, evaluator MSE/gradient
norm, 실제 action 변화와 clipping boundary fraction, optimizer steps를 기록한다.
Gate 지표는 기존 diagnostic interval의 batch snapshot이며 전체 구간 평균으로
해석하지 않는다. Evaluator EMA 적용 지표는0이다.

실행 전 검증:
1. 독립적으로 풀 수 있는 Q에서 A와 A>0 mask, min-before-mean 순서 확인.
2. Target critic 대신 live critic을 바꿔도 gate가 바뀌지 않는지 확인.
3. 전체 계산 graph에서 explorer와 target critic gradient0, evaluator MSE gradient
   nonzero 확인. TD stop-gradient와 terminal mask는 v1 검증을 재사용.
4. Adam momentum이 이미 존재하는 상태에서도 전부 reject되면 전체 state 불변.
5. 실제 세 MuJoCo에서 policy routing, no-EMA, evaluator optimizer 학습,
   production batch256 shape를 확인.
6. Full checkpoint에 explorer/evaluator/critic/target/optimizer/replay/RNG/simulator를
   저장하고 다음 행동, simulator 및 다음 training update가 정확히 복구되는지 확인.

현재 GPU를 뺏지 않도록 위 사전 검증은 CPU에서 수행한다. 각 GPU가 비면 동일
source의 짧은 GPU 검증을 해당 환경에서 수행하고 통과해야 본1M run을 시작한다.
GPU 검증용 결과는 validation_gpu 폴더이며 본 seed 결과와 섞지 않는다.

## Queue와 보관

`heejoon` branch에 exact code/config/protocol/launch를 commit 및 push하고,
immutable snapshot의 source manifest에 full SHA와 파일별 SHA256을 기록한다.

서버: heejoonorm@31.148.50.247:11717.
현재 v1의 환경별 worker lock이 해제되고 해당 GPU가 실제 비었을 때 v2를 시작한다.
이는 **GPU resource 대기**이며 다른 환경 성공 여부에 의존하지 않는다. v1 worker가
실패/종료하여 GPU를 반납해도 v2는 eligible하다. 기존 실행/대기 v1 및 GMM40는
중단하거나 재배치하지 않는다. v2 각 환경은 seeds0–3을 순서대로 실행한다.

Source/runtime: `/home/heejoonorm/OptiQ/legacy_monge/explorer_v2/COMMIT/`.
기존120초 dildata collector가 이 하위 폴더도 회수한다:
`/data1/heejoonorm/OptiQ/studies/20260919_legacy_monge/remote_3114850247/explorer_v2/COMMIT/`.
별도 index: `/data1/heejoonorm/OptiQ/studies/20260920_v2_heejoon_explorer/`.
W&B key는 기존 승인된 별도0600 파일에서 읽으며 source/log/backup에 포함하지 않는다.
Run마다 고정된 W&B ID를 사용하고, 같은 source의 checkpoint resume는 동일 ID를 유지한다.
