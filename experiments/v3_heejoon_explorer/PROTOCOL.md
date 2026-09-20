# v3_heejoon_explorer: min–max advantage와 양의 A 가중 회귀

## 변경 목적과 수식

v2의 legacy explorer와 evaluator 분리 구조를 유지하면서 evaluator의 수용 기준을
보수적으로 바꾸고, 양의 advantage가 큰 행동에 더 큰 MSE 가중치를 준다.
사용자가 선택한 가중치는 **양의 advantage 자체 w=A**다.

상태 batch 크기256, 상태당 explorer 행동 한 개, evaluator 기준값용 독립 행동16개를 사용한다.

\[
x_b=\operatorname{clip}(G_{exp}(s_b,z_b),-1,1),\qquad
\tilde a_{bj}\sim\pi_{eval}(\cdot\mid s_b),
\]
\[
A_b=\min_{k=1,2}Q_k^-(s_b,x_b)
-\frac1{16}\sum_{j=1}^{16}\max_{k=1,2}Q_k^-(s_b,\tilde a_{bj}).
\]

**같은 evaluator 행동에서 twin-max를 취한 뒤16개 sample 평균을 낸다.**
행동16개 중 Q가 가장 큰 것을 고르거나, critic별 기대값을 구한 뒤 max하지 않는다.
두 항 모두 target critic을 사용한다. Live critic은 gate에 사용하지 않는다.

\[
m_b=\mathbf1\{A_b>0\},\quad w_b=\max(A_b,0),
\]
\[
L_{eval}=
\frac{\sum_b m_bw_b
\|\operatorname{clip}(G_{eval}(s_b,z_b),-1,1)
-\operatorname{stopgrad}(x_b)\|_2^2}
{\max(\sum_bm_b,1)}.
\]

- **통과한 표본 수로 나누는 v2 정규화를 유지**하며, 분자에 A를 곱한다.
  가중치 합으로 정규화하지 않으므로 advantage의 절대 크기도 gradient scale에 반영된다.
- 지수 가중치, 추가 temperature, weight clipping, advantage normalization은 넣지 않는다.
- Explorer target과 evaluator prediction은 같은 latent를 사용한다.
- Target action, Q, A, mask, weight 및 evaluator baseline은 모두 stop-gradient한다.
  Gradient는 evaluator의 MSE prediction 경로로만 흐른다.
- 통과한 표본이0개면 params뿐 아니라 Adam state와 step도 그대로 둔다.
- Evaluator EMA는 사용하지 않는다. 별도 Adam lr3e-4로 step당 한 번 갱신한다.

TD target은 v2와 동일하다:

\[
a'\sim\pi_{eval}(\cdot\mid s'),\qquad
y=r+0.99(1-d)\min_kQ_k^-(s',a').
\]

즉 **min–max 변경은 evaluator advantage에만 적용**한다. TD의 twin-min과
explorer extraction의 live-twin-mean은 유지한다. Explorer는 기존 rowwise-argmax
OT 학습과 환경 data collection을 그대로 담당한다.

## 설정과 비교 범위

| 항목 | 설정 |
|---|---|
| 환경 / seed | Ant-v4, Humanoid-v4, HalfCheetah-v4 / 0,1,2,3 |
| 예산 | 각1M environment steps, 총12개 추가 run |
| 초기화 | 각 seed의 새 모델. v2 checkpoint에서 이어서 학습하지 않음 |
| Explorer | 기존 a=G(s,z), GELU256×3, N16/M64, rowwise-argmax MSE |
| Teacher / OT | v2와 같은 truncated Gaussian KDE, temperature0.25, beta1, epsilon0.05,30 iterations |
| Critic | scalar twins, replay batch256, target coefficient0.005 |
| 학습 | actor/critic/evaluator Adam3e-4, UTD1, delay1, 5K uniform warmup |
| Evaluation | evaluator, step1 및5K마다10episodes |
| Checkpoint | 50K마다 및 종료/정상 중단 시 full state, evaluator optimizer 포함 |
| W&B | OptiQ/legacy_explorer, 별도 v3 group |
| GPU | Ant0 / Humanoid1 / HalfCheetah2, 최대2run/GPU; GPU3 기존 할당 유지 |

v2 대비 min–max gate와 A 가중치라는 두 요소를 함께 바꾼다. 따라서 성능 차이가
나더라도 두 요소의 개별 효과를 분리한 ablation으로 해석하지 않는다.
같은 Q와 sample을 사용하면 min–max A는 min–min A 이하이지만, 실제 return 개선을
보장하지는 않는다. 초반 critic disagreement로 수용률이0일 수도 있으므로 이를 기록한다.

## 진단과 검증

기존 acceptance fraction/count, A 범위/평균, Q 기준값, MSE 및 gradient norm에
더해 weight sum/max/accepted mean/ESS, evaluator baseline twin gap,
동일 sample에서의 counterfactual min–min acceptance를 기록한다.

실행 전 검증은 독립 toy Q에서 min–max 순서, A=0 reject, 가중 MSE와 analytic
gradient, raw weight scale, explorer/Q gradient 차단, all-reject Adam 불변을 검사한다.
실제 세 환경에서 policy routing, batch256, full checkpoint와 다음 update 재현도
확인한다. Strict gate 때문에 short smoke에서 evaluator step0은 허용하되,
동일 실제 network의 별도 controlled teacher로 weighted Adam update를 검증한다.
각 GPU에서 본 run을 시작하기 전에 GPU smoke도 통과해야 한다.

## Queue와 provenance

현재 진행 중인 run은 유지한다. 대기 우선순위는 각 환경에서 **v2 → v3 → 남은 v1**,
각 버전 내 seed0–3이다. 환경 간 완료 dependency를 만들지 않는다.
원래 v1/v2 numerical source와 W&B ID를 유지하고 v3는 새 immutable source/ID를 쓴다.

`heejoon`에 소스, 설정, 검증, 프로토콜을 먼저 commit/push한 후 검증과 실행을 한다.
Scheduler commit은 numerical source commit과 별도로 기록한다.
Source/runtime은 서버 `/home/heejoonorm/OptiQ/legacy_monge/explorer_v3/COMMIT/`,
중앙 인덱스는 `dildata:/data1/heejoonorm/OptiQ/studies/20260920_v3_heejoon_explorer/`다.
기존 dildata collector가 source/runtime/checkpoint를 회수한다. 키는 기존 승인된
별도0600 파일에서 읽으며 소스·로그·백업에 포함하지 않는다.
