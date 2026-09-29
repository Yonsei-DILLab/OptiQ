# AntMaze dense + NovelD

1M 완료 12/16. 각 환경·방법당 학습 seed0 하나입니다.

주 경로 그림은 정책에서 직접 샘플링한 stochastic rollout입니다. OptiQ는 random latent와 conditional sigma를 포함하며, 외부 DACER 잡음은 제외합니다. Native 성능 평가는 별도 그림·표로 구분합니다.

| 미로 | 방법 | 직접샘플 성공률 | Native 성공률 | 성공 경로 수 | 최다 경로 비율 | 학습 중 최초 성공 |
|---|---|---:|---:|---:|---:|---:|
| v1 | optiq | 100.0% | 96.0% | 2 | 71.0% | 162980 |
| v1 | sac | 0.0% | 0.0% | 0 | N/A | 없음 |
| v1 | meow | 95.0% | 100.0% | 1 | 100.0% | 202415 |
| v1 | mfpo | 100.0% | 99.0% | 1 | 100.0% | 623460 |
| v2 | optiq | 100.0% | 100.0% | 1 | 100.0% | 26829 |
| v2 | sac | 100.0% | 100.0% | 1 | 100.0% | 42199 |
| v2 | mfpo | 100.0% | 100.0% | 1 | 100.0% | 64623 |
| v3 | optiq | 99.0% | 97.0% | 1 | 100.0% | 185390 |
| v3 | sac | 0.0% | 0.0% | 0 | N/A | 없음 |
| v3 | meow | 90.0% | 100.0% | 1 | 100.0% | 233768 |
| v3 | mfpo | 100.0% | 100.0% | 1 | 100.0% | 252166 |
| v4 | optiq | 99.0% | 94.0% | 1 | 100.0% | 339491 |

성공률은 동일한 전체 초기 상태에서 최종 정책을100회 평가한 값입니다. 실패 episode도 분모에 포함합니다. 성공이 없으면 경로 수는0이며, 이를 mode collapse의 증거로 단정하지 않습니다.
Native: SAC tanh(mu), MFPO Q-best-of10, MEOW prior-center, OptiQ random-z mu-only. v2/v3/v4의 원래 시작 상태가 고정되어 natural/fixed를 독립된200회처럼 합산하지 않습니다.
v2는 주로 두 목표 선택을 비교합니다. 경로 범주는 미리 정의한 통로 횡단 기준이며 모든 homotopy class 또는 action distribution의 multimodality 증명은 아닙니다.
각 정책은 단일 training seed이므로 seed 간 평균·표준편차나 알고리즘 일반 성능 우열을 주장하지 않습니다. MaxEntDP의 dense 보상 설명에 DDiffPG NovelD를 추가한 사용자 지정 조건입니다.
모델/최적화 설정은 각 방법의 기존 기본값을 유지했습니다. OptiQ/SAC는256x2, MFPO는256x3, MEOW는 native flow 구조입니다. NovelD는 공통이며 OptiQ는 승인된 DACER 행동 탐색도 사용합니다.
![직접 정책 궤적](trajectories-policy-fixed.png)
![기본 평가 궤적](trajectories-native-fixed.png)
