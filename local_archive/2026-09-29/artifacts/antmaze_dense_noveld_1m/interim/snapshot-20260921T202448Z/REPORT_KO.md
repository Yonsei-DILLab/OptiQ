# AntMaze 중간 평가 — 2026-09-22 05:24 KST

전체 16개는 아직 완료되지 않았다. v1/v2의 8개를 실행하고 v3/v4 8개는 대기한다.
환경은 dense reward + NovelD, training seed 0이다. 아래는 각 정책의 최신 평가이며
알고리즘별 학습 스텝이 다르므로 동일 예산의 최종 순위로 해석하지 않는다.
각 점은 학습 중 자연 초기 상태 분포에서 수행한 10회 rollout의 성공률이다.

| 환경 | 방법 | 평가 스텝 | native 성공 | 직접 확률 정책 성공 |
|---|---|---:|---:|---:|
| v1 | OptiQ | 750k | 10/10 | 9/10 |
| v1 | SAC | 525k | 0/10 | 0/10 |
| v1 | MFPO | 650k | 2/10 | 2/10 |
| v1 | MEOW | 275k | 1/10 | 2/10 |
| v2 | OptiQ | 325k | 10/10 | 10/10 |
| v2 | SAC | 250k | 10/10 | 10/10 |
| v2 | MFPO | 275k | 10/10 | 10/10 |
| v2 | MEOW | 125k | 10/10 | 9/10 |

v1 OptiQ의 750k native 평가는 위 경로 2회/아래 경로 8회, 직접 확률 정책은
위 경로 1회/아래 경로 8회/실패 1회다. MFPO의 성공 2회는 아래 경로,
MEOW의 직접 확률 정책 성공 2회는 위 경로다. 이는 각각 하나의 학습된 정책의
결과지만, v1은 초기 xy도 달라진다. 동일한 전체 시뮬레이터 상태에서 여러 경로를
선택하는지 여부는 최종 1M 고정 상태 평가로 확인해야 한다.
v2의 최신 성공은 네 방법 모두 G1에만 집중되었다.

Native: OptiQ=random-z mu-only, SAC=tanh(mu), MFPO=Q-best-of-10,
MEOW=prior center. 직접 확률 정책에는 OptiQ conditional sigma가 포함된다.
평가에서 외부 DACER 행동 잡음이나 NovelD reward를 더하지 않는다.
중간 평가에는 요약 JSON만 있으므로 이 그림은 성공률 곡선이다.
최종 1M 평가는 원시 xy 궤적을 따로 저장한다.

학습 source: 19fc37a7e51be2bb54d41ab9a980225ecd9f5ba5.
원자료: evaluation-history.json. 원자료 SHA256은 success-progress-provenance.json.
