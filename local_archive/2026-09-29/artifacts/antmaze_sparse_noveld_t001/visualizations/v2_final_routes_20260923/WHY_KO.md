# v2가 한 성공 경로에 집중한 이유 — 관측과 가설

대상은 sparse+NovelD .01, OptiQ T=.01, seed0, 3,008,256 interactions다. 학습 설정을 변경하거나 새 실험을 실행하지 않았다.

## 확인한 사실

- G2=(8,0)는 시작점에서 오른쪽으로 직진하는 경로이며 성공 보상은10이다. G1=(-8,8)은 왼쪽으로 간 뒤 아래로 꺾는 경로이며 보상은20이다. 원본 맵의 셀 중심선을 따르면 각각8m/16m다. 이는 실제 로봇 궤적 길이나 연속공간의 정확한 최단거리 측정값은 아니다.
- 첫 학습 성공은2,302,717 interactions에서 G2였다. 2.5M까지1회, 2.5M–2.75M에10회, 2.75M–3,008,256에616회였다. 총627회 모두 G2이며 G1 성공 경험은0회다.
- 최종 직접정책 샘플링(random z+conditional sigma)은 랜덤 시작82/100, 동일 full-state 시작86/100 성공했다. 모두 중앙→오른쪽 G2 경로이며 성공 궤적에서 왼쪽 통로나 오른쪽 수직 가지로 진입한 경우는0회다.
- 원본 sparse reward도 경로들을 동등하게 만들지는 않는다. 현재 gamma=.99에서 순수 종료보상만 비교하면 2배 보상 이점은 약68.97 environment steps의 추가 지연으로 상쇄된다. 이는 NovelD·실패확률을 제외한 예시 계산이며, G1 완주 데이터가 없어 실제 어느 경로의 기대 return이 더 큰지는 입증하지 못했다.
- 이전 forward 감사에서 v2의 2.00M→2.75M Q 후보 표준편차는 .001129→.003476, Q-only ESS는63.14→55.12였다. T는그대로 .01이었다. Q 구별 신호가 커졌다는 관측이며, 그 자체가 경로별 Q 차이나 온도의 인과 효과를 측정한 것은 아니다.

## 가장 일관된 해석

더 쉽게 발견한 G2의 성공 경험이 쌓이면서 그쪽의 가치와 정책이 강화되고, 다시 G2 데이터를 더 많이 수집하는 과정과 일치한다. G1 보상을 경험한 적이 없으므로 높은 보상20의 장점을 그 경로에 연결해 학습하기 어려웠다. 두 성공 경로를 확보한 뒤 하나를 잃었다는 증거는 없다.

현재 Boltzmann actor의 T=.01은 학습된 Q 차이에 민감할 수 있다. 하지만 단일 seed의 시간 경과 관측만으로 T가 원인이라고 확정할 수 없다. NovelD는 RND의 상태 예측 오차에 기초한 보너스이며 목표별 도달 비율을 균등하게 만드는 보상은 아니다. 위치 인코딩 이외 자세·속도도 RND 입력에 포함된다. GMM 표현력·행동 분산·DACER 행동 entropy는 여러 단계에 걸친 목표 선택과 성공 경로 다양성을 보장하지 않는다.

이번 v2 결과만으로 critic LR/tau 또는 sigma를 원인에서 완전히 배제할 수는 없다. 그러나 동일 설정으로 G2 학습과82% 성공이 가능했으므로, LR만 올리면 다중 경로가 해결된다는 근거는 없다. 먼저 G1 방향으로 진입·전진한 구간의 intrinsic return과 critic 예측을 대조해야 한다. 보상 신호가 부족한지, 그 신호의 가치 학습이 부족한지 구분하는 것이 다음 진단이다.

## 원자료

- `raw/training-successes.json`: 전체627회 goal 도달 기록.
- `raw/config.json`: 학습 source8bb0c50, gamma/temperature/NovelD 등 설정.
- `raw/evaluations/0003008256/`: native/policy/zero_z × natural/fixed의600개 원시 rollout.
- `remote-sha256.json`, `metrics.json`, `plot.py`: 원격15개 파일 SHA256 대조, 평가 검증, 경로 분류·그림 재현.
- 보상 구현: `antmaze/ddiffpg/env/d4rl/locomotion/goal_reaching_env.py`; 지도: `maze_env.py`; NovelD 구현: `antmaze/ddiffpg/utils/intrinsic.py`.
