# AntMaze v3: 네 방법의 1M 학습 결과

네 방법 모두 학습 seed 0으로 1,000,000회 환경 상호작용을 마쳤다. 각 정책에서 동일한 전체 MuJoCo 초기 상태로 100회 rollout했다. 초기 위치뿐 아니라 자세와 속도까지 방법 간 동일함을 원본 NPZ에서 검증했다.

## 주 비교: 정책에서 직접 stochastic sampling

| 방법 | 성공 / 100 | G1 도달 | G2 도달 | 실패 | 첫 학습 중 성공 step |
|---|---:|---:|---:|---:|---:|
| OptiQ | 99 | 99 | 0 | 1 | 185,390 |
| SAC | 0 | 0 | 0 | 100 | 없음 |
| MFPO | 100 | 0 | 100 | 0 | 252,166 |
| MEOW | 90 | 90 | 0 | 10 | 233,768 |

![같은 초기 상태에서 stochastic rollout 비교](policy-fixed-comparison.png)

OptiQ와 MEOW의 성공 궤적은 모두 `G1/passage-y+4`, MFPO의 성공 궤적은 모두 `G2/passage-y-8`이다. SAC는 1M 학습 전체에서도 성공 기록이 없었다. 성공한 세 방법은 목적지에 도달하지만, 이 최종 평가에서는 각 정책이 한 목표·한 경로에 집중했다. 서로 다른 알고리즘이 다른 목표를 선택한 결과를 합쳐서 한 정책의 다중 모드로 해석하면 안 된다. 단일 seed의 결과이므로 방법 간 일반적인 순위로 단정하지 않는다.

## 평가 모드 구분

주 그림은 모든 방법에서 정책의 직접 확률적 샘플을 사용한다. OptiQ는 random latent와 conditional sigma를 모두 포함한다. SAC도 확률적 Gaussian 정책에서 샘플링하며, MFPO는 Q로 후보를 선별하지 않은 직접 정책 샘플, MEOW는 확률적 prior 샘플을 사용한다. 평가 보상에는 NovelD를 더하지 않으며, 추가 DACER 행동 잡음도 넣지 않는다.

보조 native 평가는 OptiQ의 random-z mu-only, SAC의 `tanh(mu)`, MFPO의 Q-best-of-10, MEOW의 center prior를 사용한다. 성공 횟수는 각각 97, 0, 100, 100회다. 각 모드의 의미가 다르므로 stochastic 주 표와 섞지 않는다. OptiQ zero-z mu-only는 100회 모두 G1에 도달했다. MEOW center-prior 평가는 같은 초기 상태에서 같은 결정론적 경로를 반복한다.

![보조 native 평가 비교](native-fixed-comparison.png)

v3는 기본 reset 자체가 고정 상태이므로 natural-reset 평가도 동일한 시작 조건이다. natural과 fixed 결과를 서로 다른 200개 초기 상태로 합산하지 않았다.

## 학습 조건과 보관

DDiffPG v3 미로에서 다음 위치와 가장 가까운 목표 사이 거리의 음수를 dense reward로 사용하고, 학습에는 승인된 NovelD를 추가했다. 모든 방법에서 batch 256, UTD 1이며, 각 방법의 native 모델·optimizer 설정은 유지했다. OptiQ에는 승인된 DACER 탐색이 포함된다. 따라서 모델 구조와 optimizer까지 동일하게 맞춘 단일 요인 ablation은 아니다.

학습 source: `19fc37a7e51be2bb54d41ab9a980225ecd9f5ba5`. 네 run의 최종 1M replay, 모델·optimizer·RNG·환경 상태, 평가 원본, config와 파일 SHA256을 로컬 보관하고 archive 검증을 통과했다. 재학습은 실행하지 않았다.

원본은 `../../runs/v3-{optiq,sac,mfpo,meow}-s0/`, 그림별 원본 SHA와 초기 상태 검증 기록은 `fixed-state-verification.json`, 재현 스크립트는 `plot_comparison.py`다.

이 보고 시점에 전체 캠페인은 16개 중 11개 결과의 보관·검증을 마쳤다. v2 MEOW와 v4 네 방법은 아직 전체 완료 전이다.
