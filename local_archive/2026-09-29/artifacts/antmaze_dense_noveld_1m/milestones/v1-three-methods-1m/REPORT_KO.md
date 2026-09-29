# AntMaze v1: 완료된 세 방법의 1M 비교

OptiQ, SAC, MFPO 각각 training seed0, 1,000,000 환경 상호작용이다.
모두 dense reward + NovelD이며 모델/optimizer는 각 방법의 승인된 native 설정이다.
OptiQ는 T=.25, beta1, DACER=true, mean-init=1. 세 방법의 네트워크·temperature·
행동 탐색 조건까지 동일하게 맞춘 단일요인 ablation은 아니다.

전체 초기 시뮬레이터 상태(위치·자세·속도 포함)가 모든 평가 패널에서 동일함을
원시 NPZ로 대조했다. 평가당100회이며 성공률 분모에 실패도 포함한다.

| 평가 | 방법 | 위 경로 성공 | 아래 경로 성공 | 실패 |
|---|---|---:|---:|---:|
| 직접 확률 정책 | OptiQ | 71 | 29 | 0 |
| 직접 확률 정책 | SAC | 0 | 0 | 100 |
| 직접 확률 정책 | MFPO | 0 | 100 | 0 |
| Native | OptiQ random z, mu-only | 75 | 21 | 4 |
| Native | SAC tanh(mu) | 0 | 0 | 100 |
| Native | MFPO Q-best-of-10 | 0 | 99 | 1 |

이번 단일 seed의 완료 정책에서 OptiQ는 두 우회 경로, MFPO는 아래 경로의 성공이
관찰됐다. SAC은 아래 방향으로 우회하지만 목표 반경 .5에 도달하지 못했다.
SAC의 실패를 성공 경로 한 개에 집중한 mode collapse와 같은 결과로 분류하지 않는다.
OptiQ는 conditional sigma를 제거해도 두 경로를 사용했다. 여러 training seed에서의
일반적인 성능 우열이나 고정 상태의 action density multimodality까지 입증하지 않는다.

SAC은 학습2000 episode 전체에서 목표 도달0회였다. 학습 중 가장 가까웠던 거리는
0.760861로, 성공 반경 .5보다 컸다. 최종 직접 정책의 고정 상태100회 평가에서는
각 episode 최소 거리의 평균이2.42177였다. 자연 초기 상태100회 평가도 성공0회다.
학습 중 최초 성공은 OptiQ162,980step, MFPO623,460step, SAC없음이다.

평가에서 외부 DACER 행동 잡음이나 NovelD reward를 더하지 않는다. 직접 정책 평가의
SAC은 학습된 tanh-Gaussian을 샘플링하고 Native에서는 tanh(mu)를 쓴다.

세 최종 replay 각각100만 행, 모델/타깃/optimizer/entropy/NovelD/RNG/환경 상태와
평가 원시 궤적을 로컬에 보관했다. 모든 replay dense 보상과 terminal, 파일SHA256,
상태digest, 각 rollout의 return/goal/route 및 초기 상태 일치 검증을 통과했다.
실행 오류는 없으며 SAC의0회 성공도 유효한 학습 성능 결과로 보존한다.

전체16개 중3개 완료 시점이다. 나머지13개는 실행/대기 중이다.
학습 source19fc37a7e51be2bb54d41ab9a980225ecd9f5ba5.
증거: ../../runs/<job>/archive-verification.json 및 fixed-state-verification.json.

![고정 상태 비교](fixed-state-comparison.png)
