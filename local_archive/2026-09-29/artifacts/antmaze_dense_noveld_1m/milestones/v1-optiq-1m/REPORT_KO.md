# AntMaze v1 OptiQ: 1M 완료 결과

Dense reward + NovelD, training seed 0, OptiQ T=.25/beta=1/DACER=true/mean-init=1.
1,000,000 환경 상호작용, warmup 5,000, learner/NovelD update 각각 995,000.
다른 15개 실험은 아직 완료되지 않았으므로 방법 간 최종 비교 결과는 아니다.

## 동일한 전체 초기 상태: 각 100 rollout

| 평가 | 성공 | 위 경로 | 아래 경로 | 실패 |
|---|---:|---:|---:|---:|
| random z, mu-only | 96 | 75 | 21 | 4 |
| random z + conditional sigma | 100 | 71 | 29 | 0 |
| zero z, mu-only | 100 | 100 | 0 | 0 |

세 평가 모드 전체 300회에서 위치·자세·속도 등을 포함한 전체 초기 시뮬레이터 상태가
동일함을 원시 NPZ로 검증했다. random z는 매 행동마다 새로 샘플링한다.
외부 DACER 행동 잡음은 평가에 추가하지 않으며, 평가 reward는 dense 환경 보상만 사용한다.

이 학습된 단일 정책은 conditional sigma 없이도 두 우회 경로를 사용했다.
zero z에서는 위 경로에만 집중했다. 따라서 초기 위치 차이나 conditional sigma만으로
두 경로가 생긴 결과는 아니다. 단일 training seed 결과이며, 다른 초기 상태·seed에서의
일반화나 고정 상태의 action density 자체가 여러 봉우리라는 주장까지 입증하지 않는다.

## 자연 초기 상태 분포: 각 100 rollout

| 평가 | 성공 | 위 경로 | 아래 경로 | 실패 |
|---|---:|---:|---:|---:|
| random z, mu-only | 96 | 46 | 50 | 4 |
| random z + conditional sigma | 99 | 44 | 55 | 1 |
| zero z, mu-only | 94 | 46 | 48 | 6 |

자연 초기 상태는 xy가 달라지므로 이 표의 두 경로만으로 정책 무작위성의 효과를 분리할 수 없다.
학습 중 첫 성공은 162,980 step, 총 성공 episode는 4,377/5,145였다.

## 저장 및 검증

최종 리플레이 1,000,000행, model/target/optimizer/NovelD/DACER/RNG/환경 전체 상태,
600개 최종 원시 rollout과 41개 시점의 각 평가 history를 로컬에 보관했다.
SHA256, full-state digest, 모든 replay reward와 terminal, 각 rollout의 goal/route/return,
fixed-state 일치 검증을 통과했다. 학습 frozen source는
19fc37a7e51be2bb54d41ab9a980225ecd9f5ba5이며 변경하지 않았다.

최초 로컬 검증은 Python3.14 + NumPy1.26 런타임의 배열 연산 오류로 실패했다.
학습/저장 자료나 verifier를 수정하지 않고 Python3.12.14 + NumPy2.3.5에서
동일 검증을 통과했다. 상세 환경은 ../../REPORT_RUNTIME.json,
아카이브 증거는 ../../runs/v1-optiq-s0/archive-verification.json에 있다.

![동일 상태 궤적](fixed-state-trajectories.png)
