# AntMaze v1: 완료된 OptiQ와 MFPO의 1M 비교

각 방법은 training seed 0의 단일 정책이다. 두 방법 모두 dense reward + NovelD,
1,000,000 환경 상호작용으로 학습했다. 초기 위치·자세·속도 등을 포함한 전체
초기 시뮬레이터 상태를 네 평가 패널 모두에서 동일하게 맞췄음을 원시 NPZ로 검증했다.
각 평가는 100 rollout이며 실패를 분모에서 제외하지 않는다.

| 방법·평가 | 위 경로 성공 | 아래 경로 성공 | 실패 | 성공률 |
|---|---:|---:|---:|---:|
| OptiQ 직접 정책(random z + sigma) | 71 | 29 | 0 | 100% |
| MFPO 직접 정책 샘플 | 0 | 100 | 0 | 100% |
| OptiQ random z, mu-only | 75 | 21 | 4 | 96% |
| MFPO native Q-best-of-10 | 0 | 99 | 1 | 99% |

이번 두 정책은 직접 샘플링 평가에서 모두 100회 성공했다. 그러나 OptiQ는 위·아래
두 경로를 사용했고, MFPO의 성공은 아래 경로에만 집중됐다. OptiQ는 conditional
sigma를 제거한 평가에서도 두 경로를 사용했다. 이는 이 seed0 정책의 경로 사용
차이이며, 여러 training seed에서의 일반적인 알고리즘 우열이나 모든 상태에서의
action density multimodality를 입증하는 결과는 아니다.

학습 중 최초 성공: OptiQ 162,980 step; MFPO 623,460 step.
최종 자연 초기 상태 분포에서의 직접 정책 평가: OptiQ 99/100(위44/아래55),
MFPO 98/100(위0/아래98). 자연 초기 상태에서는 시작 xy도 변한다.

Batch256, UTD1, 공통 NovelD는 동일하다. 모델/optimizer는 승인된 각 방법의 native
설정이며 OptiQ는 T=.25, DACER=true, mean-init=1; MFPO는 기존 native 설정이다.
따라서 관찰된 차이를 정책 표현력 하나의 단독 인과효과로 해석하지 않는다.
평가에는 외부 DACER 행동 잡음이나 NovelD 보상을 더하지 않는다.

두 최종 리플레이 각각 100만 행, 모델/타깃/optimizer/NovelD/DACER/RNG/환경 상태,
OptiQ 600개·MFPO 400개 원시 rollout을 로컬에 보관했다. 전체 replay dense 보상,
terminal 의미, 파일 SHA256와 상태 digest, 원시 rollout goal/route/return, 평가 초기
상태 동일성 검증을 통과했다. 상세 증거는 ../../runs/<job>/archive-verification.json
및 이 폴더의 fixed-state-verification.json에 있다.

전체 16개 중 2개 완료 시점의 보고다. 나머지14개는 실행/대기 중이며, 이 보고는
전체 실험 완료를 뜻하지 않는다. 학습 소스19fc37a7e51be2bb54d41ab9a980225ecd9f5ba5.

![고정 상태 궤적 비교](fixed-state-comparison.png)
