# AntMaze v3: 완료된 세 방법의 1M 비교

학습 seed 0, 방법별 1,000,000 환경 상호작용. OptiQ·SAC·MFPO의 최종 원본 rollout, 모델·optimizer·RNG·환경 상태와 1M transition replay를 로컬 보관하고 검증했다. v3 MEOW는 아직 학습 중이므로 이 비교에서 제외한다.

## 동일 초기 상태의 직접 stochastic 정책 평가

각 방법에서 100회 rollout했다. 세 방법 및 모든 에피소드의 전체 초기 simulator state가 정확히 같은지 원본 배열로 확인했다. 서로 다른 학습 seed의 평균이 아니다.

| 방법 | 성공 | G1 도달 | G2 도달 | 평균 환경 return |
|---|---:|---:|---:|---:|
| OptiQ | 99/100 | 99 | 0 | -990.03 |
| SAC | 0/100 | 0 | 0 | -4129.22 |
| MFPO | 100/100 | 0 | 100 | -710.32 |

OptiQ는 G1의 `passage-y+4` 경로로 99회 성공했고, MFPO는 G2의 `passage-y-8` 경로로 100회 성공했다. SAC는 100회 모두 실패했다. 성공한 두 정책이 선택한 목표는 서로 다르지만, 각 정책은 하나의 목표·성공 경로에 집중됐다. 서로 다른 정책의 궤적을 합쳐 한 정책의 multimodality로 해석하면 안 된다.

![직접 stochastic 정책 비교](fixed-state-comparison.png)

OptiQ는 fresh random latent와 conditional sigma를 포함한 정책 샘플이고, SAC·MFPO는 직접 정책 샘플이다. 평가에는 추가 DACER 행동 잡음이나 NovelD 보상을 넣지 않았다. OptiQ mu-only, SAC tanh(mu), MFPO Q-best-of-10은 native 보조 결과로 구분한다.

## Native 보조 평가

- OptiQ: native fixed 성공 97/100, return -971.50, 성공 경로 {'G1/passage-y+4': 97}.
- SAC: native fixed 성공 0/100, return -4188.96, 성공 경로 {}.
- MFPO: native fixed 성공 100/100, return -688.11, 성공 경로 {'G2/passage-y-8': 100}.

v3의 원래 reset도 고정되어 natural/fixed 평가를 독립된 200개 초기 조건으로 합산하면 안 된다. Native 평가에서도 각 성공 정책은 하나의 목표·경로에 집중됐다.

## 학습 및 보관 근거

- OptiQ: 최초 학습 중 성공 185,390 step, 학습 중 성공 3,700/4,251 종료 에피소드, learner updates 995,000.
- SAC: 최초 학습 중 성공 없음, 학습 중 성공 0/1,428 종료 에피소드, learner updates 995,000.
- MFPO: 최초 학습 중 성공 252,166 step, 학습 중 성공 7,180/7,647 종료 에피소드, learner updates 990,000.

훈련은 공통 dense reward + NovelD이며 알고리즘별 native 모델·optimizer 설정과 OptiQ DACER를 유지했다. 따라서 하나의 알고리즘 요소만 바꾼 통제 실험은 아니다. 단일 학습 seed 결과만으로 일반적 알고리즘 우열을 결론내리지 않는다.

Frozen training source: `19fc37a7e51be2bb54d41ab9a980225ecd9f5ba5`.
원본별 `../../runs/v3-<method>-s0/archive-verification.json`에 데이터와 replay 검증 및 SHA256을 보관했다. 그림에 사용한 원본 SHA256과 동일 초기 상태 검사는 `fixed-state-verification.json`에 보관했다.
최종 재개용 상태는 각 run의 `resume/step_0001000000/`에 있으며 추가 학습은 현재 실행하지 않는다.
