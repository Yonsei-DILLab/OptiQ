# AntMaze v2: 완료된 세 방법의 1M 비교

학습 seed 0, 방법별 1,000,000 환경 상호작용. OptiQ·SAC·MFPO의 최종 원본 rollout, 모델·optimizer·RNG·환경 상태와 1M transition replay를 로컬 보관하고 검증했다. v2 MEOW는 아직 학습 중이므로 이 비교에서 제외한다.

## 동일한 초기 상태에서 직접 stochastic 정책을 샘플링한 평가

각 방법의 100개 rollout과 세 방법 사이의 전체 초기 simulator state가 정확히 같은지 확인했다. 아래 수치는 서로 다른 학습 seed의 평균이 아니다.

| 방법 | 성공 | G1 도달 | G2 도달 | 평균 환경 return |
|---|---:|---:|---:|---:|
| OptiQ | 100/100 | 100 | 0 | -174.69 |
| SAC | 100/100 | 100 | 0 | -173.83 |
| MFPO | 100/100 | 100 | 0 | -162.73 |

세 방법 모두 G1의 중앙 통로 한 경로로 성공했다. 목표 도달은 학습했지만 이 seed의 1M 정책에서 여러 목표·경로가 유지된다는 증거는 없다. 궤적의 작은 흔들림은 별도의 성공 경로로 세지 않는다.

![직접 stochastic 정책 비교](fixed-state-comparison.png)

OptiQ는 fresh random latent와 conditional sigma를 포함한 정책 샘플이고, SAC·MFPO는 직접 정책 샘플이다. 평가에 추가 DACER 행동 잡음이나 NovelD 보상을 더하지 않았다. OptiQ mu-only와 SAC tanh(mu), MFPO Q-best-of-10 평가는 native 보조 결과에 따로 보존했다.

## Native 보조 평가

- OptiQ: native fixed 성공 100/100, return -158.00, 목표별 횟수 {'1': 100}.
- SAC: native fixed 성공 100/100, return -172.90, 목표별 횟수 {'1': 100}.
- MFPO: native fixed 성공 100/100, return -158.00, 목표별 횟수 {'1': 100}.

v2는 원래 reset도 고정되어 natural/fixed 평가를 독립된 200개 초기 조건으로 합산하면 안 된다.

## 학습 및 저장 근거

- OptiQ: 최초 학습 중 성공 26,829 step, 학습 중 성공 20,061/20,221 종료 에피소드, learner updates 995,000.
- SAC: 최초 학습 중 성공 42,199 step, 학습 중 성공 21,069/21,290 종료 에피소드, learner updates 995,000.
- MFPO: 최초 학습 중 성공 64,623 step, 학습 중 성공 23,747/23,922 종료 에피소드, learner updates 990,000.

훈련은 공통 dense reward + NovelD이며 알고리즘별 native 모델·optimizer 설정과 OptiQ DACER를 유지했다. 따라서 하나의 알고리즘 요소만 바꾼 통제 실험은 아니다.

Frozen training source: `19fc37a7e51be2bb54d41ab9a980225ecd9f5ba5`.
원본별 `../../runs/v2-<method>-s0/archive-verification.json`에 데이터와 replay 검증 및 파일 SHA256을 보관했다. 그림에 사용한 원본 SHA256과 동일 초기 상태 검사는 `fixed-state-verification.json`에 보관했다.
최종 재개용 상태는 각 run의 `resume/step_0001000000/`에 있다. 추가 학습은 현재 실행하지 않는다.
