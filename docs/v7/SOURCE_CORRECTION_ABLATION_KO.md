# OptiQ v7: actor source importance 보정 제거 비교

이 비교는 **teacher의 Boltzmann importance correction은 유지하고, actor loss에 곱하는
추가 source 보정만 끄는 실험**이다. Source 보정 스위치의 기본값은 True이며,
이 실험에서만 False로 설정한다. False는 checkpoint의 설정 식별 정보에도 포함하여
보정 여부가 다른 checkpoint를 같은 알고리즘의 재개로 혼용하지 않는다.

이 문서는 실험 설계다. 아래 queue 배치는 등록 계획이며 실제 등록·시작 여부는
[실행 manifest](/root/optiq-experiments/v7/gmm40-training/noSourceIS-P256-H4096-K16-T1-e01-20260919/jobs.json)에서
확인한다. 아직 이 문서에 학습 결과나 개선 결론을 기록하지 않았다.

## 1. 유지하는 보정과 제거하는 보정

원래 teacher 후보의 가중치

\[
W_j=\operatorname{softmax}_j\left(Q(s,b_j)/T-\log q(b_j\mid s)\right)
\]

는 그대로 유지한다. 전체 proposal 밀도 q로 계산한 W에 따라 256개 후보에서 16개를
importance 재표집하고, 각 occurrence의 OT 열 질량을 1/16으로 둔다. W를 끄거나
다시 곱하지 않는다. Q가 정의하는 Boltzmann 목표를 teacher에서 근사하는 부분은 같다.

OT source 선택도 기존과 같이

\[
P_{ij}=\frac1K\Pr(i\mid\tilde b_j,s),\qquad
I_j\sim\Pr(\cdot\mid\tilde b_j,s)
\]

를 사용한다. 이번에 끄는 항은 선택된 source에 적용하던

\[
\operatorname{sg}\left[\frac{1/H}{\sum_jP_{ij}}\right]
\]

이다. False일 때 **실제 actor loss weight는 모두 1**이며, 기존 ratio는 진단용으로
계속 계산·보존한다. 큰 raw ratio가 로그에 남아 있더라도 그 값이 loss에 곱해졌다는
뜻은 아니다. 적용 weight와 raw ratio를 구분해서 해석해야 한다.

## 2. 기대 actor update가 어떻게 달라지는가

한 상태 s, 갱신 전 potential f, 현재 teacher batch를 고정한다. Source i의
새 행동에 대한 조건부 loss의 기대값을

\[
F_i(\theta)=\mathbb E_{a\sim\pi_i}
\left[T\log\pi_i(a\mid s)-Q(s,a)-T\log\Pr(i\mid a,s)\right]
\]

라 하면 두 설정은 다음을 평균적으로 학습한다.

\[
J_{\rm on}=\frac1H\sum_i F_i(\theta),\qquad
J_{\rm off}=\sum_i\left(\sum_jP_{ij}\right)F_i(\theta).
\]

True는 OT로 편향되게 선택한 source를 uniform latent 적분점 prior로 보정한다.
False는 현재 OT에서 source가 선택되는 질량에 따라 학습 비중을 둔다.
위 식은 **현재 map과 batch를 고정한 actor update의 조건부 objective**다.
Teacher와 P를 통과하는 gradient는 계산하지 않으므로, P의 actor 의존성까지 포함한
하나의 전역 함수 전체를 미분한다는 주장과 구분한다.

추론 시 latent는 두 설정 모두 같은 연속 normal prior에서 표집한다. 학습할 때의
source 선택 빈도와 추론 prior가 반드시 같아야 하는 것은 아니다. 예를 들어 조건부들이
독립 파라미터를 갖고, 모든 가중치가 양수이며, 각 조건부를 정확히 최적화할 수 있다면,
각 F_i에 다른 양수 상수를 곱해도 조건부별 최소점은 같을 수 있다. 따라서 이 source
보정이 모든 Boltzmann 정책 학습에 보편적으로 필수라는 주장은 과하다.

그러나 현재 Gaussian들은 하나의 MLP 파라미터를 공유하고 유한 표본으로 학습한다.
Source별 가중치가 달라지면 공유 파라미터의 최적화 절충과 유한시간의 학습 경로도 달라진다.
희귀 source는 보정을 끄면 거의 학습되지 않을 수 있고, 반대로 드문 거대 ratio에 의한
전체 actor 충격은 줄어들 수 있다. 어느 효과가 큰지는 실험으로 판단해야 한다.

조건부 목표 t_i를 정확히 맞추고 실제 Boltzmann 분포에 대한 source 질량이 1/H로
균형을 이루면 uniform mixture가 Boltzmann 목표를 복원한다는 충분조건은 여전히 같다.
Source 보정을 켜거나 끈 사실만으로 그 조건이 성립하거나 mode 유지가 보장되지는 않는다.
False를 Q의 density correction 오류 수정이나 Boltzmann 복원의 보장으로 설명하지 않는다.

## 3. 비교 설정과 queue 계획

| 항목 | 설정 |
|---|---|
| Source correction | False; 실제 actor weight 1, raw ratio는 진단 유지 |
| Teacher W 및 256→16 importance 재표집 | 기존 유지 |
| Seed / 길이 | 0, 3 / 각 100K actor updates |
| Actor gradient clipping | 사용하지 않음 |
| Latent 적분점 / teacher 후보 / actor 학습 쌍 | 4096 / 256 / 16 |
| T=alpha / OT epsilon | 1 / .1 |
| Actor Adam / dual Adam 학습률 | 3e-4 / 1e-4 |
| OT coupling·persistent dual·추론 latent prior | 기존 유지 |
| 나머지 설정 | 기존 network·sigma 범위·teacher floor·batch·update frequency 유지 |

계획한 배치는 GPU 0에서 clip=2 seed 0 비교가 끝난 뒤 no-source-correction seed 0,
GPU 1에서 clip=2 seed 3 비교가 끝난 뒤 no-source-correction seed 3을 실행하는 것이다.
기존 실험의 source나 checkpoint를 덮어쓰지 않고 별도 run으로 처음부터 학습한다.
실제 queue 상태는 위 manifest를 기준으로 한다.

이 실험과 [actor gradient clipping 비교](GRAD_CLIP_DIAGNOSIS_KO.md)는 서로 다른 개입이다.
Clipping은 기존 weighted gradient 전체를 제한한다. 여기서는 clipping 없이 source별
기대 학습 비중을 바꾼다. 두 결과를 같은 안정화 기법의 설정값 차이로 취급하지 않는다.

## 4. 확인할 결과

기존 unclipped source-correction=True의 동일 seed 결과와 mode coverage, mode별 질량,
near-mode 비율, MMD², sliced W₂를 함께 비교한다. Teacher에서 유지되는 mode와 actor에서
사라지는 mode를 구분하고, raw source ratio의 분산과 실제 gradient·parameter 이동량도
확인한다. Teacher W의 density correction이 유지되므로 teacher의 좋은 분포만으로
actor의 최종 분포 적합까지 성공했다고 판단하지 않는다.

관련 설명은 [알고리즘](ALGORITHM_KO.md), [공통 표기](NOTATION_KO.md),
[모드 소실 진단](GRAD_CLIP_DIAGNOSIS_KO.md)에 있다.
