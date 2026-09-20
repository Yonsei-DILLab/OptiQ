# 조건부 OptiQ v7: 모드 소실 진단과 actor gradient clipping 비교

이 문서는 복원된 조건부 objective의 특정 모드 소실 사건과 후속 비교 설계를 기록한다.
현재 목표는 [조건부 Boltzmann 정책 추출](ALGORITHM_KO.md)이며, 전체 mixture SAC로
되돌리거나 teacher의 importance correction을 제거하는 실험이 아니다.
아래 수치는 기존 unclipped seed 3에서 얻은 진단이다. Clipping의 100K 결과는 아직
이 문서에 포함하지 않으며, 장기 개선이나 mode 복원 성공을 주장하지 않는다.

## 1. Teacher가 mode를 찾는데 actor가 먼저 놓친 사례

H=4096, teacher 256→16, T=1, OT epsilon=.1의 기존 seed 0–3를 20K까지 정확히 재현했다.
0·1K·5K·10K·15K·20K의 총 24개 체크포인트에서 actor와 dual의 파라미터·Adam 상태·step,
그리고 훈련 RNG가 기존 기록과 일치했다. 진단 표본의 RNG는 훈련과 분리했다.

Seed 3의 mode 14(0-based index)는 13.5K에서 actor 질량 6.825%였지만,
14K에서 65,536 actor 표본 중 0개로 관측됐다. 같은 14K에서 teacher 질량은
W 보정 후 .3127%, 256→16 재표집 후 .3113%로 기존 coverage 기준 .256%를 넘었다.
Teacher가 먼저 mode를 완전히 놓친 것으로만 설명할 수 없다. 표본 0개는 수학적인
분포 질량 0을 뜻하지 않는다.

## 2. 단일 update의 직접 진단

13,709 update에서 최대 source importance는 **21,788.50**이었다. 해당 source의
행 질량은 1.1205e-8, prior 질량은 1/4096이다. Batch 256 × 16쌍 중 한 쌍이
weight 합의 **83.667%**를 차지했고, 전체 4096쌍의 실제 global ESS는 **1.4284**였다.

동일한 update 직전의 actor·dual·Adam 상태와 teacher·source index·noise를 고정하고,
그 한 쌍의 gradient 기여만 0으로 둔 진단은 다음과 같다. 나머지 weight와 batch 평균
분모는 유지했다.

| 조건 | 전체 actor gradient norm | Adam 후 parameter 이동 L2 |
|---|---:|---:|
| 실제 update | 3,601.176 | .173121 |
| 최대 weight 쌍의 기여만 제거 | 9.305 | .006353 |

그 쌍의 gradient와 전체 gradient의 cosine은 .9999967이었다. 이 update의 큰 이동이
해당 쌍에 지배된 것은 확인했지만, 쌍을 제거한 경로를 이후 100K까지 학습한 비교는 아니다.
모든 seed의 mode 소실 원인을 이 사건 하나로 일반화하거나 mode 소실 예방을 확정하지 않는다.

같은 actor·sample·weight를 두고 assignment query의 f만 한 step 갱신값으로 바꾸면
gradient 상대 변화는 .1842%였다. 이 검사는 source를 다시 뽑거나 weight를 다시 계산한
것이 아니다. 동시점의 급격한 dual 이동이 큰 gradient의 직접 원인이라는 설명과는 맞지 않지만,
기존 f와 작은 행 질량이 사건에 무관하다는 뜻도 아니다.

별도로 구한 항별 gradient의 전체 gradient 방향 투영은 -Q 약 69%, assignment 약 29%,
조건부 entropy 약 1.6%였다. 별도 JIT 경로의 수치 차이 때문에 정확히 합 100%의 분해는
아니다. 이 값은 gradient 방향 투영이며 loss 값 비중이나 분산 설명률이 아니다.
큰 source weight가 Q와 assignment gradient를 함께 증폭한 사건이다.

평균 per-lane ESS fraction은 이때도 .593이어서 위험을 제대로 드러내지 못했다.
전체 batch global ESS, 최대 weight의 합 대비 비중, raw gradient norm과 실제 parameter
이동량을 함께 봐야 한다. 다음 step의 raw gradient는 19.55로 줄었지만 Adam 상태의 영향으로
parameter 이동은 .1554로 계속 컸다. 최초 충격 뒤의 optimizer 상태도 중요한 관측 대상이다.

## 3. 두 importance correction을 유지하는 이유

Teacher 보정은

\[
W_j=\operatorname{softmax}_j\left(Q(s,b_j)/T-\log q(b_j\mid s)\right)
\]

이며 proposal의 행동을 Boltzmann 목표로 보정한다. W는 256→16 재표집 빈도에 반영하고
그 뒤 다시 곱하지 않는다. 추가 source 보정은

\[
\operatorname{sg}\left[\frac{1/H}{\sum_jP_{ij}}\right]
\]

이며 OT에 따라 선택한 latent의 학습을 uniform 적분점 prior에 맞춘다. 역할이 다르다.
현재 sampling을 그대로 둔 채 source 보정을 없애면 latent 선택의 기대 objective가 바뀌고,
W를 없애면 기존 Boltzmann teacher 추출 방식이 달라진다. 이번 비교에서는 둘 다 유지한다.

고정 map과 teacher batch를 조건으로 source 보정은 기대 actor gradient를 올바르게 만들지만,
희귀 source의 큰 weight로 인한 유한표본 분산을 제거하지는 못한다. 정확한 importance ratio와
안정적인 신경망 optimizer 이동은 별개 조건이다. Gaussian conditional의 표현 한계 및 모드별
질량 보존 문제까지 source 보정만으로 해결된다고 가정하지 않는다.

## 4. Actor global-norm clipping 비교 설계

기존 weighted conditional loss를 그대로 미분한 actor gradient g에 대해 Adam 입력 직전에

\[
g_{\rm clip}=g\min\left(1,\frac{c}{\lVert g\rVert_2}\right)
\]

를 적용한다. Actor 전체 parameter pytree의 global L2 norm을 사용한다. Teacher W와
개별 source importance를 자르거나 다시 정규화하지 않는다. Dual optimizer와 dual gradient는
변경하지 않는다.

| 항목 | 비교 설정 |
|---|---|
| Actor global-norm 상한 c | 2, 10 |
| Seed | 각 설정에서 0, 3 |
| 학습 길이 | 각 100K updates, 총 4개 실험 |
| 원래 teacher / importance 재표집 / actor 학습 쌍 | 256 / 16 / 16 |
| Latent OT 적분점 | H=4096 |
| GMM 온도 T=alpha / OT epsilon | 1 / .1 |
| Actor Adam / dual Adam 학습률 | 3e-4 / 1e-4 |
| Dual update | Actor update당 1회, 기존 persistent parameter·Adam 유지 |
| 나머지 설정 | 기존 network·sigma 범위·teacher floor·batch·update frequency 유지 |

한 batch 안의 모든 gradient 성분을 같은 양수 배율로 줄이므로 그 step의 raw gradient
방향은 유지한다. 그러나 배율이 batch에 의존하므로 일반적으로
E[g_clip] != E[g]다. **명시적인 loss·importance ratio는 유지하지만 stochastic optimizer
동작과 기대 clipped gradient는 달라진다.** Importance weights를 유지했다는 이유로
unbiased gradient까지 보존한다고 설명하면 안 된다.

또한 Adam은 과거 모멘트를 사용하므로 gradient norm clipping은 parameter 이동량 자체의
엄밀한 상한이나 trust region이 아니다. 사건 이후를 이어받기보다 초기 상태부터 paired seed로
비교하여 optimizer 이력 전체의 영향을 본다. 이번 설정은 full RL campaign의 결과를 의미하지 않는다.

## 5. 판정 기준과 한계

Unclipped 동일 seed의 기존 결과와 mode별 질량, coverage, near-mode 비율, MMD², sliced W₂를
함께 비교한다. 근처 mode에 더 조밀해져 near-mode 비율만 올라가는 것을 성공으로 보지 않는다.
가능하면 raw/clipped gradient norm, clip 비율, parameter 이동량, source global ESS와
최대 쌍 비중도 함께 확인한다.

Clipping은 공유 MLP에 들어오는 드문 큰 충격을 줄이는 실험이다. 목표 조건부가 multimodal인데
각 conditional이 단일 Gaussian인 표현 한계나 actor 기반 proposal이 mode를 재발견하지 못하는
문제까지 자동으로 해결한다고 가정하지 않는다. 성공 여부는 새 100K 결과로 판단해야 한다.

## 원자료

- [전체 관측·진단 보고서](/root/optiq-experiments/v7/diagnostics/mode_loss_timeline/REPORT_KO.md)
- [0–20K mode별 곡선](/root/optiq-experiments/v7/diagnostics/mode_loss_timeline/dense/mode_loss_timeline.png)
- [단일 update gradient 분해 JSON](/root/optiq-experiments/v7/diagnostics/mode_loss_timeline/cause/gradient_breakdown.json)
- [공통 표기](NOTATION_KO.md) · [조건부 알고리즘](ALGORITHM_KO.md)
