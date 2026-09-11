# OptiQ와 Boltzmann softmax Bellman operator: 실험 대조 메모

2026-09-10. 기존 argmax 5-seed 분석과 검증 완료된 categorical 60/60을 사용했다. 새 학습은 수행하지 않았다.

현재 판단은 **일부 조건에서 Boltzmann backup 평균을 잘 복구하지만, 두 선행연구의 학습상 장점이 OptiQ에 전이됐다는 증거는 아직 부족하다**는 것이다. 일부는 직접 검증하지 않았고, 일부는 재현되지 않았으며, 고차원 extraction에는 명확한 차이가 있다. 이 셋을 구분해야 한다.

| 선행연구의 주장 | 현재 실험으로 가능한 판단 |
|---|---|
| Revisiting §3: temperature와 max 근접성, 이상적인 value iteration의 suboptimality bound | Temperature sweep 및 Q*와의 반복 오차 실험이 없어 직접 검증하지 않았다. 현재 고정 τ에서의 평균 비교는 제한된 진단이다. |
| Revisiting §5.1 / SD2 §4.2.1: overestimation 완화 | MoveCar의 rollout 대비 bias는 DDPG +7.24 → SD2 +0.56으로 5/5 개선돼 방향이 일치한다. OptiQ는 이미 음의 bias여서 DDPG 대비 값이 낮다는 사실만으로 더 정확하다고 할 수 없다. Frozen backup 오차와 이 지표는 구분한다. |
| Revisiting §5.2: critic gradient norm/variance 감소 | 동일 critic·minibatch에서 gradient 통계를 비교하지 않았다. K=1 actor sampling 자체의 추가 분산도 구분해야 한다. |
| SD2·SD3 §4.1: 더 유리한 actor value landscape | 일관된 smoothing 우위를 관측하지 못했다. 정확한 grid Boltzmann도 max 대비 barrier 1개 개선/1개 악화/3개 동률이다. |
| SD3 §4.3: TD3 underestimation 완화 | Rollout 대비 TD3 −82.45, SD3 −20.66, OptiQ −33.18. OptiQ도 5/5에서 TD3보다 과소평가가 작지만 SD3보다 크다. 방향은 부분적으로 일치하나 같은 Q에서 backup만 바꾼 인과 비교는 아니다. |
| 제어 성능·sample efficiency 개선 | MoveCar 네 baseline 모두 188, OptiQ 183.56. 원 논문의 bad→good policy 상황을 재현하지 못해 그 메커니즘에 대한 판별력이 부족하다. |

논문 원문: [Revisiting the Softmax Bellman Operator](https://proceedings.mlr.press/v97/song19c/song19c.pdf), [Softmax Deep Double Deterministic Policy Gradients](https://proceedings.neurips.cc/paper/2020/file/884d247c6f65a96a7da4d1105d584ddd-Paper.pdf).

## 실제 rollout 대비 Q bias

최종 1M steps, 5-seed 평균이다. 같은 첫 action을 고정한 뒤 각 방법의 frozen behavior policy로 2048-step discounted continuing return을 계산했다. 비교 Q는 online critic의 twin-mean이다. 이는 100-step undiscounted evaluation return 188과 다른 값이며, target/noisy backup policy와도 구분한다.

| 방법 | Q − rollout value |
|---|---:|
| DDPG | +7.24 |
| SD2 | +0.56 |
| TD3 | −82.45 |
| SD3 | −20.66 |
| OptiQ | −33.18 |

SD2와 SD3의 bias 개선 방향은 현재 구현에서도 재현됐다. OptiQ도 TD3 대비 같은 개선 방향을 5/5 seeds에서 보인다. 다만 서로 다른 critic·학습 policy·sampling 및 noise를 가진 알고리즘 간 비교다. 이를 Boltzmann extraction 정확도의 단독 효과나 선행연구 정리의 직접 실증으로 동일시하지 않는다.

## 같은 Q에서 max와 Boltzmann 평균 사이 차이는 얼마나 유지되는가?

상태를 고정하고 M=max_a Q(a), B=E_{π_Q^τ}Q, A=E_{π_θ}Q라 두면

`M − A = (M − B) − (A − B)`.

따라서 extraction의 양의 평균 오차 A−B는 원래 Boltzmann이 max에서 낮춘 양을 상쇄한다. `(M−A)/(M−B)`가 1이면 Boltzmann 평균과 같고, 0이면 max와 같다. 1보다 크다고 더 좋은 분포인 것은 아니며, 이 비율은 전체 확률밀도나 실제 return에 대한 bias의 지표가 아니다.

원래 argmax/default, 20k updates, K=50, 5-seed 추정 평균. M은 dense grid와 다중 시작점 연속 최적화로 계산했다.

| 문제 | M | B | A | (M−A)/(M−B) |
|---|---:|---:|---:|---:|
| 1D single | 0.4287 | 0.3037 | 0.3348 | 75% |
| 1D asymmetric | 0.4287 | 0.2686 | 0.3394 | 56% |
| 2D 4-mode | 0.8939 | 0.5363 | 0.5196 | 105% |
| 2D 8-mode | 0.7471 | 0.3873 | 0.4082 | 94% |
| 4D separable | 1.2528 | 0.7061 | 1.0668 | 34% |
| 8D separable | 2.5055 | 1.4121 | 2.2651 | 22% |

2D 8-mode는 평균 수준에서 이상적 backup과 가깝다. 4D·8D는 max에 훨씬 가까운 값을 만들며 mode collapse와도 함께 관측된다. 2D 4-mode는 평균이 가까워도 mode 질량과 seed별 오차가 남는 사례다. 고차원 separable Q는 차원별 Q를 합산하므로 차원·mode 수·Q scale의 영향을 이번 비교만으로 분리하지 않는다.

## Actor–critic 구현에서 추가되는 구조

1. Production actor extraction은 live twin-mean Q, backup은 target twin-min Q를 사용한다. Target actor 지연도 있다. Actor가 자신이 학습하는 분포를 정확히 복구해도, 같은 Q를 추출과 평가에 쓰는 단일 Boltzmann operator와 바로 같아지지 않는다.
2. TD action noise가 실제 backup 분포를 바꾼다. 같은 저장 online actor의 learned-Q 진단에서 raw RMSE 0.06059, noise 추가 0.24851이었다. 이 진단은 실제 target actor 전체 오차의 측정은 아니다.
3. 정확한 actor도 유한 K sampling을 사용하면 조건부 분산 Var_{π_Q^τ}(Q)/K가 남는다. 정확한 기대값이 같다는 사실이 gradient noise나 최적화 경로까지 같게 만들지는 않는다.
4. Frozen Q를 두고도 넓은 분포가 첫 100 actor updates에 무너진다. 이 현상은 Bellman iteration의 비수축성이나 critic drift 없이 발생하므로 actor extraction update 자체에서 원인을 분석할 수 있다.
5. OptiQ의 실제 actor update는 OT target 회귀다. 기존 landscape의 −E[Q(s,g_θ(s,z))]는 SD2의 직접 최적화 목적과 맞지만 OptiQ의 실제 회귀 loss와는 다르다.

특히 target lag와 TD noise를 제외하고 같은 twin Q 쌍을 사용한다고 가정해도, q_m=(Q1+Q2)/2, d_Q=|Q1−Q2|/2라 놓으면 `E_{π_{q_m}}[min(Q1,Q2)] = Bτ(q_m) − E_{π_{q_m}}[d_Q]`다. 따라서 twin-mean을 정확하게 추출하는 actor도 twin-min evaluation을 거치면 critic 간 불일치에 따른 음의 항이 남는다. 이 식은 대수적 항등식이며 새 성능 정리나 현재 underestimation의 원인이 입증됐다는 의미는 아니다. 현재 구현을 분석할 때 추출 정확도와 별도로 측정할 항을 제공한다.

Categorical의 최종 5-seed 결과에서도 4D/8D, default/coverage 모두 100 updates 안에 effective modes=1이 되고 20k에도 유지된다. 일부 scalar RMSE는 개선됐지만 argmax 제거만으로 collapse는 해결되지 않았다. 후보 생성·가중치·유한 Sinkhorn·회귀의 역할은 추가로 분리해야 한다.

따라서 Part I은 **actor로 Boltzmann backup을 구현할 때 어떤 조건에서 원래 성질을 보존하고, 어떤 구현 요소가 이를 바꾸는가**를 중심으로 구성하는 편이 현재 근거와 맞는다. Operator는 이론적 기준으로 유지하고, 학습 메커니즘 주장은 동일 Q·replay에서 직접 검증한다.

후속 검증의 우선순위는 (1) 같은 noisy Q에 exact max / exact Boltzmann / actor를 적용해 actual-value bias와 gradient variance 비교, (2) 같은 Q·noise 없는 설정에서 extraction 오차와 시간 변화 측정 후 mean/min 및 TD noise를 순서대로 추가, (3) actor의 실제 OT 회귀 update를 따라 value와 mode mass가 어떻게 변하는지 측정하는 것이다.

자료: [기존 상세 리포트](../20260910_5seed_analysis/report.md), [수치 집계](analysis.json), [재계산 코드](analyze.py), [categorical 전체 원자료](categorical_summary.json).
