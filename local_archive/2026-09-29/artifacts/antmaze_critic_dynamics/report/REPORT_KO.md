# OptiQ critic dynamics 500k 결과

12개 모두 완료했습니다. 마지막 controller 상태 기준 2026-09-25 03:05 KST까지 완료, 실행·대기·실패 0개입니다.

v3/v4 × 6조건, 각 training seed0 하나. 최종 예산은 warmup 제외 500,224 transitions, warmup 포함 508,416입니다. 모든 조건 critic 15,632 updates; actor 지연 조건만 7,816 updates입니다.

보상은 100(d_current−d_next), 성공 bonus0·step penalty0입니다. 스케일 조건은 20(d_current−d_next)와 T=.2입니다. 다른 조건은 T=1. DACER/NovelD OFF, replay1M,256env,batch4096,8updates/256transitions를 유지했습니다.

## 경로와 성공

동일한 원래 고정 full state에서 각 정책을 100회 평가했습니다. 직접 정책 샘플링은 random latent와 conditional sigma를 모두 포함하며, 외부 DACER 잡음은 없습니다. 아래 경로 수는 통로 진입 수이며 목표 도달 성공 수와 다릅니다.

| 조건 | v3 왼쪽/오른쪽/미진입 | v3 성공 | v4 위/아래/미진입 | v4 성공 |
|---|---:|---:|---:|---:|
| 대조군 | 0/93/7 | 44% | 99/0/1 | 0% |
| 보상·T ×0.2 | 100/0/0 | 60% | 99/0/1 | 0% |
| EMA τ=.01 | 0/97/3 | 62% | 100/0/0 | 0% |
| Critic LR ×2 | 0/99/1 | 51% | 100/0/0 | 0% |
| Actor 업데이트 절반 | 100/0/0 | 60% | 99/1/0 | 0% |
| 스케일 축소 + EMA | 0/100/0 | 63% | 100/0/0 | 0% |

**다중 경로를 안정적으로 유지한 조건은 없습니다.** v3는 선택 방향이 바뀐 조건도 있으나 최종에는 한쪽만 관측됐습니다. v4 actor delay2의 아래쪽 1/100회를 제외하면 모두 위쪽으로 집중됐습니다.

v4 여섯 조건은 최종 평가뿐 아니라 500k 학습 전체의 목표 도달 기록도 모두0입니다. 따라서 첫 goal 성공 경험은 이번 경로 집중 현상의 필요조건이 아닙니다. 종료 구조나 보상 목적의 영향 전반을 배제하는 결과는 아닙니다.

![경로 비율 변화](route_evolution.png)

![v3 최종 궤적](v3_final_trajectories.png)

![v4 최종 궤적](v4_final_trajectories.png)

중간 평가는40회, 최종100회입니다. 경로 그림의100–508k는 warmup 포함 total counter이고, Q 진단 곡선은 warmup을 제외한 counter입니다.

## Q 추정과 가설

아래는 최종 정책의 시작 상태에서 평균 MC return−online twin-mean Q입니다. 보상 축소 조건도 reward_multiplier=.2로 나눠 원래 단위로 복원했습니다. 전체100회(실패 포함)의 평균입니다.

| 조건 | v3 MC−Q | v4 MC−Q |
|---|---:|---:|
| 대조군 | 285.5 | 316.8 |
| 보상·T ×0.2 | 267.9 | 364.5 |
| EMA τ=.01 | 214.5 | 268.0 |
| Critic LR ×2 | 333.7 | 391.4 |
| Actor 업데이트 절반 | 305.1 | 311.0 |
| 스케일 축소 + EMA | 243.0 | 148.2 |

1. **Target 지연: 추종 오차 개선은 확인됐지만 경로 유지에는 부족했습니다.** τ .005→.01에서 마지막 replay online−target RMS는 v3 1.770→0.996, v4 2.774→2.011로 줄었습니다. 시작 상태 MC−Q도 줄었지만 반대 경로는 오히려 더 일찍 미관측됐습니다. v3 성공률은44→62%로 높아졌습니다.

2. **최적화 지연: 단순한 critic LR 부족이라는 설명은 이번 비교에서 뒷받침되지 않았습니다.** LR×2에서 마지막 replay TD residual RMS는 v3 6.130→6.661, v4 4.446→4.392입니다. 큰 개선은 없었습니다. 진단은 다음 행동8개 표본 평균을 사용하므로 남은 표본 변동까지 전부 최적화 오차로 해석할 수 없습니다. 마지막 replay 진단은 total500224/global492032이고, 최종 rollout은 total508416/global500224입니다.

3. **정책 변화·방문 편중: 반복적인 경로 소실은 재현됐지만 원인을 하나로 확정하지 못했습니다.** v3 control은300k 평가 left14/right24에서400k left8/right27,500k left0/right39로 바뀌었습니다(나머지는 미진입). Actor 업데이트를 절반으로 줄이면 최종 left100으로 방향만 바뀌었습니다. v4 criticLR2는300k 위18/아래22에서 최종 위100으로 바뀌었습니다.

v3 control의 마지막 replay 진단1024개 표본에는 왼쪽 영역159개, 오른쪽94개가 남아 있었습니다. 이번500k는 replay1M이 가득 차기 전이기도 합니다. 따라서 반대쪽 경험이 버퍼에서 전부 지워져서 생긴 현상으로 설명할 수 없습니다. 영역별 표본은 완전한 경로 수가 아닙니다.

![Critic 진단 곡선](critic_diagnostics.png)

## 해석의 제한

- 단일 training seed입니다. 평가100회는 training seed100개가 아닙니다.
- MC−Q는 공통적인 가치 과소추정도 포함합니다. 후보 행동들의 공통 Q 편향은 teacher softmax에서 상쇄되므로, MC−Q 크기만으로 경로 소실 원인을 입증할 수 없습니다.
- 조건마다 학습된 정책과 replay 방문 분포가 다릅니다. 같은 state/action bank를 고정한 통제 비교는 아닙니다.
- TD fitting·EMA·twin-min의 할인합은 대수적 분해입니다. 예를 들어 v3 control의 gap 약285.5는 TD84.9+EMA43.1+twin-min157.5로 표현되지만 각 설정을 바꿨을 때의 인과적 개선량을 뜻하지 않습니다.
- Raw MC는 timeout에서 종료합니다. Online critic으로 tail을 보완한 값은 독립적인 정답이 아니므로 별도로 보존했습니다. 이번 시작 상태 평균의 보완량은 절댓값0.15 미만이었습니다.
- Teacher ESS는 현재 상태 주변 행동 후보의 집중도이며, 여러 장기 경로가 남아 있다는 증거는 아닙니다.

## 저장과 검증

학습 source: `23603a7e7696aa64e8e49a38b986d6b430b6d6f3`, branch `direct-gmm-trg-antmaze`. W&B `OptiQ/antmaze`, group `antmaze-optiq-critic-dynamics-500k-s0-20260925`.

원시 평가 NPZ156개·진단 JSON72개 및 설정/검증 자료를 로컬에 보관했습니다. 922파일 SHA256 검증을 통과했습니다. 모든 단계·mode를 합친 episode를 한 정책의 독립 평가처럼 합치지 않았습니다. 최종 replay/checkpoint는 서버에 보존돼 있으며 이번 보고에는 대용량 학습 상태를 내려받지 않았습니다.

정리된 수치는 `evaluation_metrics.csv`, `results.json`; 수집 검증은 상위 `result-collection-verification.json`에 있습니다. 보고 생성은 학습이나 서버 파일을 변경하지 않았습니다.
