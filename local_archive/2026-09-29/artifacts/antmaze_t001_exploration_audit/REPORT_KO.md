# T=0.01 AntMaze 탐색 진단 — 2026-09-23

첫 목표 도달 전 Q가 평평하다는 것만으로 설정 오류나 temperature 문제를 확정할 수 없다. 외재보상만 있다면 도달 전 replay의 reward는 모두0이므로 학습된 Q가 행동들을 비슷하게 평가하는 것이 자연스럽다. 현재는 NovelD까지 critic에 넣고 있으므로, Q가 구분할 수 있는 대상은 미래 intrinsic return의 차이다. 먼저 그 보상 자체가 새 공간 방문을 얼마나 우대하는지, 그리고 그 차이를 critic이 학습하는지 구분해야 한다. 이번 측정은 actor에 들어가는 Q 기반 선택이 약하고 행동 분산이 이미 크다는 것을 확인했다. T를 더 낮추는 것이 1순위 해법이라는 초기 해석은 철회한다.

## 측정 대상과 결과

학습 source `8bb0c50a1356dc082106393f161ff3fc3d2de0af`. v1–v4 OptiQ seed0, sparse reward + 공식 NovelD 0.01, T=0.01, 256env/batch4096/8updates per256 transitions. 저장된 최신 평가 checkpoint의 SHA256을 검증하고, 각 평가의 **서로 다른 random 시작 상태40개**, 상태당64개 행동 후보로 actor/critic을 CPU에서 다시 계산했다. 공식 관측은 qpos15+qvel14이며 평가 원시 NPZ의 시작 상태와 동일하다. 이 상태 집합은 replay에서 뽑은 상태 집합이 아니다. 학습·optimizer update·새 환경 step은 실행하지 않았다.

| Maze | 체크포인트 interactions | 상태 내 후보 Q 표준편차 | Q-only ESS | Q 제거 전후 teacher TV | 전체 행동 좌표 RMS 표준편차 |
|---|---:|---:|---:|---:|---:|
| v1 | 2,750,208 | 0.001065 | 63.22/64 | 4.56% | 0.520 |
| v2 | 2,000,128 | 0.001129 | 63.14/64 | 4.66% | 0.521 |
| v3 | 2,500,096 | 0.001131 | 63.15/64 | 4.86% | 0.522 |
| v4 | 2,500,096 | 0.001179 | 63.05/64 | 4.95% | 0.521 |

실제 teacher는 `softmax(Q/T − log q)`이고 twin Q의 평균을 쓴다. Q-only ESS는 `softmax(Q/T)`로 별도 계산한 진단값이며, density correction까지 포함한 실제 teacher의 ESS와 다르다. 64에 가까울수록 이 후보 집합에서 Q만으로는 거의 균일한 가중치를 준다는 뜻이다. TV는 동일 후보에서 Q 항을 제거했을 때의 `0.5 sum(abs(w_full−w_without_Q))`다. 평균 Q 자체는 약0.40–0.46이고, 행동 간 차이는 약0.001이다.

현재 T=0.01에서 Q가 전혀 무시되는 것은 아니다. 하지만 후보64개가 Q 관점에서 거의 동등하고, Q를 빼도 가중치 변화가 5% 안팎이다. 학습 중 replay minibatch 로그에서도 Q 표준편차가 약0.001–0.0013으로 비슷하다. density correction이 잘못됐다는 증거는 아니다. Q가 평평하면 Boltzmann 목표 분포가 거의 균일해지는 것이 수식의 결과다.

조건부 sigma는 네 정책 모두 exp(-1)=0.367879 상한에100% 붙어 있고 raw log sigma도 모두 상한보다 높다. 그러나 latent가 바뀔 때 mean도 바뀌므로 전체 행동 표준편차는 약0.52다. 비교로 [-1,1] 균등분포의 표준편차는0.577이다. 이 두 번째 모멘트만으로 행동 분포가 완전히 균일하다고 주장할 수는 없지만, 전체 행동이 거의 결정적이거나 좁다는 설명은 맞지 않는다. 큰 독립 행동 변화와 여러 step에 걸쳐 먼 상태에 도달하는 보행은 별개의 능력이다.

## 원본 DDiffPG/DIPO와의 차이

공식 upstream commit `7edd06c4799abbab0f8fa534c21deb56253b018e`를 기준으로 확인했다. 지금까지 우리 서버에서 목표 도달을 관측했던 원본 계열 방법은 DIPO baseline이다. DDiffPG 본체를 실행한 것으로 표현하면 안 된다.

| 항목 | 현재 OptiQ | 원본 DIPO baseline | 원본 DDiffPG 본체 |
|---|---|---|---|
| 정책 개선 | 고정 T의 Boltzmann teacher를 GMM NLL로 회귀 | replay target action을 Adam/Q-gradient로20번 개선 후 diffusion 회귀 | 같은 Q-gradient 개선 + 모드별 학습 |
| Random warmup | 8,192 interactions | 8,192 | 128,000 |
| Target critic tau | .005 | .05 | .05 |
| Critic LR | 3e-4 | 5e-4 | 5e-4 |
| 정책 외부 잡음 | DACER 약.02 | worker별.05–.6 mixed noise | worker별.05–.6 mixed noise |
| 추가 경로 유지 구조 | 없음 | 없음 | 성공 궤적 군집화·경로별 critic·탐색 critic |

원본 NovelD 계수도.01이며, 수식과 RND 갱신은 이식 코드와 일치한다. 256개 수집 후8회 update, batch4096도 baseline과 같다. 현재 learner/RND counter도 일치한다. NovelD가 누락됐거나 병렬 환경 때문에 업데이트를 잘못 세었다는 증거는 없다. DIPO는 고정 T로 작은 Q 차이를 지수 가중하는 방식이 아니므로, 같은 보상 계수라도 actor에 전달되는 정책 개선의 강도가 같지 않다. Adam도 epsilon/gradient 크기의 영향을 받으므로 완전한 보상 스케일 불변이라고 주장하지 않는다.

참고로 이전 T=.25 체크포인트의 replay 진단에서는 실제 NovelD 평균이 약.0036–.0038이었고 `(s,s)` 가상 입력의 보너스 평균도 비슷했다. 공식 수식 `max(n(s')−.5n(s),0)`는 RND 오차가 남으면 이동이 없어도 양수다. 이는 기존 측정이며 이번 T=.01의 RND를 재측정한 결과는 아니다. NovelD 전체 값을 새로운 xy 영역으로 나아가는 추가 보상과 동일시할 수 없고, 계수를 올리면 그 배경 보너스도 같이 커진다.

## 첫 도달 이후 변화와 추가 NovelD 확인

분석 중 v2는 약2.75M에서 학습 성공9회를 기록했고, 2.5M random-start 평가에서1/40 성공했다. v1은3,008,256 interactions를 마쳤고 최종 random-start direct-policy 평가0/100이었다. v3/v4는 이 시점에 성공0이었다. 상태 스냅샷은 `training-status-1723.json`에 보존했다.

v2를 동일 T=.01의2.75M checkpoint에서 다시 계산하면 Q 표준편차는.001129→.003476, Q-only ESS63.14→55.12, Q 제거 teacher TV4.66%→13.83%로 변했다. 온도를 바꾸지 않아도 목표를 발견한 이후 가치 차이가 커질 수 있다는 해석과 일치한다. 동시에 학습도250k 진행됐으므로 이것만으로 sparse 보상만의 인과 효과를 분리한 것은 아니다. 추가 원자료는 `v2-after-first-goal-forward.json`이다.

완료한 현재 v1의 마지막1M replay에서8,192 transitions를 뽑아 최종 RND를 고정하고 재계산했다. NovelD 평균.004022, 표준편차.001424였고 `(s,s)` 가상 입력의 평균은.004034였다. 마지막1M replay의0.5m xy 격자 방문 빈도로 나눴을 때, 덜 방문한 그룹의 보너스 평균.004550, 자주 방문한 그룹.003746으로 약21% 높았다. 보너스와 log 방문 빈도 상관은-.263, 한 step xy 이동량 상관은.124였다.

따라서 공간 방문 빈도와 무관한 보너스라고 단정할 수는 없다. 하지만 기본적으로 존재하는 양의 보너스와 공간 탐색에 따른 추가 차이는 구분해야 한다. 방문 빈도는 보존된 마지막1M 구간의 값이며, 최초 방문 여부가 아니다. 최종 RND의 재계산이므로 당시 수집 시점의 novelty 값과도 다르다. 자세/속도 차이 등도 통제하지 않은 관측 비교여서 인과 실험은 아니다. `inspect_current_noveld.py`와 `current-noveld.log/json`에 보존했다.

## 다음 검증 순서 — 설정 변경은 실행하지 않음

1. **새 xy 영역으로 진행한 궤적과 익숙한 영역을 반복한 궤적의 누적 intrinsic return을 비교**하고, 그 차이를 critic이 예측하는지 확인한다. 즉시 NovelD 한 값과 Q 분산만으로 원인을 정하지 않는다. 학습 당시의 NovelD/RND와 시점을 맞춘 로그가 있어야 최초 방문 보상을 정확히 검증할 수 있다.
2. 보상은 구별되는데 critic이 못 따라가면 원본과 다른 target tau(.005 대.05), critic LR(3e-4 대5e-4)를 각각 대조할 근거가 생긴다. 보상·Q 모두 구별되는데 actor가 거의 동일한 가중치를 준다면 T를 낮추는 대조를 검토한다. 이번 고정 checkpoint에서 T=.003/.001은 Q-only ESS54–56/21–25로 바뀌지만, 학습 성능이나 최적 T를 입증하지 않는다.
3. 10k learner updates(약328k interactions) 대조에서는 **누적 신규 xy 격자 수, 최장 시작점 이탈 거리, 통로 진입 깊이, 방문 집중도**를 우선 비교한다. 성공만 기준으로 고르지 않고 학습 중 탐색과 최종 정책의 경로 다양성도 분리한다.

sigma 상한 해제, NovelD 계수 급증, 매우 낮은 T를 이번 분석만으로 확정하지 않는다. NovelD만 키우면 반복 상태의 배경 보너스도 커지고, Q에 유효한 구별 신호가 없을 때 T를 낮추면 추정 오차를 증폭할 수 있다. 고정-T 대신 적응형 T를 도입하는 것도 별도 알고리즘 변경이다.

## 재현 자료

- `inspect_eval_start_policy.py`: 기존 forward-only 진단을 평가 시작 상태에 적용한 코드.
- `forward-results.json`: 체크포인트별 SHA256, 측정 정의, 수치 및 온도 재가중 결과.
- `v1-forward.log`–`v4-forward.log`: CPU 진단 원시 출력.
- `temperature_signal.png/pdf`: 학습을 다시 하지 않고 동일 체크포인트를 재가중한 그림.
- `../antmaze_settings_audit/REPORT_KO.md`: 이전 T=.25의 replay/RND 감사. 이번 결과와 구분해야 한다.

공식 근거: [DDiffPG 저장소](https://github.com/supersglzc/ddiffpg), [DDiffPG 논문 §4 및 Appendix E](https://arxiv.org/html/2406.00681v1#A5). 현재 실험의 하이퍼파라미터·학습 소스는 수정하지 않았다.
