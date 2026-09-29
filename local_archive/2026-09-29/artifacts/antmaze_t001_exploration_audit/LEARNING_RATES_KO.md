# AntMaze 학습률 비교 — 2026-09-23

**OptiQ의 actor LR는 원본과 같다. critic LR는 원본의60%이고 target critic 갱신 계수도10배 작다. 하지만 작은 보상이 곧 거의0인 파라미터 업데이트를 뜻하지는 않는다. 현재 checkpoint의 Adam 상태에서 업데이트가 실제로 진행되는 것을 확인했고, v2는 같은 LR로 첫 도달 이후 최종 stochastic 성공률82%까지 올라왔다.**

현재 OptiQ source `8bb0c50a1356dc082106393f161ff3fc3d2de0af`, T=.01, sparse+NovelD.01, seed0을 기준으로 했다. 원본은 vendored DDiffPG commit `7edd06c4799abbab0f8fa534c21deb56253b018e`다. DDiffPG 본체를 우리 서버에서 실행한 결과와 혼동하지 않는다. 비교 가능한 실제 원본 계열 로그는 이전 DIPO baseline 실행이다.

| 항목 | 현재 OptiQ | 원본 DIPO / DDiffPG |
|---|---:|---:|
| Actor network LR | 3e-4 | 3e-4 |
| Critic LR | 3e-4 | 5e-4 |
| RND predictor LR | 1e-4 | 1e-4 |
| Network optimizer | Adam | AdamW |
| Adam β1,β2 | .9,.999 | .9,.999 |
| Network Adam ε | 1e-8 | 1e-8 |
| Weight decay | 0 | .01 (Torch AdamW 기본값) |
| Gradient norm clip | 없음 | 1 |
| Target critic τ | .005 | .05 |
| Batch | 4096 | 4096 |
| 수집/learner 갱신 | 256 transitions/8 updates | 동일 |
| NovelD 수식/계수 | .01 max(n(next)−.5n(current),0) | 동일 |

원본 SAC도 actor3e-4/critic5e-4, AdamW, tau.05다. 현재 MFPO adapter의 범용 설정은 actor/critic3e-4, Adam, tau.005다. 이 값들은 구현 설정 비교이며 모든 알고리즘이 현재 실행 중이라는 뜻은 아니다.

τ는 학습률이 아니라 `target=(1−τ)target+τ*online`의 혼합 계수다. 일정한 source parameter 변화를 가정한 EMA 반감기는 .005에서약138 updates, .05에서약13.5 updates다. 같은8/256 업데이트 비율로 환산하면약4,425/432 interactions다. 더 빠른 target 갱신은 정보 전달을 빠르게 하는 후보지만 안정성에도 영향을 주므로, 이것이 현재 탐색 실패의 원인으로 확정되지는 않았다.

또 원본 DIPO/DDiffPG의 `action_lr=.03`, `diffusion.update_times=20`은 **replay의 target action 자체를 Q-gradient로 개선하는 내부 최적화**다. Actor network LR가.03이라는 뜻이 아니다. 이 action optimizer의 epsilon은1e-5이며, 위 표의 actor/critic network optimizer epsilon1e-8과 구분해야 한다. 이 내부 action 개선과 OptiQ의 teacher-weighted NLL 한 번을 학습률 하나로 동등하게 비교할 수 없다.

## 실제 OptiQ 업데이트 크기

별도 forward/backward나 optimizer step 없이 저장된 Adam m/v/count로 `−lr*m_hat/(sqrt(v_hat)+1e-8)`을 복원했다. 아래는 파라미터 반올림 전 마지막 nominal update의 float64 재구성이며, 현재 한 batch의 gradient norm 실측값이 아니다. 실험 로그에는 OptiQ actor/critic gradient norm이 기록되어 있지 않다.

| Maze / checkpoint | Actor update 좌표 RMS | Critic update 좌표 RMS | Critic sqrt(v_hat)<ε인 좌표 비율 |
|---|---:|---:|---:|
| v1 /3.00M | 6.31e-5 | 3.36e-5 | 23.7% |
| v2 /3.00M, 첫 도달 후 | 1.23e-4 | 6.12e-5 | 0% |
| v3 /4.00M | 7.07e-5 | 2.89e-5 | 31.6% |
| v4 /4.00M | 6.86e-5 | 2.46e-5 | 33.8% |

업데이트가0에 가까워 학습이 완전히 멈췄다는 증거는 없다. 동시에 작은 critic gradient의 일부는 ε 크기 이하이므로, Adam이 보상 스케일을 완벽하게 상쇄한다는 주장도 맞지 않는다. 이 수치만으로 업데이트 크기가 최적인지 또는 부족한지는 판단할 수 없다. 업데이트 방향에 탐색에 유용한 신호가 있는지가 별도 문제다.

이전 sparse DIPO의 수치 오류 직전 마지막100k 로그에서 actor global gradient norm 평균은 v1–v4 약.098–.121, critic은.00110–.00355였다. 원본 clip=1보다 작아 이 구간 평균에서는 clipping이 지속적으로 작동하는 상태가 아니었다. 현재 OptiQ의 저장된 gradient second-moment RMS와는 측정 정의·모델크기·loss·시점이 달라 직접 배수 비교하지 않는다. DIPO는 BCE/C51 critic, OptiQ는 scalar TD loss라는 차이도 있다. DIPO source는 이전 `0ebd8d26c711d79723f457343b36eb8788bd87b8`이며 해당 실행은 수치 오류로 조기 종료됐으므로 완주 결과가 아니다.

## 작은 reward에 대한 해석

첫 goal 이전 외재보상은0이며 NovelD minibatch 평균은 약.004였다. 이때 Q 평균 약.4는 `reward/(1−.99)` 수준과 맞지만, 이 단순 근사만으로 critic의 정확성을 검증할 수는 없다. 작은 Q 차이가 실제 미래 novelty가 비슷하기 때문인지 critic이 차이를 놓쳐서인지 먼저 구분해야 한다. 작은 보상에 학습률을 곱한 값이 곧 Adam 업데이트 크기라는 계산은 부정확하다.

최신 v2는3,008,256 interactions를 완료했고 학습 중627번 goal 도달을 기록했다. 최종 random-start direct-policy는82/100, mu-only native는83/100이며 성공은 모두goal2였다. 같은 LR와 T=.01에서 첫 도달 후 성능이 크게 바뀌었으므로, 이 seed에서는 첫 goal을 발견하는 단계와 발견 후 보상 학습을 나눠 봐야 한다. 이것은 여러 goal/path를 유지했다는 증거가 아니다. v1 direct-policy는0/100이며 환경별 차이가 남아 있다.

## 다음에 비교한다면

actor LR를 우선 크게 올릴 근거는 약하다. 원본과 가까운 설정을 대조하려면 **현재값**, **critic LR만5e-4**, **τ만.05** 세 조건으로 먼저 나누는 편이 원인 파악에 낫다. 필요하면 두 변경을 함께 넣은 네 번째 조건으로 상호작용을 볼 수 있다. NovelD/T/sigma/actor LR는 유지하고 첫 도달 이전 누적공간coverage·통로진입깊이·누적intrinsic return/critic 예측을 비교한다. 이는 제안이며 새 실험을 실행하거나 현재 학습 설정을 수정하지 않았다.

재현 자료: `inspect_adam_steps.py`, `adam-steps.json/log`, `dipo-original-gradient-history.json`, `current-noveld.json`, `v2-after-first-goal-forward.json`.

원본 소스: [공통 LR 설정](https://github.com/supersglzc/ddiffpg/blob/7edd06c4799abbab0f8fa534c21deb56253b018e/ddiffpg/cfg/algo/actor_critic.yaml), [DIPO tau/update 설정](https://github.com/supersglzc/ddiffpg/blob/7edd06c4799abbab0f8fa534c21deb56253b018e/ddiffpg/cfg/algo/dipo_algo.yaml), [AdamW 생성 및 clipping](https://github.com/supersglzc/ddiffpg/blob/7edd06c4799abbab0f8fa534c21deb56253b018e/ddiffpg/algo/ac_base.py), [target action 최적화](https://github.com/supersglzc/ddiffpg/blob/7edd06c4799abbab0f8fa534c21deb56253b018e/ddiffpg/algo/dipo.py).
