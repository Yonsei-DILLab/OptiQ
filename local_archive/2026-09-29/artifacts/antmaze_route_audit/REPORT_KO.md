# AntMaze 경로 다양성과 intrinsic reward 점검

현재 네 방법 모두 dense reward에 NovelD intrinsic reward를 더해 학습한다. v3에서 관찰되는 한 경로 집중은 intrinsic reward 누락으로 설명되지 않는다. 이번 진단에서는 학습이나 설정을 바꾸지 않고, 완료된 1M checkpoint로 새 난수를 사용한 rollout 700회를 실행했다.

## 실제 적용된 보상

`r_train = -min_goal_distance(s_next) + 0.01 * max(N(s_next) - 0.5 * N(s), 0)`

`N`은 고정 RND target과 학습하는 predictor 출력 사이의 L2 오차다. Replay에서 learner용 배치를 뽑을 때 intrinsic reward를 계산해 환경 보상에 더한다. 저장 replay에는 원래 환경 보상이 남고, 평가 return에는 intrinsic reward를 넣지 않는다. DACER의 행동 탐색 및 정책 entropy와는 별도다.

v3 OptiQ의 RND predictor는 995,000번 업데이트되었고 파라미터 hash가 변했다. 고정 target의 hash는 동일하다. 네 방법 모두 같은 NovelD 설정(coefficient 0.01, novelty discount 0.5, normalize false)을 사용한다.

| v3, 초기 50k까지 기록된 학습 배치 | NovelD 평균 | 환경 보상 절댓값 평균 | 평균 크기 비율 |
|---|---:|---:|---:|
| OptiQ | 0.006296 | 16.0605 | 0.0392% |
| SAC | 0.005991 | 15.9987 | 0.0374% |
| MFPO | 0.005999 | 16.0286 | 0.0374% |
| MEOW | 0.006327 | 16.0701 | 0.0394% |

이는 저장된 학습 배치 통계의 평균이다. 모든 transition의 평균이나 Q-gradient 기여율이 아니다. 보상의 절대 크기만으로 원인을 확정할 수는 없지만, dense reward와 조합한 현재 NovelD 강도가 탐색을 충분히 유도하는지는 의심할 근거가 있다. 수렴 이후뿐 아니라 초기에도 작은 값이었다.

![보상 성분](noveld-reward-scale.png)

## 새 rollout 결과

각 방법의 training seed 0, 1M checkpoint를 복원했다. 비교마다 위치·자세·속도를 포함한 동일 전체 초기 상태를 사용했고, 기존 평가와 다른 난수열로 100회씩 평가했다. 학습한 모델 상태가 평가 전후 동일함을 검증했다.

| v3 직접 stochastic 정책, 새 100회 | G1 성공 | G2 성공 | 실패 | 성공 경로 수 |
|---|---:|---:|---:|---:|
| OptiQ | 100 | 0 | 0 | 1 |
| SAC | 0 | 0 | 100 | 0 |
| MFPO | 0 | 100 | 0 | 1 |
| MEOW | 75 | 0 | 25 | 1 |

OptiQ의 이 표는 random z와 conditional sigma를 포함한다. MFPO도 Q-best-of-10 선택을 하지 않고 정책에서 직접 뽑는다. 모든 방법에서 추가 DACER 행동 잡음은 없다. 100회 각각의 첫 action과 10번째 step 위치가 서로 달랐고, 초기 상태에서 따로 뽑은 action 1,024개도 각 방법 모두 서로 달랐다. 결정적 action이나 같은 궤적을 복사해서 생긴 결과가 아니다.

MEOW는 기존 평가 90/100, 이번 새 난수 평가 75/100이다. 같은 모델에서 평가 난수에 따른 변동이 관찰되므로 두 결과를 보존한다. 새 결과로 기존 점수를 덮어쓰거나 이를 서로 다른 training seed처럼 취급하지 않는다. SAC는 성공하지 못했으므로 '성공 mode 하나로 collapse했다'고 표현하지 않는다.

![새 v3 rollout](v3-new-policy-rollouts.png)

OptiQ v3를 더 분리하면 random-z mu-only는 G1 99/100, episode 동안 z를 고정한 mu-only는 G1 92/100이었다. 어느 쪽도 G2 성공은 없었다. 따라서 이번 checkpoint에서는 sigma 제거 또는 z의 시간적 고정만으로 두 번째 목표가 나타나지 않는다. Held-z는 시간적 sampling 방식을 바꾼 진단이며 원래 정책의 공식 평가 점수를 대체하지 않는다.

![OptiQ sampling 진단](optiq-v3-sampling-diagnostic.png)

v1 OptiQ 대조에서는 새 100회 중 위 경로 71회, 아래 경로 28회, 실패 1회였다. 같은 평가 절차로 두 경로를 검출한다. v3와 v1은 서로 다른 환경과 학습된 정책이다.

![v1 두 경로](optiq-v1-two-route-control.png)

## 학습 중에도 다른 목표를 발견했는가

각 v3 replay의 1M transition 전체를 점검했다. 성공한 세 방법 모두 선택하지 않은 목표의 4m 이내에 들어간 transition이 없었다.

| v3 원래 학습 기록 | G1 성공 episode | G2 성공 episode | 미사용 목표까지 최소 거리 |
|---|---:|---:|---:|
| OptiQ | 3,700 | 0 | G2: 6.20m |
| MFPO | 0 | 7,180 | G1: 13.49m |
| MEOW | 975 | 0 | G2: 11.99m |

SAC는 학습 중 두 목표 모두 성공이 없었다. 최소 거리는 G1 4.16m, G2 13.99m였다.

![학습 방문 분포](training-coverage-early-late.png)

현재 증거는 두 목표를 모두 성공적으로 학습한 뒤 하나를 잃었다기보다, 탐색과 학습이 한쪽 목표에 집중되었다는 해석을 지지한다. 가까운 어느 목표에 도달해도 목적을 만족하는 보상과, 상대적으로 작은 NovelD 보너스가 다른 목표를 계속 탐색할 유인을 충분히 제공하지 못했을 가능성이 있다. 다만 보상 계수, 초기 탐색, critic 오차, 정책 표현 중 어느 요소가 원인인지는 이번 관찰만으로 분리할 수 없다. Intrinsic reward 또는 action entropy가 켜져 있다는 사실만으로 목표·경로 다양성이 보장되지 않는다.

모든 결과는 방법별 training seed 하나다. 여러 방법의 서로 다른 성공 목표를 합쳐서 하나의 multimodal policy처럼 해석하지 않는다. 학습·보상·NovelD 계수·생산 큐는 변경하지 않았다.

## 검증과 보존

- 학습 frozen source: `19fc37a7e51be2bb54d41ab9a980225ecd9f5ba5`.
- 별도 inference 진단 source: `642327b6c706ec78c35098eeb866d5758bd3377e`.
- [검증 JSON](rerollout-verification.json): 총 700회, checkpoint SHA와 검증된 원본 archive 일치, NPZ SHA, reward·success·route 재계산, 초기 전체 상태 동일성, 모델 무변경 확인.
- [원래 학습 방문 통계](original-training-audit.json), [학습 NovelD 로그](original-noveld-metrics.json), [보상 크기 요약](original-noveld-scale-summary.json).
- 원시 trajectory의 xy·observation·action·초기 simulator 상태·평가 seed는 `rerollouts/` 아래 보존했다.
