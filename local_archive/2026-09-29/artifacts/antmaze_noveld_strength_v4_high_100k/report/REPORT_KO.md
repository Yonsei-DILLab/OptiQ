# AntMaze v4 OptiQ NovelD coefficient: 100k

학습 소스: `edde403369ed87a90d497c307379e99881d5faab`. 각 계수당 학습 seed 0 하나. 100k 환경 step, 95k learner/RND update.
기본값은 0.01을 유지하고 이번 실행만 계수 50, 100을 적용했다. 나머지 설정 및 초기 모델/RND는 동일하다.

고정된 동일 전체 초기 상태에서 각 정책 100회 평가. native는 random-z μ-only, policy는 random-z + conditional sigma다. 평가에 외부 DACER 잡음/NovelD 보상을 넣지 않는다.
natural/fixed는 이 환경에서 같은 시작 분포이므로 합쳐 200개의 독립 초기 상태처럼 해석하지 않는다.

| 계수 | 학습 G1/G2 도달 | 최초 G1/G2 step | 학습 최소 G1/G2 거리(m) | μ-only 성공 | full-policy 성공 | full-policy G1/G2 | 방문 bin |
|---:|---:|---|---|---:|---:|---|---:|
| 50 | 0/0 | None/None | 5.10/3.75 | 0% | 0% | 0%/0% | 511 |
| 100 | 0/0 | None/None | 6.70/8.06 | 0% | 0% | 0%/0% | 431 |

실패 궤적도 회색으로 전부 표시했다. 성공이 없는 경우 경로 다양성을 확보했다거나 mode collapse로 단정하지 않는다.
100k는 초기 탐색 비교다. 단일 seed이며 이 예산에서 실패했다고 이후 학습 불가능을 의미하지 않는다.
각 실행의 전체 100k replay 환경보상, checkpoint SHA256/full-state digest, 최종 600개 원시 rollout, 업데이트 수를 검증했다.

## 실제 intrinsic 보너스 (마지막 25k, 매 1k 기록된 minibatch 평균들의 평균)

| 계수 | intrinsic | environment |
|---:|---:|---:|
| 50 | 33.7082 | -15.7851 |
| 100 | 63.5240 | -15.5521 |
