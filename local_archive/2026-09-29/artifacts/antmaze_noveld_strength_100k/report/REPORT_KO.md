# AntMaze v3 OptiQ NovelD coefficient: 100k

학습 소스: `262a10d6280eb6e79eec24e6170541a7b0f58e72`. 각 계수당 학습 seed 0 하나. 100k 환경 step, 95k learner/RND update.
기본값은 0.01을 유지하고 이번 실행만 계수 0.1, 1, 5, 10을 적용했다. 나머지 설정 및 초기 모델/RND는 동일하다.

고정된 동일 전체 초기 상태에서 각 정책 100회 평가. native는 random-z μ-only, policy는 random-z + conditional sigma다. 평가에 외부 DACER 잡음/NovelD 보상을 넣지 않는다.
natural/fixed는 이 환경에서 같은 시작 분포이므로 합쳐 200개의 독립 초기 상태처럼 해석하지 않는다.

| 계수 | 학습 G1/G2 도달 | 최초 G1/G2 step | 학습 최소 G1/G2 거리(m) | μ-only 성공 | full-policy 성공 | full-policy G1/G2 | 방문 bin |
|---:|---:|---|---|---:|---:|---|---:|
| 0.1 | 0/0 | None/None | 3.35/14.14 | 0% | 0% | 0%/0% | 360 |
| 1 | 0/0 | None/None | 1.58/13.96 | 0% | 0% | 0%/0% | 385 |
| 5 | 0/0 | None/None | 0.65/11.87 | 6% | 2% | 2%/0% | 598 |
| 10 | 0/0 | None/None | 2.51/7.92 | 0% | 0% | 0%/0% | 901 |

실패 궤적도 회색으로 전부 표시했다. 성공이 없는 경우 경로 다양성을 확보했다거나 mode collapse로 단정하지 않는다.
100k는 초기 탐색 비교다. 단일 seed이며 이 예산에서 실패했다고 이후 학습 불가능을 의미하지 않는다.
각 실행의 전체 100k replay 환경보상, checkpoint SHA256/full-state digest, 최종 600개 원시 rollout, 업데이트 수를 검증했다.

## 실제 intrinsic 보너스 (마지막 25k, 매 1k 기록된 minibatch 평균들의 평균)

| 계수 | intrinsic | environment |
|---:|---:|---:|
| 0.1 | 0.0452 | -12.4580 |
| 1 | 0.4430 | -11.8563 |
| 5 | 2.7356 | -13.0593 |
| 10 | 6.1958 | -13.5669 |
