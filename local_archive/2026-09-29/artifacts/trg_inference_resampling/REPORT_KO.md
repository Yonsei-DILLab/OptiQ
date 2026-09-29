완료된 T=.25, beta=1, DACER=true의 최종 1M actor와 critic을 재학습 없이 비교했다. HalfCheetah 5개 학습 seed, Ant 4개 학습 seed마다 동일한 새 reset/policy seed 20개로 평가했다. 기존 마지막100k 평균과는 별도의 최종 체크포인트 재평가다.

| 환경 | 평가 | return 평균 ± 학습 seed SD | 기존 대비 Δ | ESS / 후보 수 |
|---|---|---:|---:|---:|
| halfcheetah | mu_one | 10429.4 ± 812.8 | +0.0 | 1.00 / 1 |
| halfcheetah | mu_q64 | 10596.7 ± 794.6 | +167.2 | 40.66 / 64 |
| halfcheetah | mu_kde_is64 | 10189.8 ± 774.5 | -239.6 | 8.60 / 64 |
| ant | mu_one | 5221.2 ± 228.8 | +0.0 | 1.00 / 1 |
| ant | mu_q64 | 5387.9 ± 81.4 | +166.7 | 24.98 / 64 |
| ant | mu_kde_is64 | 5168.1 ± 65.7 | -53.1 | 8.21 / 64 |

![결과](comparison.png)

`mu_one`: 기존 stochastic_z와 같은 μ-only sampler. `mu_q64`: μ 64개를 softmax(Q/.25)로 categorical 재샘플링. `mu_kde_is64`: 동일 후보를 softmax(Q/.25-log q_hat_mu)로 재샘플링. Q는 각 학습 설정과 동일하게 current twin critic 평균이다. 세 방식 모두 조건부 σ와 DACER 행동잡음을 넣지 않는다.

μ 분포의 정확한 밀도는 직접 계산할 수 없어, 독립 μ 256개로 diagonal Scott bandwidth의 box-normalized KDE를 추정했다. bandwidth floor는 정규화 action 단위 .001이다. 학습 σ를 μ 분포 밀도로 대신 사용하지 않았다. 이는 근사 importance resampling이며, Q-only 방식은 일반적으로 q_mu(a)exp(Q/T)에 비례하는 쪽으로 재가중한다. reward가 높아져도 실제 exp(Q/T) 분포에 가까워졌다는 증거는 아니다. KDE 추정 오차와 critic 오차에 민감하다.

기존 sample_action과 baseline 수치 일치, σ head 변경에 대한 세 방식 불변성, 모든 입력 checkpoint/config 해시 보존, 평가 전후 actor/critic 상태 불변성, finite action 및 가중치를 검증했다. 1M 체크포인트 source를 각 run별로 로드했다. 20개 episode는 반복 학습 seed 20개가 아니다.

평가 commit: `3524ea56d389733e4be97b645804d282971245b9`. 실행 환경·checkpoint SHA256와 seed별 모든 return/length는 [manifest](manifest.json) 및 results/*.json에 보존했다.

| 환경 | seed | 기존 μ | Q64 | 근사 IS64 | Δ Q64 | Δ IS64 |
|---|---:|---:|---:|---:|---:|---:|
| ant | 0 | 4893.6 | 5269.0 | 5113.6 | +375.4 | +220.0 |
| ant | 1 | 5238.9 | 5403.6 | 5110.7 | +164.6 | -128.3 |
| ant | 2 | 5402.3 | 5447.3 | 5238.2 | +45.0 | -164.1 |
| ant | 3 | 5350.0 | 5431.8 | 5209.9 | +81.8 | -140.2 |
| halfcheetah | 0 | 10141.7 | 10398.4 | 9933.0 | +256.7 | -208.6 |
| halfcheetah | 1 | 9681.4 | 9789.6 | 9378.3 | +108.3 | -303.1 |
| halfcheetah | 2 | 11820.4 | 11927.1 | 11461.6 | +106.7 | -358.8 |
| halfcheetah | 3 | 10287.6 | 10512.1 | 9952.5 | +224.5 | -335.0 |
| halfcheetah | 4 | 10216.1 | 10356.1 | 10223.7 | +140.0 | +7.6 |
