# spline_energy

`spline_energy`는 정책과 soft Q를 하나의 정규화 가능한 positive spline
circuit으로 표현하는 알고리즘입니다. 추론할 때 diffusion, MCMC, action
optimization, candidate search를 사용하지 않고 상태 네트워크를 한 번 통과한
뒤 analytic inverse CDF로 행동을 샘플링합니다.

## 핵심 구조

상태 `s`에서 하나의 네트워크가 `V(s)`, mixture weight `w_r(s)`, action 축별
positive linear spline `f_rd(a_d | s)`를 동시에 출력합니다.

```text
pi(a | s) = sum_r w_r(s) prod_d f_rd(a_d | s)
Q(s, a)   = V(s) + alpha log pi(a | s)
```

각 spline leaf와 mixture weight를 정확히 정규화하므로 normalized action box에서
다음 관계가 성립합니다.

```text
integral exp(Q(s,a) / alpha) da = exp(V(s) / alpha)
```

따라서 별도 actor와 critic 사이를 맞추는 projection 없이 같은 출력에서 Q, V,
policy density, stochastic action을 얻습니다. 행동 샘플링은 root categorical CDF와
선택된 spline leaf의 analytic inverse CDF를 병렬로 계산합니다.

## 코드 위치

- 공용 conditional 모델: [`algorithms/spline_energy/model.py`](algorithms/spline_energy/model.py)
- GMM40 성공본: [`benchmarks/gmm40/spline_energy/`](benchmarks/gmm40/spline_energy/)
- GMM40 핵심 회로: [`benchmarks/gmm40/spline_energy/spline_energy.py`](benchmarks/gmm40/spline_energy/spline_energy.py)
- 실제 100k 비교 runner: [`benchmarks/gmm40/spline_energy/reference_100k_compare_latest.py`](benchmarks/gmm40/spline_energy/reference_100k_compare_latest.py)
- GMM40 결과와 provenance: [`benchmarks/gmm40/spline_energy/RESULTS.md`](benchmarks/gmm40/spline_energy/RESULTS.md)
- MuJoCo 학습 코드: [`analysis_tools/experiments/20260921_mujoco_spline_energy/`](analysis_tools/experiments/20260921_mujoco_spline_energy/)

## GMM40에서 측정된 결과

state-free 2D GMM40 실험은 rank 64, leaf당 129 knots, Adam `3e-4`, 5 seeds,
100k updates로 실행했습니다. 아래 값은 저장된 최종 결과의 5-seed 평균과 표준편차입니다.

| 지표 | 결과 |
|---|---:|
| mode coverage | 모든 seed에서 40/40 |
| target 3-sigma 내부 비율 | 99.228% ± 0.033% |
| forward KL | 0.003256 ± 0.001097 |
| reverse KL | 0.003295 ± 0.001152 |
| component-mass TV | 0.02598 ± 0.00272 |
| MMD² | 0.0002595 ± 0.0000196 |

모든 seed가 10k update에서 3-sigma mass 90%, 15k에서 95%를 넘었습니다.
98%는 세 seed가 20k, 나머지 두 seed가 25k에서 처음 넘었습니다. 이 값들은
GMM40 toy에서 측정한 결과이며 MuJoCo 성능을 의미하지 않습니다.

## MuJoCo 이식

MuJoCo 버전은 GMM40의 state-free spline parameter를 2-layer state encoder의
출력으로 바꿉니다. 현재 설정은 rank 16, 33 knots, hidden width 256,
`alpha=0.25`이며 Hopper-v4, Walker2d-v4, HalfCheetah-v4, Ant-v4,
Humanoid-v4에서 seed 0, 1, 2를 평가하도록 고정했습니다.

학습 target은 EMA copy가 제공하는 soft Bellman value입니다.

```text
y = r + 0.99 (1 - terminal) V_target(s')
loss = Huber(Q(s,a) - y)
```

EMA는 TD target 안정화를 위한 동일 circuit의 지연 복사본입니다. 별도 actor,
Gaussian sigma, tanh action compression, reward scaling, observation normalization,
prioritized replay는 사용하지 않습니다. 환경의 실제 action bound와 `[-1,1]`
사이는 affine map으로 변환합니다.

W&B project는 `gsmin2018` 계정의 personal entity인
`models/OptiQ-MuJoCo-Spline-Energy`이며, group과 config의 algorithm
값은 모두 `spline_energy`입니다. 주 지표는 10k environment steps마다 기록하는
`eval/mean_reward`입니다.

## 검증

GMM40 analytic integral, inverse-CDF sampler, gradient 검증:

```bash
cd benchmarks/gmm40/spline_energy
JAX_PLATFORMS=cpu python validate_spline_energy.py
```

Conditional 모델의 normalization, sampling, TD gradient, 다섯 환경의 action
bound 검증:

```bash
cd analysis_tools/experiments/20260921_mujoco_spline_energy
JAX_PLATFORMS=cpu python validate.py
```

GMM40의 측정 결과는 표현력과 one-shot sampler를 검증합니다. 고차원
state-conditioned TD 학습의 성능은 별도의 MuJoCo 실험 결과로 판단해야 합니다.
