# spline_energy

현재 코드는 `legacy`, `gmm_reference`, `gmm_soft` 세 profile을 보존합니다.
GMM40 성공본의 unnormalized energy 표현과 rank 64 / 129 knots를 MuJoCo에
복원했지만, HalfCheetah에서는 안정적인 성능 이전에 실패했습니다. 따라서 현재
결론은 **GMM40 표현력과 one-step sampler는 검증됐지만, raw-energy TD 학습은
MuJoCo용 기본 알고리즘으로 검증되지 않았다**입니다. 수식과 Direct GMM/TRG 대조는
[`DIRECT_ALIGNMENT.md`](algorithms/spline_energy/DIRECT_ALIGNMENT.md)에 기록했습니다.

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
- GMM40 표현을 복원한 conditional 모델: [`raw_energy.py`](algorithms/spline_energy/raw_energy.py)
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

## HalfCheetah 결과와 결론

동일한 `HalfCheetah-v4` 평가 protocol에서 확인한 결과입니다.

| 설정 | 측정 결과 |
|---|---|
| 기존 `legacy`, 3 seeds × 1M | 최종 reward `5842.9`, `5964.6`, `5531.4` |
| raw energy + ordinary TD (`gmm_reference`) | 100k reward `-20.8`, `-258.0`, `29.6`; seed 1 발산 |
| raw energy + exact soft target (`gmm_soft`) | seed 0/2가 30–36k까지 finite였으나 사용자 중단으로 성능 결론 없음 |

ordinary-TD seed 1은 100k에서 Q 평균 `72,228`, gradient norm `3.61e6`,
loss `5.77e7`까지 증가했습니다. 같은 시점의 다른 두 seed loss는 `0.60`,
`0.76`이어서 GPU 공통 오류보다 tied policy/Q 학습의 seed-dependent feedback
실패로 판단했습니다. 해당 배열은 모든 seed의 100k checkpoint를 저장한 뒤
중단했습니다. soft-target 파일럿도 중단했으며 추가 실험은 실행 중이지 않습니다.

GMM40과 MuJoCo의 차이는 다음과 같습니다.

- GMM40은 고정된 oracle target을 넓은 proposal로 직접 질의하므로 target과
  학습 데이터가 움직이지 않습니다.
- MuJoCo는 현재 정책이 replay 데이터를 만들고, bootstrap Q가 다음 target과
  같은 정책을 다시 바꿉니다. 하나의 energy가 Q와 정책을 모두 맡으면 오차가
  `정책 → 데이터 → target`으로 되먹임됩니다.
- `pi=exp(Q/alpha)/Z`는 정확한 Q가 주어졌을 때의 정책 관계입니다. 잘못된 Q도
  정확히 정규화할 수 있으므로 이 식만으로 올바른 Q를 학습하지는 못합니다.
- GMM generalized-KL은 total mass `Z`를 직접 고정하지만, TD는 긴 horizon의
  절대 Q scale을 bootstrap으로 학습합니다. `gamma=.99`, `alpha=.25`에서는 작은
  scale 오차도 raw log-energy와 gradient를 크게 만들 수 있습니다.
- 2D state-free GMM과 달리 HalfCheetah는 상태조건부 6D 행동 분포이므로 replay
  coverage와 factorized mixture 최적화도 더 어렵습니다.

실행 정의는 [`HALFCHEETAH_GMM_REFERENCE.md`](analysis_tools/experiments/20260921_mujoco_spline_energy/HALFCHEETAH_GMM_REFERENCE.md)와
[`HALFCHEETAH_GMM_SOFT.md`](analysis_tools/experiments/20260921_mujoco_spline_energy/HALFCHEETAH_GMM_SOFT.md)에 있습니다.

## 기존 MuJoCo 이식 (`legacy`)

`legacy` 버전은 GMM40의 state-free spline parameter를 2-layer state encoder의
출력으로 바꿉니다. 설정은 rank 16, 33 knots, hidden width 256,
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

W&B는 외부로 전송하지 않고 각 output directory의 offline run으로만 저장합니다.
로컬 project는 `OptiQ-MuJoCo-Spline-Energy`이며 profile별 group과 algorithm
이름을 사용합니다. 주 지표는 10k environment steps마다 기록하는
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

## GMM40 에너지 loss를 잇는 확장

`--loss-kind relative_energy`는 모델을 바꾸지 않고 Huber TD loss만
GMM40 generalized KL에서 유도한 상대 에너지 loss로 교체하는 실험 옵션입니다.
정규화·one-step sampler는 동일합니다. 결정론적 Bellman 해의 보존,
조건부 Q/정책 오차 bound, 확률적 target의 편향, 수치 안정화를
[`THEORY.md`](algorithms/spline_energy/THEORY.md)에 증명과 함께 기록했습니다.
이는 현재 MuJoCo 성능 향상이 검증됐다는 뜻은 아닙니다.

[Ant 100k 파일럿 결과](algorithms/spline_energy/PILOT_20260922.md):
seed 0 최종 보상은 기존 -61.6, 수정본 -34.1이지만 둘 다 음수이며,
수정본의 TD 오차가 더 컸습니다. 개선이 검증된 기본 알고리즘으로 취급하지 않습니다.
