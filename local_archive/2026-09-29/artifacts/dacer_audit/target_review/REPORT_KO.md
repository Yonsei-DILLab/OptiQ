# OptiQ DACER target entropy 검토 — 2026-09-21

결론: 현재 확인한 정책에서 H_target=-0.9×action_dim은 추가 탐색을 적극 늘릴 기준으로 낮다. 분포가 좁아졌을 때 개입하는 하한으로는 가능한 값이지만, 성능 최적값으로 검증된 것은 아니다.

## 진단 방법

- 학습은 수행하지 않았다. 실행 중 소스, 하이퍼파라미터, 큐를 변경하지 않았다.
- source: 84f1e0a884349d6c4b0dae521839a8d4e5f46437. Ant T=.05/.1 및 Humanoid T=.1/.5, seed 0의 actor checkpoint를 CPU에서 읽었다.
- 5,001 / 100,000 / 1,000,000 step checkpoint를 각 run의 저장된 probe state 32개에서 비교했다.
- 추가로 각 1M checkpoint의 full policy(random z + truncated Gaussian)를 실행해 1,200 step 동안 방문한 state 중 균등 간격 32개를 선택했다. 아래 표는 이 방문 상태 결과다. Replay 전체의 상태 분포와는 다르다.
- 상태당 200개 action, K=3/full-covariance GMM, random_state=42, 기본 reg_covar=1e-6. 현재 DACER regulator와 동일한 proxy 공식을 사용했다.
- 별도 추가 noise std=0, 초기값(Ant .027 / Humanoid .0405), .1을 비교했다. 각 상태에서 같은 base action 및 표준정규 draw를 재사용했다.
- 정책 모수의 평균 sigma는 truncation 전 scale parameter이며, 실제 행동의 표준편차와 다르다.

## 1M checkpoint 방문 상태 결과

| 환경 | T | 추가 noise 없음 H/d | 초기 추가 noise 적용 H/d | actor 평균 sigma | H/d < -.9인 상태 비율 |
|---|---:|---:|---:|---:|---:|
| ant | 0.05 | -0.692 | -0.662 | 0.114 | 6.25% |
| ant | 0.1 | -0.351 | -0.337 | 0.170 | 0.00% |
| humanoid | 0.1 | +0.269 | +0.276 | 0.359 | 0.00% |
| humanoid | 0.5 | +0.462 | +0.464 | 0.366 | 0.00% |

각 32-state 평균은 모두 -.9보다 높다. 따라서 이 배치에서 gradient H-H_target은 양수이며 log alpha를 낮추는 방향이다. 이는 실제 DACER를 학습한 결과가 아니라, DACER 없이 학습한 checkpoint에 regulator의 측정을 적용한 반사실적 진단이다. Adam의 실제 매 step 변화는 이전 momentum에도 의존한다.

## 수치 해석

단일 비절단 isotropic Gaussian의 h/d = log(sigma) + 0.5 log(2πe)를 기준으로 하면:

| H_target/d | 등가 sigma |
|---:|---:|
| -0.9 | 0.0984 |
| -0.5 | 0.1468 |
| -0.3 | 0.1793 |

이는 수치를 이해하기 위한 환산일 뿐, 실제 TRG mixture의 entropy와 sigma가 일대일 대응한다는 뜻은 아니다. 현재 초기 sigma=e^-1=.3679이고 중심 0인 [-1,1] truncated normal의 entropy는 차원당 약 +.385다. Latent로 인한 다양성과 상태별 중심의 경계 근접성도 실제 entropy에 영향을 준다.

## 추정 오차 및 적용 범위

Proxy H(component label)+E[H(component)]는 fitted GMM marginal entropy의 상한이다. 실제 clipped behavior distribution의 정확한 entropy는 아니다. K=3일 때 fitted marginal과의 차이는 0~log(3) nats이나, GMM fitting 오차와 clipping의 경계 원자는 별개다.
모든 측정 GMM은 수렴했다. 표본 수를 늘린 민감도 확인:

- ant T=0.05: 동일한 8개 상태에서 200개→2,000개 action 시 H/d -0.716→-0.606.
- ant T=0.1: 동일한 8개 상태에서 200개→2,000개 action 시 H/d -0.346→-0.256.
- humanoid T=0.1: 동일한 8개 상태에서 200개→2,000개 action 시 H/d +0.298→+0.382.
- humanoid T=0.5: 동일한 8개 상태에서 200개→2,000개 action 시 H/d +0.486→+0.595.

200→2,000 action 비교는 같은 상태지만 독립 action sample이다. 이번 표본에서 평균이 약 .08~.11/d 증가했으므로 target 간격 .1 정도의 차이는 추정 해상도와 함께 봐야 한다. 이 체크가 실제 entropy의 정확도를 보장하지는 않는다.

## 판단 및 다음 비교 제안

- Ant: -.9는 보수적인 collapse 방지 기준. T=.05에서 -.5, T=.1에서는 -.3 같은 더 높은 목표가 추가 noise를 늘릴 가능성이 있다. -.9/-.5/-.3 비교는 가설 검증용 후보이며 최적값 주장이 아니다.
- Humanoid: H/d가 이미 +.27~+.46이며 actor sigma도 상한 .368 부근이다. -.9를 -.5나 -.3으로 바꿔도 이번 측정에서는 gradient 부호가 바뀌지 않는다. 전체 entropy 부족이라는 설명은 이 진단으로 지지되지 않는다. 목표를 양수로 올려 강제로 noise를 키우는 것이 reward 개선을 뜻하지 않는다.
- 현재 scalar alpha는 배치 평균을 보고 수집용 noise만 조절한다. 개별 state나 mode의 collapse, actor sigma, teacher 혹은 TD target을 직접 제어하지 않는다.
- 논문 baseline 재현을 위한 -.9는 유지할 수 있다. OptiQ에서 성능 향상을 노린다면 위 후보를 별도 ablation으로 평가하고, 기존 사용자 기준인 stochastic_z 마지막 100k reward의 5-seed 평균으로 판단해야 한다.
- 본 검토에서는 훈련 실험이나 코드 변경을 하지 않았다. seed 0의 제한된 checkpoint/state 진단으로 최적 target이나 성능 인과를 확정할 수 없다.

출처: [DACER Table 3 및 §4.3](https://arxiv.org/html/2405.15177v4#A1.T3), 현재 regulator.py, frozen checkpoint와 본 폴더 ant.json/humanoid.json/probe_entropy.py.
