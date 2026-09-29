# 최신 heejoon 3-mode 재현: 64×64, batch 32

2026-09-21 서버 `vast-heechan-180`의 `/home/heechan/OptiQ-heejoon`에서 `git pull --ff-only origin heejoon`을 수행했다. 상류 소스 커밋은 `6a06acabf03005d1a15edb646f59d6278821e98e`이며, 해당 소스를 freeze.py로 고정하고 SHA256 manifest를 검증했다. 재현·초기화 대조 실행 코드는 실행 전에 `heejoon`에 `cd082d9a1e6d2bc03fb4a8d13f6fcb9b852b9730`으로 커밋했다. 이 로컬 실험 커밋은 push하지 않았다.

## 결론

기본 초기화는 네 seed 모두 20k와 100k에서 넓은 단봉 분포에 머물렀다. "두 모드만 학습한다"는 형태는 이번 재현에서는 나오지 않았다. 평균 출력층 초기화만 키우면 같은 64×64·batch32에서 네 seed 모두 세 모드를 학습했다. 이 결과는 64×64의 표현력 한계보다 초기화에 민감한 최적화 실패를 지지한다. 다른 환경이나 TRG 구현 전체에 대한 보장은 아니다.

| mean_output_init_scale | 초기 σ | 20k 세 모드 성공 | 평균 TV ↓ | 100k 세 모드 성공 | 평균 TV ↓ |
|---|---:|---:|---:|---:|---:|
| 기본 1e-4 | 0.5 | 0/4 | 0.36620 | 0/4 | 0.36625 |
| 대조 1.0 | 0.5 | 4/4 | 0.07246 | 4/4 | 0.04431 |

## 어떤 초기화를 바꿨는가

행동 분포의 σ를 바꾼 것이 아니다. 조건부 평균 μ(s,z)를 출력하는 마지막 Dense 층의 variance-scaling initializer scale을 1e-4에서 1.0으로 변경했다. 가중치 표준편차와 latent별 초기 μ의 표준편차는 100배 커진다. μ 값을 1.0으로 고정하거나 초기 모드 위치를 지정하지 않았다. 매 seed에서 Gaussian log_sigma 배열이 두 조건 간 완전히 동일함을 확인했다.

| Seed | 기본 초기 μ의 표준편차 | 큰 초기화의 μ 표준편차 |
|---|---:|---:|
| 0 | 0.00044295 | 0.0442949 |
| 1 | 0.00048577 | 0.0485772 |
| 2 | 0.00054579 | 0.0545794 |
| 3 | 0.00030436 | 0.0304360 |

위 표는 고정된 평가 latent 2,048개에서 계산했다. 초기 σ=0.5가 크더라도 μ가 거의 동일하면 서로 다른 Gaussian 성분이 생기는 것은 아니다. 거의 동일한 성분의 responsibility와 평균 gradient가 유사하게 움직여 분화가 지연될 수 있다는 해석과 일치하지만, 원인 전체를 증명한 것은 아니다.

## 재현 조건

- 최신 `experiments/gmm_mode_gradient_batch32`의 baseline training engine을 수정 없이 사용했다. known-mode routing 기법은 적용하지 않았다.
- Fresh z: 각 업데이트에서 32개 독립 그룹, 그룹마다 latent64 / 행동후보64. 각 그룹 안에서 importance weight를 정규화하고 gradient를 평균내어 Adam을 한 번 갱신했다.
- 1D 고정 상태 / 고정 analytic Q. Target은 중심 -0.6,0,0.6, std0.1, 동일 질량의 Gaussian mixture를 [-1,1]로 조건화한 분포다. Q=.25log f, T=.25.
- Actor256×256 GELU, 1D normal latent, tanh-squashed Gaussian, 초기 σ=.5, logσ[-5,1], proposal floor .05, Adam3e-4. 현재 direct-gmm-trg의 truncated Gaussian 구현과 다르다.
- Seeds0–3. 20k에서 결과를 저장하고 동일 checkpoint/Adam/RNG를 이어 100k까지 학습했다. 초기화 외의 학습 코드는 동일하다.
- 기존 runner의 반사실 gradient-routing 진단만 생략했다. 이 진단은 학습 상태를 바꾸지 않는다. 학습 엔진, 초기화 seed, RNG 분할, 평가 RNG는 상류 코드와 같다.
- 각 checkpoint에서 실제 정책 행동 32,768개를 평가했다. 추가 Gaussian 잡음을 포함한 학습 정책 자체의 평가다.

## 모드 판정과 결과

히스토그램의 무작위 잔봉우리를 세지 않았다. 고정 독립 latent2,048개에서 계산한 각 조건부 tanh-Gaussian의 density를 평균내어 정책 밀도를 근사했다. KDE는 사용하지 않았다. 국소 극대점 prominence≥0.05, 최소 간격0.10, target 중심 ±0.15 이내 봉우리 존재 여부로 판정했다. 이는 전체 연속 latent density의 Monte Carlo 근사이지 정확한 전역 모드 수 증명은 아니다.

| Seed | 기본 100k TV | 큰 초기화 100k TV | 큰 초기화 peak 위치 |
|---|---:|---:|---|
| 0 | 0.37302 | 0.04271 | -0.5984, 0.0050, 0.6044 |
| 1 | 0.36222 | 0.04266 | -0.6004, -0.0030, 0.5934 |
| 2 | 0.36368 | 0.04191 | -0.5904, -0.0010, 0.5964 |
| 3 | 0.36608 | 0.04998 | -0.6064, -0.0090, 0.6144 |

기본 조건도 세 basin에 약30%/40%/30%의 샘플이 있지만, 세 개의 봉우리를 형성하지는 않았다. 따라서 basin 방문과 multimodal fitting 성공을 구분해야 한다.

## 검증 및 재현 파일

상류 전체 validator는 JIT/비JIT gradient의 오차0 검증에서 최대절대오차1.12e-10으로 실패했다. 학습 코드를 바꾸지 않고 요청한 baseline64×64만 별도로 검사했다. 32개 그룹의 명시적 gradient 평균, microbatch1/4/32, Adam1회, RNG 정확 일치, 단일그룹 gradient 비교가 atol3e-6/rtol3e-4 내에서 통과했다. 이 검사의 마지막 결과파일 쓰기는 runtime 디렉터리 부재로 실패했으며, 완료된 assertion들을 `BASELINE_VALIDATION.json`에 사후 기록했다. 개별 최대오차 수치는 보존되지 않았다.

모든8개 run의 COMPLETE와100k 결과, 소스 커밋 일치를 확인했다. 결과는 `runs/`, 요약은 `summary_20k.json` 및 `summary_100k.json`, 그림은 `comparison_20k.png` 및 `comparison_100k.png`다. 분석 코드는 `analyze.py`이며 최종 checkpoint를 포함한 원본은 `results_100k.tar.gz`에 보존했다.

원격 결과: `/home/heechan/optiq-experiments/three-mode-recheck-64x64-b32-20260921/results`.
