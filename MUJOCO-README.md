# MuJoCo 5개 환경 — OptiQ scalar critic, fixed β=1

브랜치: `mujoco-setting` (`critic-dime-no-anchor` 기반).
**256×3 GELU scalar twin-Q, BN 없음, UTD=1, anchor 포함, 고정 β=1**을 사용한다.
이 문서는 실행 프로토콜이며, 코드 준비만으로 실험이 자동 시작되지는 않는다.

## 1. 환경과 학습 예산

| 순서 | benchmark | Gymnasium 환경 | seed당 steps | seeds |
|---|---|---|---:|---:|
| 1 | `hopper` | Hopper-v4 | 1,000,000 | 0, 1, 2 |
| 2 | `walker2d` | Walker2d-v4 | 1,000,000 | 0, 1, 2 |
| 3 | `halfcheetah` | HalfCheetah-v4 | 1,000,000 | 0, 1, 2 |
| 4 | `ant` | Ant-v4 | 1,000,000 | 0, 1, 2 |
| 5 | `humanoid` | Humanoid-v4 | 1,000,000 | 0, 1, 2 |

실험 구성은 **5개 환경 × seeds 0,1,2 = 15개 런, 총 15M 학습 환경 steps**다.
평가용 steps는 별도다. 환경·seed·학습 예산은 위와 같이 명시적으로 적용하며,
이전 비교 프로젝트의 5 seeds나 환경별 1M~5M 예산을 자동으로 가져오지 않는다.
별도 인자 없이 단일 실행하면 Humanoid-v4, seed=0이 선택된다.
3개 seed 실험에는 아래 `--multirun` 또는 전체 실행 루프를 사용한다. DMC 환경이 아니다.

## 2. 공통 하이퍼파라미터

| 항목 | 설정 |
|---|---|
| Actor | 256×3, GELU, one-step implicit policy |
| Critic | 256×3, GELU, scalar twin-Q |
| Critic loss | 두 critic의 TD MSE 합 |
| Critic target | 별도 target critic, target update τ=0.005 |
| TD backup | `min(Q1_target, Q2_target)` |
| BN / batch renorm / layer norm / dropout | 모두 없음 |
| Optimizer | Actor·critic 모두 Adam 기본 계수 (0.9, 0.999) |
| Gradient clipping | Actor·critic 각각 global L2 norm ≤ 2, Adam 적용 전 (`alg.optimizer.ac_grad_norm=2.0`) |
| Learning rate | Actor·critic 모두 0.0003 |
| Discount γ | 0.99 |
| Warmup | Actor·critic 모두 5,000 steps |
| Replay capacity / batch size | 1,000,000 / 256 |
| UTD / actor update 주기 | 환경 1스텝당 업데이트 1회 / 매 업데이트 |
| Actor EMA | 없음 (`policy_tau=1.0`) |
| TD action noise | Truncated Gaussian, σ=0.2, clip=0.5 |
| Entropy 보너스 | 0 |
| Scalar Q clipping | 없음 (`v_min/v_max`는 scalar 경로에서 사용하지 않음) |

## 3. Proposal과 OT

| 항목 | 설정 |
|---|---|
| KDE policy centers N | 16 |
| 센터당 random 후보 | 4 |
| 센터당 전체 후보 | 5 = random 4 + anchor 1 |
| 전체 후보 M / transport 크기 | 80 = random 64 + anchor 16 / 16×80 |
| Sampling | `stratified` |
| Anchor | **포함 (`include_anchor=true`)** |
| Perturbation σ / truncation | 0.2 / 2.5σ = 0.5, action box도 반영 |
| Q-weight temperature | **0.25** |
| Density correction β | **고정 1.0** |
| Adaptive β / ESS 제약 | 사용하지 않음 |
| 후보 Q 평가 | Twin mean |
| Sinkhorn ε / iterations | 0.05 / 30 |
| Transport target / actor loss | 행별 argmax / pointwise MSE |

후보 가중치는 `softmax(Q / 0.25 - log q_KDE)`다.
이는 후보 위에서 β=1 density correction을 적용한다는 의미이며,
유한 후보·OT argmax·actor fitting 이후의 정확한 Boltzmann 복원을 보장하는 주장은 아니다.
Q-weight temperature=0.25와 critic target update τ=0.005는 서로 다른 값이다.

Gradient clipping은 DIPO의 `ac_grad_norm` 방식으로, actor 파라미터 전체와
twin critic 파라미터 전체에 각각 적용한다. Action gradient나 Adam 이후의
parameter update를 clipping하는 것이 아니다. DIPO 기본 상한 2를 모든 환경에
통일하며, DIPO 환경별 예외(Hopper=1, Ant=0.8)는 가져오지 않는다.
`alg.optimizer.ac_grad_norm=null`로 비활성화할 수 있다.

## 4. 평가·로깅·체크포인트

- 평가: 시작 시와 매 5K steps, stochastic policy로 10 episodes.
- 진단 로그: 매 5K steps. 학습 기본 로그는 완료 episode마다 기록.
- 체크포인트: 매 50K steps 및 기존 runner의 초기 학습 체크포인트.
- W&B: online, 기본 프로젝트 `optiq_mujoco_scalar_no_anchor`.
- 기본 출력: `outputs/optiq_mujoco_scalar_no_anchor`.
- 프로젝트·출력 경로의 `no_anchor`는 기존 이름을 유지한 것이며, 현재 `mujoco_setting` 기본값은 anchor=True다.
- 각 실행은 고유 run ID를 추가해 기존 결과를 덮어쓰지 않는다.
- `WANDB_PROJECT` / `WANDB_ENTITY` 환경변수는 W&B 기본값보다 우선한다.

## 5. 실행

기존 [설치·인증 안내](docs/NO_ANCHOR_BASELINE.md)에 따라 환경을 준비한다.
API key는 README나 명령줄에 쓰지 않고 기존 로그인 또는 무시된 `.env`로 관리한다.

```bash
python run_optiq_dime.py --multirun --config-name=mujoco_setting benchmark=hopper seed=0,1,2
python run_optiq_dime.py --multirun --config-name=mujoco_setting benchmark=walker2d seed=0,1,2
python run_optiq_dime.py --multirun --config-name=mujoco_setting benchmark=halfcheetah seed=0,1,2
python run_optiq_dime.py --multirun --config-name=mujoco_setting benchmark=ant seed=0,1,2
python run_optiq_dime.py --multirun --config-name=mujoco_setting benchmark=humanoid seed=0,1,2
```

모든 명령은 기본적으로 β=1을 사용한다. 명시하려면 canonical 설정인
`alg.actor.density_correction_beta=1.0`을 붙인다.
`density_beta`는 이 값의 alias이므로 둘을 서로 다르게 지정하지 않는다.

GPU 한 개에서 총15개 런을 순차 실행하려면 다음을 **사용자가 직접 실행**한다.
실패 시 다음 환경으로 넘어가지 않는다.

```bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
for benchmark in hopper walker2d halfcheetah ant humanoid; do
  for seed in 0 1 2; do
    python run_optiq_dime.py --config-name=mujoco_setting \
      benchmark="$benchmark" seed="$seed" total_steps=1000000 \
      alg.actor.density_correction_beta=1.0
  done
done
```

장기 실행은 서버의 기존 supervisor 등 프로세스 관리 도구로 관리한다.
병렬 실행 시에는 명령별로 비어 있는 GPU를 지정해 GPU당 한 런만 배치한다.

## 6. 확인

2026-09-07 CPU 격리 검증: **37개 테스트 통과** (57.66초).
15개 환경·seed 조합의 설정과, 5개 환경 각각 β=1·256×3 네트워크로
8 environment steps / 6 updates의 smoke 학습을 확인했다.
이전 커밋의 git object가 필요한 역사적 sampling 비교 2개는 임시 폴더에서 제외했다.
이는 실행 가능성 검증이며 1M benchmark 결과가 아니다. GPU sweep과 W&B 업로드는 시작하지 않았다.

학습 없이 상속된 설정을 출력:

```bash
python run_optiq_dime.py --config-name=mujoco_setting benchmark=hopper --cfg job --resolve
```

온라인 W&B 테스트 fixture를 제외한 회귀 테스트:

```bash
python -m pytest --noconftest -q tests/test_scalar_critic.py
```
