# Best-of-8 winner 분포를 OT로 증류

Profile `mujoco_v5_bestof8` / `v5/bestof8`, branch `v5_bestk`.
2026-09-15 사용자 요청으로 collection-only 실험에서 winner distillation으로
변경했다. 이전 collection-only 코드는 commit `2cd0532`와 기존 캠페인의
frozen source에 보존한다. Canonical `mujoco_v5` / branch `v5`는 그대로다.

## 목표와 업데이트

현재 full Gaussian actor를 π라 하면 teacher는 각 상태에서 π의 독립적인
행동 8개 중 live min(Q1,Q2)가 가장 높은 행동의 분포 β8이다.
연속 Q 분포에서 β8(a|s) = 8 π(a|s) F_Q(Q(s,a)|s)^7이다.
구현은 이 밀도를 계산하지 않고 β8에서 직접 샘플링한다. 동점은 각 묶음의
첫 argmax를 택한다(수집과 동일).

1. Student는 기존처럼 독립적인 z 16개에 대한 tanh(μ(s,z))를 사용한다.
2. Teacher는 상태마다 **64개의 독립적인 묶음 × 8개 후보 = 512개**의
   full policy 행동을 생성한다. 모든 후보는 새 continuous z와 Gaussian ε를
   사용한다. 기존 16개 student component를 재사용하지 않는다.
3. 각 묶음에서 live twin-min winner 하나를 선택한다. 전체 512개에서 상위
   64개를 고르는 방식이 아니다. Teacher σ floor나 추가 noise는 없다.
4. Student 16개에 각각 1/16, winner 64개에 각각 **1/64**의 질량을 부여한다.
   Squared action distance, Sinkhorn ε=.1, 100회, cost normalization=false.
5. Teacher의 원래 pre-tanh u와 OT coupling을 stop-gradient한다.
   모든 OT row의 조건부 Gaussian NLL로 μ와 σ를 함께 업데이트한다.
   Argmax teacher 하나나 barycenter로 손실을 대체하지 않는다.

`exp(Q/T)/q` 가중치는 제거한다. 이미 선택된 winner에 이를 다시 적용하면
β8과 다른 목표가 된다. **온도 T는 사용하지 않으며 temperature=null**이다.
따라서 이전 Ant T=.25 / Hopper T=.01을 새 실험의 활성 하이퍼파라미터로
표시하지 않는다. `density_correction=false`, beta=0, teacher_std_floor=0.
이 설정에 활성 온도·밀도 보정·teacher K 불일치를 넣으면 시작 전에 거부한다.

## 수집, TD, 평가

- Warmup 5K uniform 수집 이후 **Kb=8**, live twin-min 선택. Replay에는 실제
  실행한 정규화 행동을 저장한다. 별도 collector RNG는 이전 구현과 동일하다.
- **Kt=1**: TD target은 current actor의 단일 full Gaussian 행동과 target
  twin-min critic. TD 코드와 정책 sampling 함수는 변경하지 않는다.
- 평가는 기존 **zero-z/ε=0와 stochastic-z/ε=0**, 각 10 episodes, 5K 간격과
  step 1 평가. 평가에서 best-of-k를 사용하지 않고 평가 RNG도 그대로 격리한다.
- Actor/critic 256×2, initial σ=.5, batch 256, clipping 없음, UTD=1,
  학습 1M steps. 표준 v5 baseline의 actor loss 경로는 변경하지 않는다.

Gaussian conditional NLL은 winner law의 OT projection이다. 유한 student
표현과 Gaussian 조건부 분포 때문에 이를 완벽히 재현하거나 μ-only 평가의
성능 향상을 보장하는 것은 아니다. Teacher critic 계산은 64개에서 512개로
늘어난다. OT 행렬 크기는 16×64 그대로다.

## 실행과 기록

```bash
OPTIQ_CONFIG=mujoco_v5_bestof8 JAX_PLATFORMS=cpu \
  /root/.venv-optiq-mujoco/bin/python scripts/verify_v5.py benchmark=hopper seed=0

# Long runs are launched by a supervisor-managed campaign worker.
python run_optiq_dime.py --config-name=mujoco_v5_bestof8 \
  benchmark=hopper seed=0 alg.actor.sinkhorn_epsilon=.1 \
  wandb.entity=OptiQ wandb.project=v5-bestk
```

W&B: `OptiQ/v5-bestk`; 새로운 winnerOT 이름·group·run ID·output directory.
GPU 2 Ant seed0, GPU 3 Hopper seed0. 둘 모두 1M을 완료한 뒤 같은 GPU에서
seed1 두 개를 시작한다. 취소된 collection-only seed0 로그는 보존한다.
Seed4 복구는 보류 상태로 유지한다.

`train/teacher_best_of_k=8`, `teacher_winner_count=64`,
`teacher_candidate_count=512`, `teacher_uniform_mass=1`, `teacher_winner_q_gain`을
기록한다. 후자의 gain은 winner Q와 같은 묶음의 평균 candidate Q의 차이다.
기존 collection 지표도 유지한다. Inactive Boltzmann/density 지표는 출력하지 않는다.

검증은 `tests/test_winner_distillation.py`(묶음별 twin-min, 원래 u 보존,
정책 σ 사용, 균등 OT, full-row NLL, 두 head gradient, teacher stop-gradient),
`tests/test_best_of_k.py`(실제 Ant/Hopper 수집·replay·TD·평가 경로), 기존
v4/v5/exploration 회귀를 포함한다. 짧은 검증은 1M 학습 결과가 아니다.

2026-09-15 검증: CPU 회귀 76개 통과, winner GPU 단위 검사 14개 통과.
실제 Ant/Hopper GPU integration도 통과했고, 각 환경에서 batch=256,
16 student / 64 winners / 8 candidates의 40회 update를 추가 확인했다.
GPU 전체 회귀의 기존 v4 수치 비교 1개는 이전 commit에서도 동일하게 실패했다
(μ-only float32 연산 차이, 약 5.5e-6). 해당 policy 코드는 변경하지 않았고
CPU 회귀에서는 통과했다. 테스트 결과와 짧은 검증 로그는 새 캠페인
`hopper_ant_v5_bestk_winner_ot_seeds01_20260915/validation`에 보존한다.
