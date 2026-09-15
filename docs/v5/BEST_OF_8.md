# 학습 행동 수집에만 best-of-8 적용

Profile: `mujoco_v5_bestof8`, alias `v5/bestof8`.
Canonical v5 config는 변경하지 않고, 추가 `alg.behavior_best_of_k: 8`로 선택한다.
이번 설정은 **Kb=8, Kt=1**, 평가에서는 후보 비교를 수행하지 않는다.
작업 브랜치는 `v5_bestk`이며 `v5`에는 이 변경을 커밋하지 않는다.

Warmup 5K는 기존 uniform 행동이다. 이후 같은 상태에서 현재 Gaussian
정책의 행동 8개를 만들고 **live twin critic min(Q1,Q2)**가 가장 큰 하나를
실행한다. 후보마다 z와 conditional Gaussian epsilon을 샘플링한다.
기존 collector의 정책 draw가 후보 0이며, 나머지 7개는 seed로 고정한 별도
RNG에서 뽑는다. 추가 RNG는 평가나 learner RNG를 진행시키지 않는다.
Replay에는 실제 실행한 행동의 정규화 좌표를 저장한다.

TD target은 기존 current actor의 단일 Gaussian 행동과 target twin-min을
유지한다. OT teacher와 NLL, 100회 Sinkhorn, mean-action student OT,
optimizer, 평가 callback은 기존 v5와 같다. 정책 예측 함수도 변경하지 않는다.
따라서 평가는 **zero-z/epsilon=0 및 sampled-z/epsilon=0**, 각 10 episode,
5K 간격으로 기존과 동일하며, 평가에서 Q로 행동을 선택하지 않는다.

이는 수집 정책만 바꾸는 off-policy 실험이다. 선택된 행동의 critic 점수가
높아져도 평가 actor의 return 향상을 보장하지는 않는다. 평가 점수가 바뀌는
이유는 학습 결과의 차이여야 하며 평가 시 best-of-8의 도움을 받지 않는다.

`rollout/behavior_best_of_k`, `best_of_k_action_count`, `best_of_k_selected_q`,
`best_of_k_base_q`, `best_of_k_q_gain`, `best_of_k_kept_base_fraction`을 기록한다.
마지막 다섯 지표도 `rollout/` prefix를 가진다. Q gain은 같은 후보 집합에서
선택값과 후보 0의 차이다. 환경 메타데이터에 collection-only 적용을 명시한다.

`tests/test_best_of_k.py`는 live twin-min 선택, Gaussian 후보의 분산,
warmup/K=1 보존, 실행/replay 일치, 실제 Hopper/Ant 업데이트 및 평가·TD
분리를 검증한다. 기존 v5/behavior/dual-evaluation 회귀 검사도 함께 수행한다.

## 문헌과 이번 실험의 관계

- [QVPO §4.4/§5.2](https://papers.nips.cc/paper_files/paper/2024/file/6111371a868af8dcfba0f96ad9e25ae3-Paper-Conference.pdf)는
  behavior Kb와 target Kt를 분리하며 target-Q 과대추정 때문에 Kt<Kb를
  권장한다. Ant ablation에서 (4,1)/(4,2)가 (4,4)보다 안정적이다.
  이를 모든 환경에서 Kt>1이 금지된다는 뜻으로 해석하지 않는다.
- [SMFP §4.4](https://arxiv.org/html/2605.21282v1#S4.SS4)는
  Kt=0.5Kb를 명시한다. TD target에서도 후보 선택을 사용한다.
  이번 Kb=8/Kt=1은 사용자가 고른 수집-only ablation이며 SMFP 재현이 아니다.
- [FASTER](https://arxiv.org/html/2604.19730v1)는 best-of-N의 test-time
  계산량을 줄이기 위해 denoising 도중 후보를 거른다. 학습/배포의 후보
  선택을 다루지만, 이번처럼 평가 시 선택을 끈 설정과 결과를 동일시하지 않는다.
- [Q-Planning §3.3](https://arxiv.org/html/2608.21204v1#S3.SS3)는 배포 시
  BC 후보들의 softmax Q-weighted 평균을 사용한다. Hard argmax best-of-N은
  별도 비교 대상이며, 이번 선택 연산을 Q-Planning 전체 알고리즘으로 부르지 않는다.

## 실행과 검증

```bash
# 설정 검사만 수행한다. Seed 4도 직접 entrypoint에서 지원한다.
OPTIQ_CONFIG=mujoco_v5_bestof8 JAX_PLATFORMS=cpu \
  /root/.venv-optiq-mujoco/bin/python scripts/verify_v5.py \
  benchmark=hopper seed=4 alg.actor.temperature=.01

# Supervisor worker에서 호출하는 학습 entrypoint 예시.
python run_optiq_dime.py --config-name=mujoco_v5_bestof8 \
  benchmark=hopper seed=4 alg.actor.temperature=.01 \
  alg.actor.sinkhorn_epsilon=.1 wandb.entity=OptiQ wandb.project=v5-jaehoon
```

2026-09-15 검증: `test_best_of_k.py`, `test_behavior_uniform.py`, `test_v5.py`,
`test_v4.py`, `test_v5_exploration.py`의 CPU 회귀 **62개 통과**.
`test_real_training_replay_and_unchanged_dual_evaluation`의 Hopper/Ant GPU
검증 **2개 통과**. GPU 검증은 각각 8 environment steps의 짧은 integration
test이며 1M 학습 실험 결과가 아니다.

AST 대조에서 `OptiQDIME`의 변경 메서드는 `__init__`, `_sample_action`뿐이다.
`update_critic`, `update_actor`, `_train`은 기존과 동일하다.
Policy/dual-evaluation/base collector 파일은 이전 Hopper 38f48bd와 Ant
71c5ba8 source와 byte 단위로 동일함을 확인했다.
