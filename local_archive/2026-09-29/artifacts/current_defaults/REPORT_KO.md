**현재 실험 기본 설정 — 2026-09-21 실제 코드·Hydra 합성 설정 기준**

대상은 최근 실험에 사용한 `direct-gmm-trg` 브랜치, 커밋 `c1a73ec41a50ffddc4b205c402f12e9767d8f121`이다. 원격 서버의 `/home/heechan/OptiQ-direct-gmm-trg`에서 확인했다. 작업 트리는 깨끗하고 `origin/direct-gmm-trg`와 동기화되어 있다. 과거 `heejoon`, `v5-gmm` 또는 다른 실행 진입점 전체에 공통인 설정이라는 뜻은 아니다.

실행 진입점은 `analysis_tools/experiments/20260920_truncated_mll/train.py`이다. 기존 v5 runner와 설정을 재사용하되, 해당 실험 폴더의 전용 `optiq_dime` 구현이 먼저 import된다. 아래는 추가 CLI override 없이 이 진입점으로 합성한 기본값이며, 최근 20k 비교의 별도 설정은 뒤에서 구분한다. 보고를 위해 설정을 읽었으며 학습 코드는 수정하지 않았다.

**모델 구조와 초기화**

| 항목 | 실제 설정 |
|---|---|
| Actor | semi-implicit conditional Gaussian policy, 하나의 공유 MLP |
| Actor 입력 | observation과 latent z를 concatenate |
| Actor trunk | Dense 256 → GELU → Dense 256 → GELU |
| Actor 출력 | action 차원의 μ head, action 차원의 log σ head |
| 중심 | μ = tanh(raw μ), 각 좌표가 [-1,1] |
| 조건부 공분산 | diagonal; σ는 state·z·action 좌표별로 학습 |
| Critic | 서로 독립인 scalar Q network 2개 |
| Critic 입력/구조 | concatenate(s,a) → 256 GELU → 256 GELU → scalar |
| Actor–critic trunk 공유 | 없음 |
| BatchNorm / LayerNorm / dropout | 모두 없음 |
| Mean latent skip | 기본 0 |
| Actor hidden 초기화 | variance_scaling(1, fan_avg, uniform), bias 0 |
| μ head 초기화 | variance_scaling(0.0001, fan_avg, uniform), bias 0 |
| log σ head 초기화 | kernel 0, bias -1 |
| Critic 초기화 | Flax Dense 기본 초기화 사용 |

μ head의 `0.0001`은 initializer의 variance scale이며, 모든 weight를 0.0001로 고정한다는 의미가 아니다. 초기 μ는 0 근처이고, 초기 σ는 모든 조건에서 exp(-1) ≈ 0.367879이다.

| 환경 | observation 차원 | action = latent 차원 | Actor 파라미터 | Critic 1개 | Twin critic 합계 |
|---|---:|---:|---:|---:|---:|
| Ant-v4 | 27 | 8 | 79,120 | 75,265 | 150,530 |
| Humanoid-v4 | 376 | 17 | 175,394 | 166,913 | 333,826 |

파라미터 수는 확인한 Dense 구조와 입출력 차원으로 계산했으며 target 복사본과 Adam 상태를 제외한다.

**Latent와 64×64의 의미**

- 기본 latent prior는 표준정규분포 `z ~ N(0,I)`이고 고정 codebook이 아니다. `latent_prior`를 생략했을 때 구현의 기본값이 `normal`이다.
- latent 차원은 action 차원이다. Ant는 8차원, Humanoid는 17차원이다. 64는 latent 차원이 아니다.
- 각 actor update에서 replay state마다 새 z 64개를 뽑아 64개 조건부 성분을 만든다. 가중치는 균등한 1/64이며 학습하는 gating network는 없다.
- teacher candidate는 state당 총 64개다. `proposals_per_policy_sample=1`, `proposal_sampling_mode=exact`에서 성분 index를 균등하게 복원추출한다. 따라서 각 성분에서 반드시 하나씩 뽑는 방식은 아니다.
- NLL에서 teacher 64개 각각에 student 64개 성분의 밀도를 모두 계산한다. state당 4,096쌍, batch 256에서 1,048,576쌍이다. 실제 teacher action이 state당 4,096개인 것은 아니다.
- 연속적인 z를 쓰므로 전체 policy를 영구적인 64성분 GMM으로 고정한 것은 아니다. 업데이트마다 64성분의 Monte Carlo mixture를 구성한다.

**σ 범위와 실제 행동 분포**

사용자가 지정한 앞으로의 기본 허용 범위는 `log σ ∈ [-5,-1]`, 즉 `σ ∈ [0.006737947,0.367879441]`이다. 범위만 고정되며 σ 자체는 학습된다. 범위 제한은 raw log σ에 hard clip으로 적용한다.

행동 분포는 중심 μ를 tanh로 제한한 뒤, `N(μ,diag(σ²))`를 각 좌표의 `[-1,1]` 구간으로 **절단·재정규화한 Gaussian**이다. inverse CDF로 표본을 뽑고, 마지막 clamp는 수치 오차에 대한 보호다. `tanh(μ+σε)` 분포와는 다르다. σ는 절단 전 Gaussian의 파라미터이므로 최종 bounded action의 실제 표준편차와 같지 않다. μ도 절단 분포의 정확한 기댓값과 일반적으로 다르다.

Teacher floor는 exp(-5)이며 actor의 최소 σ와 같아서 별도 폭 확장이 없다. 정규화된 action 공간에서 학습하고, 환경에 전달할 때 원래 action 범위로 환산한다. Ant는 [-1,1], Humanoid는 약 [-0.4,0.4]다.

**Teacher 가중치와 actor objective**

Temperature는 **0.25 고정**, density correction은 **β=1 고정**이다. adaptive β와 temperature schedule은 기본적으로 꺼져 있다. Actor teacher 점수에는 두 online critic의 평균을 쓴다.

`Q̄(s,a) = (Q1(s,a)+Q2(s,a))/2`

`w_j = softmax_j[ Q̄(s,a_j)/0.25 − log q_old(a_j|s) ]`

`L_actor = −mean_state Σ_j w_j log[(1/64) Σ_i pθ,box(a_j|s,z_i)]`

여기서 q_old는 해당 업데이트 시작 시점의 64성분 teacher proposal이다. Teacher 성분 파라미터, candidate, Q 점수와 최종 가중치는 stop-gradient한다. Student mixture의 log-sum-exp와 그에 따른 성분 responsibility, μ, σ 및 절단 정규화 항에는 gradient가 흐른다. Q에 별도로 0.25를 곱하는 설정은 현재 이 경로에 없다.

OT/Sinkhorn, argmax target regression, reweighted teacher resampling, anchor, actor entropy penalty, soft guard는 사용하지 않는다. `experiment.resampling=false`는 가중치에 따른 teacher 재표본화를 하지 않는다는 뜻이며, 최초 proposal 성분 index를 복원추출하는 것과는 별개다. ESS는 진단값이고, 기본 설정에서 ESS≥16을 강제하지 않는다.

**Critic objective와 업데이트**

TD target은 `y = r + 0.99(1-done) min(Q1_target(s',a'), Q2_target(s',a'))`이다. a′는 **현재 actor**에서 새로운 z와 조건부 행동 잡음을 사용해 뽑는다. Actor teacher 점수의 mean-Q와 TD backup의 min-Q를 구분해야 한다.

Critic loss는 두 critic 각각의 batch mean squared TD error를 더한 값이다. Critic을 먼저 갱신하고, target critic에 Polyak update τ=0.005를 적용한 뒤 actor를 갱신한다. Actor target 복사 설정은 `policy_tau=1`이지만, 이 semi-implicit TD 경로의 next action은 target actor 대신 current actor에서 나온다.

Entropy coefficient는 상수 0이고 SAC식 entropy-regularized backup이나 자동 entropy temperature 학습을 하지 않는다. 위의 T=0.25는 teacher 가중치용 temperature다. TD action smoothing도 0이다. Scalar critic이므로 `v_min=-3600`, `v_max=3600`은 Q clipping 범위가 아니다.

**RL 학습 및 optimizer**

| 항목 | 기본값 |
|---|---:|
| 환경 | Humanoid-v4; Ant 등은 override |
| 학습 seed | 0, 기본 실행 1개 seed |
| 총 환경 step | 1,000,000 |
| 수집 환경 수 | 1 |
| Random action warm-up | 5,000 step |
| Actor 학습 시작 | 5,000 step 이후 |
| Replay capacity | 1,000,000 transitions |
| Replay sampling | uniform, batch 256 |
| Train frequency / UTD | 환경 1 step당 update 1번 |
| Actor policy delay | 1: 매 update마다 actor도 학습 |
| Discount γ | 0.99 |
| Target critic τ | 0.005 |
| Actor / critic optimizer | 각각 Adam |
| Actor / critic learning rate | 각각 0.0003, 상수 |
| Adam β1 / β2 | 0.9 / 0.999 |
| Adam ε / ε_root | 1e-8 / 0 |
| Gradient clipping | 없음, ac_grad_norm=null |
| Weight decay / LR schedule / model reset | 없음 |
| Warm-up 이후 uniform action 교체 확률 | 0 |
| 별도 action noise / SDE | 없음 |
| Observation / reward normalization | 별도 적용 없음 |
| JIT / GPU 요구 | true / true |

RL의 batch 256은 서로 다른 replay transition 256개를 평균내는 설정이다. 같은 state에서 teacher 묶음 256개만 독립적으로 반복하는 설정과 다르다. Actor와 critic은 업데이트에서 같은 replay batch를 사용한다. Replay는 CPU에 저장하고, 기본 memory optimization은 꺼져 있으며 time-limit truncation은 실제 terminal과 구분해 bootstrap한다. 1M step을 끝내면 warm-up 5k를 제외한 약 995k번의 actor/critic update가 생긴다.

**Gymnasium 환경 세부**

추가 환경 인자 없이 `gym.make(env_name)`를 호출한다. 다음 값은 설치된 환경을 직접 생성해 확인했다.

| 항목 | Ant-v4 | Humanoid-v4 |
|---|---:|---:|
| Episode time limit | 1,000 | 1,000 |
| frame skip / dt | 5 / 0.05 | 5 / 0.015 |
| Control cost weight | 0.5 | 0.1 |
| Healthy reward | 1 | 5 |
| Healthy z 범위 | [0.2,1.0] | [1.0,2.0] |
| Unhealthy 시 종료 | true | true |
| Reset noise scale | 0.1 | 0.01 |
| Observation에서 현재 x,y 제외 | true | true |

Ant의 `use_contact_forces=false`, contact cost weight 설정은 0.0005다. Humanoid forward reward weight는 1.25다. 커스텀 reward scale을 덧씌우지 않는다.

**평가: 두 모드를 별도 기록**

| 모드 | z | 실제 action |
|---|---|---|
| zero_z | 항상 0 | bounded center μ(s,0) |
| stochastic_z | 매 action 표준정규 z 새로 추출 | bounded center μ(s,z) |

`dual_mu_eval=true`, `mu_only_eval=true`이므로 **둘 다 조건부 Gaussian 행동 잡음을 넣지 않는다**. stochastic_z reward는 전체 noisy policy reward와 다르다. 전체 noisy policy는 warm-up 이후 수집과 TD next-action sampling에 쓰인다.

시작 평가가 켜져 있으며 callback상 첫 환경 step에서 평가하고, 이후 매 5,000 step마다 각 모드 10 episode씩 평가한다. 두 모드는 동일한 episode별 환경 seed와 policy seed를 배정받는다. 평가 후 policy RNG와 평가 모드 상태를 복구해서 수집 RNG를 보존한다. 현재 전용 dual 평가 callback은 normal latent를 전제로 하므로, 예전 다른 브랜치의 fixed-codebook 평가 지원과 혼동하면 안 된다.

주요 reward key는 `eval/zero_z/mean_reward`, `eval/stochastic_z/mean_reward`이고 각각 `std_reward`, `mean_ep_length`도 기록한다. 기존 `eval/mean_reward`와 최종 `final_eval_return`은 zero_z의 alias다. 모드별 원시 episode reward·길이·환경 seed·policy seed를 NPZ로 저장한다. 다른 브랜치에 있는 모든 best/std/episode 관련 추가 key가 이 전용 callback에도 전부 있다는 의미는 아니다.

**기록, 진단, 체크포인트**

- W&B: online, entity `OptiQ`, 기본 project `heejoon-truncated-mll`.
- 기본 run name: `humanoid-truncatedMLL-r2-N64-M64-T025-s0`.
- 기본 group: `Humanoid-v4_truncatedMLL_r2_N64_M64_T025`; job type: `direct-marginal-likelihood`.
- 기본 output_root 설정 문자열: `../optiq-experiments/v5/outputs`.
- Eval interval 5,000, diagnostic interval 5,000, checkpoint interval 50,000.
- `log_interval=1`은 RL runner의 episode 단위 flush 설정이며 모든 scalar가 매 환경 step에 출력된다는 뜻은 아니다. 상세 진단은 별도 주기를 따른다.
- Actor/critic loss, Q, σ와 density/weight/ESS 관련 진단 등을 기록하고 local CSV, TensorBoard, 평가 NPZ, model checkpoint를 남긴다.
- 모델 checkpoint는 replay buffer 전체를 포함하는 실험 상태 snapshot과 구분해야 한다.

**설정에 남아 있지만 현재 학습에서 비활성인 값**

| 값 | 현재 효과 |
|---|---|
| Sinkhorn ε=0.1, iterations=100, transport_target_mode=argmax | Direct NLL이므로 OT loss에 사용되지 않음 |
| normalize_ot_cost=false, ot_student_action=mean | 현재 Direct NLL objective를 정하는 항목이 아님 |
| proposal_clip=0.5 및 legacy proposal_std_pretanh | 현재 truncated conditional mixture를 예전 pre-tanh perturbation proposal로 바꾸지 않음 |
| minimum_source_ess=16, density_beta_grid_size=257 | adaptive_density_beta=false라 ESS 제약 β 탐색에 사용하지 않음 |
| BN momentum=.99, mode=brn_actor, warmup=100000 | bn=false라 학습에 효과 없음 |
| Critic v_min/max=±3600 | scalar n_atoms=1이라 distributional support로 사용하지 않음 |
| entropy_samples=0, entropy_diagnostics=false | entropy 진단 꺼짐 |

**최근 완료한 20k 비교와의 차이**

| 항목 | 현재 기본값 | 최근 Ant/Humanoid 비교 |
|---|---|---|
| log σ 범위 | [-5,-1] | [-5,0] 또는 [-5,-2] |
| 초기 log σ / σ | -1 / 0.367879 | log(0.1)≈-2.302585 / 0.1 |
| 학습 길이 | 1M | 20k |
| 평가·진단 주기 | 5k | 1k |
| W&B project | OptiQ/heejoon-truncated-mll | OptiQ/abla |
| 환경·seed 구성 | Humanoid, seed 0 | Ant/Humanoid × 상한 2개, 각각 seed 0 |

최근 비교에서도 256×2, batch 256, UTD 1, LR 3e-4, T=.25, N=M=64, 랜덤 latent를 유지했다. 비교 실험의 override는 앞으로의 기본값으로 자동 승격되지 않는다.

**확인된 설명 메타데이터의 불일치**

재사용 runner가 W&B에 넣는 설명 문자열 일부에는 예전 `tanh(mu+sigma*eps)`, `pre-tanh`, `teacher_hard_cutoff=false` 문구가 남아 있다. 실제 전용 구현과 `experiment.distribution=box_truncated_gaussian` 설정은 위에서 설명한 box-truncated Gaussian이다. 해당 예전 문구만 보고 분포와 loss를 해석하면 안 된다. README의 beta annealing 관련 표현과 달리 현재 합성 기본값은 β=1 고정이다. 이 보고에서는 실제 활성 코드를 기준으로 판단했고, 설명 문구를 수정하지 않았다.

**검증 자료**

같은 폴더의 `resolved_default.json`은 실제 진입점에서 합성한 전체 설정이다. `dual_evaluation.py`는 확인한 전용 평가 구현의 읽기 전용 사본이다. 아래 부록은 누락 없이 전체 합성 설정을 펼친 것이다.


| 전체 설정 key | 값 |
|---|---|
| `env_name` | `"Humanoid-v4"` |
| `task` | `"humanoid"` |
| `successful_steps` | `null` |
| `alg.tau` | `0.005` |
| `alg.policy_tau` | `1.0` |
| `alg.utd` | `1` |
| `alg.gamma` | `0.99` |
| `alg.policy_delay` | `1` |
| `alg.batch_size` | `256` |
| `alg.buffer_size` | `1000000` |
| `alg.learning_starts` | `5000` |
| `alg.reset_models` | `false` |
| `alg.ent_coef.type` | `"const"` |
| `alg.ent_coef.init` | `0.0` |
| `alg.critic.backup_mode` | `"td"` |
| `alg.critic.activation` | `"gelu"` |
| `alg.critic.n_critics` | `2` |
| `alg.critic.hs` | `[256, 256]` |
| `alg.critic.dropout_rate` | `null` |
| `alg.critic.use_layer_norm` | `false` |
| `alg.critic.n_atoms` | `1` |
| `alg.critic.v_min` | `-3600` |
| `alg.critic.v_max` | `3600` |
| `alg.critic.entr_coeff` | `0.0` |
| `alg.critic.type` | `"scalar"` |
| `alg.critic.crossq_style` | `false` |
| `alg.optimizer.bn` | `false` |
| `alg.optimizer.bn_momentum` | `0.99` |
| `alg.optimizer.bn_mode` | `"brn_actor"` |
| `alg.optimizer.bn_warmup` | `100000` |
| `alg.optimizer.lr_critic` | `0.0003` |
| `alg.optimizer.lr_actor` | `0.0003` |
| `alg.optimizer.critic_b1` | `0.9` |
| `alg.optimizer.critic_b2` | `0.999` |
| `alg.optimizer.actor_b1` | `0.9` |
| `alg.optimizer.actor_b2` | `0.999` |
| `alg.optimizer.ac_grad_norm` | `null` |
| `alg.actor.hidden_dims` | `[256, 256]` |
| `alg.actor.num_policy_samples` | `64` |
| `alg.actor.proposals_per_policy_sample` | `1` |
| `alg.actor.proposal_sampling_mode` | `"exact"` |
| `alg.actor.proposal_std` | `0.006737946999085467` |
| `alg.actor.proposal_clip` | `0.5` |
| `alg.actor.include_anchor` | `false` |
| `alg.actor.density_correction` | `true` |
| `alg.actor.density_beta` | `1.0` |
| `alg.actor.adaptive_density_beta` | `false` |
| `alg.actor.minimum_source_ess` | `16.0` |
| `alg.actor.density_beta_grid_size` | `257` |
| `alg.actor.source_q_eval` | `"mean"` |
| `alg.actor.source_reference` | `"uniform_action"` |
| `alg.actor.temperature` | `0.25` |
| `alg.actor.sinkhorn_epsilon` | `0.1` |
| `alg.actor.sinkhorn_iterations` | `100` |
| `alg.actor.transport_target_mode` | `"argmax"` |
| `alg.actor.td_noise_std` | `0.0` |
| `alg.actor.td_noise_clip` | `0.0` |
| `alg.actor.learning_starts` | `5000` |
| `alg.actor.density_correction_beta` | `1.0` |
| `alg.actor.distillation_loss` | `"direct_gmm_nll"` |
| `alg.actor.type` | `"semi_implicit"` |
| `alg.actor.entropy_samples` | `0` |
| `alg.actor.entropy_diagnostics` | `false` |
| `alg.actor.log_std_min` | `-5.0` |
| `alg.actor.log_std_max` | `-1.0` |
| `alg.actor.initial_log_std` | `-1.0` |
| `alg.actor.mean_output_init_scale` | `0.0001` |
| `alg.actor.log_std_output_init_scale` | `0.0` |
| `alg.actor.proposal_std_pretanh` | `0.006737946999085467` |
| `alg.actor.normalize_ot_cost` | `false` |
| `alg.actor.teacher_distribution` | `"conditional_mixture"` |
| `alg.actor.teacher_std_floor` | `0.006737946999085467` |
| `alg.actor.soft_guard.enabled` | `false` |
| `alg.actor.ot_student_action` | `"mean"` |
| `alg.behavior_uniform_probability` | `0.0` |
| `seed` | `0` |
| `use_jit` | `true` |
| `require_gpu` | `true` |
| `total_steps` | `1000000` |
| `eval_interval` | `5000` |
| `num_eval_episodes` | `10` |
| `eval_at_start` | `true` |
| `stochastic_eval` | `true` |
| `mu_only_eval` | `true` |
| `diagnostic_interval` | `5000` |
| `checkpoint_interval` | `50000` |
| `log_interval` | `1` |
| `progress_bar` | `false` |
| `output_root` | `"../optiq-experiments/v5/outputs"` |
| `run_name` | `"humanoid-truncatedMLL-r2-N64-M64-T025-s0"` |
| `wandb.activate` | `true` |
| `wandb.mode` | `"online"` |
| `wandb.entity` | `"OptiQ"` |
| `wandb.project` | `"heejoon-truncated-mll"` |
| `wandb.group` | `"Humanoid-v4_truncatedMLL_r2_N64_M64_T025"` |
| `wandb.job_type` | `"direct-marginal-likelihood"` |
| `dual_mu_eval` | `true` |
| `experiment.method` | `"truncated_marginal_nll"` |
| `experiment.distribution` | `"box_truncated_gaussian"` |
| `experiment.center_transform` | `"tanh"` |
| `experiment.bounded_mu` | `true` |
| `experiment.ot` | `false` |
| `experiment.resampling` | `false` |
| `experiment.components` | `64` |
| `experiment.candidates` | `64` |
| `experiment.teacher_extra_floor` | `false` |
