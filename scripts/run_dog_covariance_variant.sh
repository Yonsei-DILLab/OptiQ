#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 TASK METHOD" >&2
  exit 2
fi

task=$1
method=$2
case "$task" in
  trot|walk|run) ;;
  *) echo "unknown dog task: $task" >&2; exit 2 ;;
esac

case "$method" in
  qgradcov_truncated_stratified)
    proposal_family=qgradcov_truncated
    proposal_sampling_mode=stratified
    ;;
  qgradcov_gaussian_stratified)
    proposal_family=qgradcov_gaussian
    proposal_sampling_mode=stratified
    ;;
  isotropic_truncated_exact)
    proposal_family=isotropic_truncated
    proposal_sampling_mode=exact
    ;;
  *) echo "unknown proposal method: $method" >&2; exit 2 ;;
esac

repo_dir=/lustre/hobbit9882/OptiQ
python_bin=/lustre/hobbit9882/.venvs/optiq-dime/bin/python
output_root_rel=outputs/optiq_dime_dog_covariance_comparison_1m
output_root="$repo_dir/$output_root_rel"
seed=1

completion_dir="$output_root/completed"
timing_dir="$output_root/timings"
wandb_dir="$output_root/wandb"
mkdir -p "$completion_dir" "$timing_dir" "$wandb_dir"

marker="$completion_dir/${task}_${method}_seed${seed}.done"
timing_path="$timing_dir/${task}_${method}_seed${seed}.txt"
if [[ -f "$marker" ]]; then
  echo "Skipping completed task=$task method=$method seed=$seed"
  exit 0
fi

cd "$repo_dir"
export PYTHONPATH="$repo_dir"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export HYDRA_FULL_ERROR=1
export WANDB_DIR="$wandb_dir"
unset LD_LIBRARY_PATH

run_name="optiq_dime_dog_${task}_seed${seed}_${method}_N16R5_T0p25_sigma0p1_clip0p15_1m"
group="dog-${task}_proposal-covariance-comparison-1m"

printf 'task=%s\nmethod=%s\nseed=%s\nslurm_job_id=%s\ncuda_visible_devices=%s\nstarted_utc=%s\n' \
  "$task" "$method" "$seed" "${SLURM_JOB_ID:-none}" \
  "${CUDA_VISIBLE_DEVICES:-unset}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$timing_path"

echo "Starting task=$task method=$method seed=$seed CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
started_epoch=$(date +%s)
set +e
"$python_bin" run_optiq_dime.py \
  task="$task" \
  seed="$seed" \
  total_steps=1000000 \
  eval_interval=5000 \
  num_eval_episodes=10 \
  eval_at_start=true \
  stochastic_eval=true \
  checkpoint_interval=50000 \
  alg.tau=1.0 \
  alg.policy_tau=1.0 \
  alg.utd=2 \
  alg.gamma=0.99 \
  alg.policy_delay=1 \
  alg.batch_size=256 \
  alg.buffer_size=1000000 \
  alg.learning_starts=5000 \
  alg.reset_models=false \
  alg.ent_coef.type=const \
  alg.ent_coef.init=0.0 \
  alg.critic.activation=relu \
  alg.critic.n_critics=2 \
  'alg.critic.hs=[2048,2048]' \
  alg.critic.dropout_rate=null \
  alg.critic.use_layer_norm=false \
  alg.critic.n_atoms=101 \
  alg.critic.v_min=-200 \
  alg.critic.v_max=200 \
  alg.critic.entr_coeff=0.005 \
  alg.optimizer.bn=true \
  alg.optimizer.bn_momentum=0.99 \
  alg.optimizer.bn_mode=brn_actor \
  alg.optimizer.bn_warmup=100000 \
  alg.optimizer.lr_critic=0.0003 \
  alg.optimizer.lr_actor=0.0003 \
  alg.optimizer.critic_b1=0.5 \
  alg.optimizer.critic_b2=0.999 \
  alg.optimizer.actor_b1=0.9 \
  alg.optimizer.actor_b2=0.999 \
  'alg.actor.hidden_dims=[256,256,256]' \
  alg.actor.num_policy_samples=16 \
  alg.actor.proposals_per_policy_sample=5 \
  alg.actor.proposal_family="$proposal_family" \
  alg.actor.proposal_sampling_mode="$proposal_sampling_mode" \
  alg.actor.proposal_std=0.1 \
  alg.actor.proposal_clip=0.15 \
  alg.actor.proposal_perpendicular_std_ratio=0.5 \
  alg.actor.clip_untruncated_proposals=true \
  alg.actor.include_anchor=true \
  alg.actor.density_correction=true \
  alg.actor.density_beta=1.0 \
  alg.actor.adaptive_density_beta=false \
  alg.actor.source_q_eval=mean \
  alg.actor.source_reference=uniform_action \
  alg.actor.temperature=0.25 \
  alg.actor.sinkhorn_epsilon=0.05 \
  alg.actor.sinkhorn_iterations=30 \
  alg.actor.transport_target_mode=argmax \
  alg.actor.td_noise_std=0.2 \
  alg.actor.td_noise_clip=0.5 \
  output_root="$output_root_rel" \
  run_name="$run_name" \
  wandb.activate=true \
  wandb.mode=online \
  wandb.entity=hobbit9882-yonsei-university \
  wandb.project=optiq_dime_dog_n16r5_anchor_t025_2seed_1m \
  wandb.group="$group" \
  wandb.job_type="seed-${seed}"
status=$?
set -e
finished_epoch=$(date +%s)

printf 'finished_utc=%s\nelapsed_seconds=%s\nexit_code=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "$((finished_epoch - started_epoch))" "$status" >> "$timing_path"
if [[ $status -eq 0 ]]; then
  touch "$marker"
fi
exit "$status"
