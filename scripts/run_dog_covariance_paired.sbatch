#!/usr/bin/env bash
#SBATCH --job-name=dog-cov-paired
#SBATCH --partition=suma_a6000
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/lustre/hobbit9882/slurm_logs/optiq/dog-cov-paired-%A_%a.out
#SBATCH --error=/lustre/hobbit9882/slurm_logs/optiq/dog-cov-paired-%A_%a.err

set -euo pipefail

tasks=(trot walk run)
task=${tasks[$SLURM_ARRAY_TASK_ID]}
runner=/lustre/hobbit9882/OptiQ/scripts/run_dog_covariance_variant.sh
log_dir="/lustre/hobbit9882/slurm_logs/optiq/dog-covariance/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$log_dir"

echo "Launching paired task=$task on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

bash "$runner" "$task" qgradcov_truncated_stratified \
  > "$log_dir/qgradcov_truncated_stratified.out" 2>&1 &
qgrad_pid=$!
bash "$runner" "$task" isotropic_truncated_exact \
  > "$log_dir/isotropic_truncated_exact.out" 2>&1 &
isotropic_pid=$!

set +e
wait "$qgrad_pid"
qgrad_status=$?
wait "$isotropic_pid"
isotropic_status=$?
set -e

echo "Finished paired task=$task qgrad_status=$qgrad_status isotropic_status=$isotropic_status"
if [[ $qgrad_status -ne 0 || $isotropic_status -ne 0 ]]; then
  exit 1
fi
