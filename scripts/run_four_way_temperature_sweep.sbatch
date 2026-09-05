#!/usr/bin/env bash
#SBATCH --job-name=optiq-4way-temp
#SBATCH --partition=suma_rtx4090
#SBATCH --gres=gpu:RTX4090:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-71%12
#SBATCH --output=/lustre/hobbit9882/slurm_logs/optiq/4way-temp-%A_%a.out
#SBATCH --error=/lustre/hobbit9882/slurm_logs/optiq/4way-temp-%A_%a.err

set -euo pipefail

temperatures=(4 16 64)
methods=(
  stratified_truncated_normal
  exact_truncated_normal
  qgradcov_exact_truncated_normal
  qgradcov_stratified_truncated_normal
  qgradcov_exact_normal
  qgradcov_stratified_normal
  gamma_qgrad_exact_normal
  gamma_qgrad_stratified_normal
)

runs_per_temperature=$(( ${#methods[@]} * 3 ))
temperature_index=$(( SLURM_ARRAY_TASK_ID / runs_per_temperature ))
within_temperature=$(( SLURM_ARRAY_TASK_ID % runs_per_temperature ))
method_index=$(( within_temperature / 3 ))
seed=$(( within_temperature % 3 + 1 ))
temperature=${temperatures[$temperature_index]}
method=${methods[$method_index]}
output_dir="outputs/four_way_multigoal/proposal_recovery_temperature/temperature_${temperature}"

cd /lustre/hobbit9882/OptiQ
export PYTHONPATH=/lustre/hobbit9882/OptiQ
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-optiq"
mkdir -p "$MPLCONFIGDIR" /lustre/hobbit9882/slurm_logs/optiq
unset LD_LIBRARY_PATH

echo "Starting temperature=$temperature method=$method seed=$seed CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
exec /lustre/hobbit9882/.venvs/optiq-dime/bin/python -m \
  experiments.four_way_multigoal.proposal_recovery \
  --method "$method" \
  --seed "$seed" \
  --total-steps 100000 \
  --warmup-steps 1000 \
  --init-steps 1500 \
  --trajectory-count 400 \
  --initial-action-count 4096 \
  --proposal-std 0.2 \
  --proposal-clip 0.5 \
  --perpendicular-std-ratio 0.5 \
  --gamma-shape 2.0 \
  --gamma-scale 0.1 \
  --gamma-perpendicular-std 0.1 \
  --include-anchor \
  --clip-untruncated \
  --temperature "$temperature" \
  --output-dir "$output_dir" \
  --log-interval 5000
