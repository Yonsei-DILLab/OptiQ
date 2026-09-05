#!/usr/bin/env bash
#SBATCH --job-name=dog-cov-normal
#SBATCH --partition=suma_a6000
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/lustre/hobbit9882/slurm_logs/optiq/dog-cov-normal-%A_%a.out
#SBATCH --error=/lustre/hobbit9882/slurm_logs/optiq/dog-cov-normal-%A_%a.err

set -euo pipefail

tasks=(trot walk run)
task=${tasks[$SLURM_ARRAY_TASK_ID]}
runner=/lustre/hobbit9882/OptiQ/scripts/run_dog_covariance_variant.sh

echo "Launching task=$task method=qgradcov_gaussian_stratified CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
exec bash "$runner" "$task" qgradcov_gaussian_stratified
