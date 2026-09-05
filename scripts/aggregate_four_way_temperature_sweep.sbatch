#!/usr/bin/env bash
#SBATCH --job-name=optiq-4way-temp-aggregate
#SBATCH --partition=suma_rtx4090
#SBATCH --gres=gpu:RTX4090:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/hobbit9882/slurm_logs/optiq/4way-temp-aggregate-%j.out
#SBATCH --error=/lustre/hobbit9882/slurm_logs/optiq/4way-temp-aggregate-%j.err

set -euo pipefail

cd /lustre/hobbit9882/OptiQ
export PYTHONPATH=/lustre/hobbit9882/OptiQ
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-optiq"
mkdir -p "$MPLCONFIGDIR" /lustre/hobbit9882/slurm_logs/optiq
unset LD_LIBRARY_PATH

for temperature in 4 16 64; do
  /lustre/hobbit9882/.venvs/optiq-dime/bin/python -m \
    experiments.four_way_multigoal.aggregate_proposal_recovery \
    --input-dir \
    "outputs/four_way_multigoal/proposal_recovery_temperature/temperature_${temperature}"
done
