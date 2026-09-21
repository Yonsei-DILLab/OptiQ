#!/bin/bash
#SBATCH --job-name=hc-wandb-live
#SBATCH --partition=dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch2/gsmin2024/research/optiq_halfcheetah_smem_tr_20260921/logs/wandb-sidecar-%j.log

set -euo pipefail

PYTHON=/scratch2/gsmin2024/research/optiq_v5_humanoid_density_beta_20260915/.venv/bin/python
CAMPAIGN=/scratch2/gsmin2024/research/optiq_halfcheetah_smem_tr_20260921
export WANDB_API_KEY="$(cat /scratch2/gsmin2024/.secrets/optiQ_wandb)"
export WANDB_ENTITY=OptiQ

exec "$PYTHON" -u wandb_eval_sidecar.py \
  --campaign "$CAMPAIGN" \
  --array-job-id 2303206
