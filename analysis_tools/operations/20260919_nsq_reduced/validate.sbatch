#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --exclude=cs-gpu-01,node05,node14,node23,node24,node31,node35,node40
set -euo pipefail
ROOT="$1"
DIMS=(1 2 4 8)
DIM="${DIMS[$SLURM_ARRAY_TASK_ID]}"
PY=/scratch2/hobbit9882/.venvs/optiq-mujoco-scalar/bin/python
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/v5:$ROOT/vendor"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 JAX_DEFAULT_MATMUL_PRECISION=highest
export POT_BACKEND_DISABLE_PYTORCH=1 POT_BACKEND_DISABLE_TENSORFLOW=1 POT_BACKEND_DISABLE_CUPY=1
unset LD_LIBRARY_PATH
"$PY" -m nsq.validate --root "$ROOT" --dim "$DIM"
"$PY" -m nsq.validate_legacy_td --root "$ROOT" --dim "$DIM"
