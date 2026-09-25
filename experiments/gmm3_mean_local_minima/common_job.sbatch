#!/bin/bash
#SBATCH --job-name=GMM3-common-I
#SBATCH --partition=dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
set -euo pipefail
unset LD_LIBRARY_PATH PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$STUDY_ROOT/source"
exec /scratch2/hobbit9882/.venvs/optiq-mujoco-scalar/bin/python -m experiments.gmm3_mean_local_minima.range_probe --root "$STUDY_ROOT" --config experiments/gmm3_mean_local_minima/common_interval.json
