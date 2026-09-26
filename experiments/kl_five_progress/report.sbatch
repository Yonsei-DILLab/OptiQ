#!/bin/bash
#SBATCH --job-name=KL-five-figures
#SBATCH --partition=dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
set -euo pipefail
unset LD_LIBRARY_PATH PYTHONPATH
export JAX_PLATFORMS=cpu PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLCONFIGDIR="$STUDY_ROOT/runtime/report_mpl"
cd "$STUDY_ROOT/source"
exec /scratch2/hobbit9882/.venvs/optiq-mujoco-scalar/bin/python -m experiments.kl_five_progress.figure \
 --root "$STUDY_ROOT" --out "$STUDY_ROOT/reports/ten_k" --commit "$SOURCE_COMMIT"
