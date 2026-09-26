#!/bin/bash
#SBATCH --job-name=KL5-logstep-figures
#SBATCH --partition=dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
set -euo pipefail
unset LD_LIBRARY_PATH PYTHONPATH
export JAX_PLATFORMS=cpu PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLCONFIGDIR="$STUDY_ROOT/runtime/mpl_report"
cd "$STUDY_ROOT/source"
EXTRA=()
if [[ ${EARLY_ONLY:-0} == 1 ]]; then EXTRA+=(--early-only); fi
exec /scratch2/hobbit9882/.venvs/optiq-mujoco-scalar/bin/python -m experiments.kl_five_logsteps.figure \
 --root "$STUDY_ROOT" --parent "$PARENT_ROOT" --out "$STUDY_ROOT/reports/$REPORT_NAME" "${EXTRA[@]}"
