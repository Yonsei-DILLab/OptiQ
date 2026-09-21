#!/usr/bin/env bash
set -euo pipefail
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1
unset LD_LIBRARY_PATH
cd /lustre/hobbit9882/OptiQ/outputs/boltzmann_analysis/20260910_categorical_v2/code
export PYTHONPATH="$PWD"
exec /scratch2/hobbit9882/.venvs/optiq-mujoco-scalar/bin/python -m analysis_boltzmann.categorical_smoke --campaign /lustre/hobbit9882/OptiQ/outputs/boltzmann_analysis/20260910_categorical_v2
