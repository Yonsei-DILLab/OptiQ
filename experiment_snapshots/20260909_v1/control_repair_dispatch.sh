#!/usr/bin/env bash
#SBATCH --partition=dell_cpu
#SBATCH --qos=cpu_qos
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -euo pipefail
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLBACKEND=Agg
unset LD_LIBRARY_PATH
cd /lustre/hobbit9882/OptiQ/outputs/boltzmann_analysis/20260909_v1/code
exec /scratch2/hobbit9882/.venvs/optiq-mujoco-scalar/bin/python -m analysis_boltzmann.repair_dispatch --campaign /lustre/hobbit9882/OptiQ/outputs/boltzmann_analysis/20260909_v1
