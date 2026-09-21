#!/bin/bash
#SBATCH --job-name=oq-nsq-report
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=04:00:00
set -euo pipefail
ROOT=/lustre/hobbit9882/OptiQ-nonstationary-q-20260917
PY=/scratch2/hobbit9882/.venvs/optiq-mujoco-scalar/bin/python
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/v5"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg
"$PY" build_report.py --root "$ROOT" --out "$ROOT/report"
