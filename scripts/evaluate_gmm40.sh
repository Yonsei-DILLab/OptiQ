#!/usr/bin/env bash
set -euo pipefail
repo_root="${OPTIQ_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
python_bin="${OPTIQ_PYTHON:-$(dirname "$repo_root")/.venv-optiq-no-anchor/bin/python}"
runs_root="${1:?Completed run directory required}"
report_dir="${2:?New report directory required}"
shift 2
cd "$repo_root"
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
"$python_bin" -m benchmarks.gmm40.evaluate \
  --runs-root "$runs_root" --output-dir "$report_dir" "$@"
exec "$python_bin" -m benchmarks.gmm40.report "$report_dir"
