#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
set -euo pipefail
ulimit -c 0
cd "${OPTIQ_ANALYSIS_ROOT:-/lustre/hobbit9882/OptiQ}"
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1
export WANDB_MODE=disabled WANDB_DISABLED=true MPLBACKEND=Agg MUJOCO_GL=egl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda
unset LD_LIBRARY_PATH
export MPLCONFIGDIR="/tmp/optiq-analysis-mpl-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
PYTHON=/scratch2/hobbit9882/.venvs/optiq-mujoco-scalar/bin/python
if ! srun --cpu-bind=cores "$PYTHON" -X faulthandler -c 'import jax; print(jax.devices(),flush=True); assert jax.default_backend()=="gpu"'; then
    "$PYTHON" - <<'GUARD'
import os,re,subprocess,json,socket
from pathlib import Path
array=os.environ.get('SLURM_ARRAY_JOB_ID')
record={'host':socket.gethostname(),'job':os.environ.get('SLURM_JOB_ID'),'array':array,'reason':'GPU preflight failure'}
if array:
    result=subprocess.run(['squeue','--array','--noheader','--jobs='+array,'--states=PENDING','--format=%i'],capture_output=True,text=True,check=True)
    ids=[i for i in result.stdout.split() if re.fullmatch(re.escape(array)+r'_\d+',i)]
    if ids:
        hold=subprocess.run(['scontrol','hold',','.join(ids)],capture_output=True,text=True)
        record.update(held_pending_jobs=ids,hold_returncode=hold.returncode,hold_stderr=hold.stderr)
root=Path.cwd().parent
(root/('gpu_failure_'+str(os.environ.get('SLURM_JOB_ID'))+'.json')).write_text(json.dumps(record,indent=2))
print(json.dumps(record),flush=True)
GUARD
    exit 70
fi
srun --cpu-bind=cores "$PYTHON" -X faulthandler -m "$@"
