#!/bin/bash
#SBATCH --job-name=hc-smemtr-1m
#SBATCH --array=0-5%6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=/scratch2/gsmin2024/research/optiq_halfcheetah_smem_tr_20260921/logs/train-%A_%a.log

set -euo pipefail
REPO=/scratch2/gsmin2024/research/optiq_halfcheetah_smem_tr_20260921/source
CAMPAIGN=/scratch2/gsmin2024/research/optiq_halfcheetah_smem_tr_20260921
EXP=$REPO/analysis_tools/experiments/20260921_halfcheetah_smem_tr
PYTHON=/scratch2/gsmin2024/research/optiq_v5_humanoid_density_beta_20260915/.venv/bin/python
SECRET=/scratch2/gsmin2024/.secrets/optiQ_wandb
EXPECTED=$($PYTHON -c "import json;print(json.load(open('$CAMPAIGN/manifest.json'))['git_commit'])")
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED"
test -z "$(git -C "$REPO" status --porcelain)"
$PYTHON - <<'PY'
import json
from pathlib import Path
campaign=Path('/scratch2/gsmin2024/research/optiq_halfcheetah_smem_tr_20260921')
expected=json.loads((campaign/'manifest.json').read_text())['git_commit']
for method in ('direct','smem_tr'):
    result=json.loads((campaign/'smoke'/method/'result.json').read_text())
    assert result['passed'] and result['git_commit']==expected
PY
SEED=$((SLURM_ARRAY_TASK_ID / 2))
METHOD_INDEX=$((SLURM_ARRAY_TASK_ID % 2))
METHODS=(direct smem_tr)
METHOD=${METHODS[$METHOD_INDEX]}
RUN_ID=halfcheetah-${METHOD}-seed${SEED}
STATE=$CAMPAIGN/state/$RUN_ID.json
test ! -e "$STATE"
export METHOD SEED RUN_ID STATE EXPECTED
$PYTHON - <<'PY'
import json,os
from datetime import datetime,timezone
from pathlib import Path
Path(os.environ['STATE']).write_text(json.dumps({
    'id':os.environ['RUN_ID'],'method':os.environ['METHOD'],'seed':int(os.environ['SEED']),
    'status':'running','git_commit':os.environ['EXPECTED'],
    'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
    'slurm_array_job_id':os.environ.get('SLURM_ARRAY_JOB_ID'),
    'slurm_array_task_id':os.environ.get('SLURM_ARRAY_TASK_ID'),
    'started_utc':datetime.now(timezone.utc).isoformat(),
},indent=2)+'\n')
PY
finish() {
  RC=$?
  export RC
  $PYTHON - <<'PY'
import json,os
from datetime import datetime,timezone
from pathlib import Path
p=Path(os.environ['STATE']); d=json.loads(p.read_text())
d.update(status='completed' if int(os.environ['RC'])==0 else 'failed',
         exit_code=int(os.environ['RC']),finished_utc=datetime.now(timezone.utc).isoformat())
p.write_text(json.dumps(d,indent=2)+'\n')
PY
  exit "$RC"
}
trap finish EXIT
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
test -f "$SECRET" && test "$(stat -c '%a' "$SECRET")" = 600
unset WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT WANDB_NAME WANDB_RUN_GROUP WANDB_RUN_ID WANDB_RESUME PYTHONPATH LD_LIBRARY_PATH
IFS= read -r WANDB_API_KEY < "$SECRET"
test -n "$WANDB_API_KEY"
export WANDB_API_KEY WANDB_ENTITY=OptiQ WANDB_MODE=online
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=1
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
cd "$EXP"
"$PYTHON" -u train.py "$METHOD" benchmark=halfcheetah seed="$SEED" \
  run_name="$RUN_ID" output_root="$CAMPAIGN/outputs" require_gpu=true
