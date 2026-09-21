#!/bin/bash
#SBATCH --job-name=spline-mj-1m
#SBATCH --array=0-14%15
#SBATCH --partition=big_suma_rtx3090,suma_rtx4090,suma_a6000,gigabyte_a6000,gigabyte_a5000,asus_6000ada,tyan_a6000,asus_a5000
#SBATCH --qos=big_qos
#SBATCH --exclude=node24,node48,cs-gpu-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=/scratch2/gsmin2024/research/optiq_mujoco_spline_energy_20260921/logs/train-%A_%a.out
#SBATCH --error=/scratch2/gsmin2024/research/optiq_mujoco_spline_energy_20260921/logs/train-%A_%a.err

set -euo pipefail
CAMPAIGN=/scratch2/gsmin2024/research/optiq_mujoco_spline_energy_20260921
SOURCE="$CAMPAIGN/snapshot"
EXP="$SOURCE/analysis_tools/experiments/20260921_mujoco_spline_energy"
PYTHON=/scratch2/gsmin2024/research/optiq_v5_humanoid_density_beta_20260915/.venv/bin/python
SECRET=/scratch2/gsmin2024/.secrets/optiQ_wandb
EXPECTED=$($PYTHON -c "import json;print(json.load(open('$CAMPAIGN/manifest/campaign.json'))['git_commit'])")
test "$(git -C "$SOURCE" rev-parse HEAD)" = "$EXPECTED"
test -z "$(git -C "$SOURCE" status --porcelain)"
$PYTHON -c "import json;assert json.load(open('$CAMPAIGN/checks/gpu_smoke.json'))['passed']"
ENVS=(Hopper-v4 Walker2d-v4 HalfCheetah-v4 Ant-v4 Humanoid-v4)
ENV_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
SEED=$((SLURM_ARRAY_TASK_ID % 3))
ENV_NAME=${ENVS[$ENV_INDEX]}
TASK=${ENV_NAME%-v4}
TASK=${TASK,,}
RUN_ID="${TASK}-spline-energy-s${SEED}"
OUTPUT="$CAMPAIGN/outputs/$RUN_ID"
STATE="$CAMPAIGN/state/$RUN_ID.json"
test ! -e "$OUTPUT" && test ! -e "$STATE"
export ENV_NAME SEED RUN_ID OUTPUT STATE EXPECTED
$PYTHON - <<'PY'
import json,os
from datetime import datetime,timezone
from pathlib import Path
Path(os.environ['STATE']).write_text(json.dumps({
  'id':os.environ['RUN_ID'],'env':os.environ['ENV_NAME'],'seed':int(os.environ['SEED']),
  'status':'running','git_commit':os.environ['EXPECTED'],'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
  'array_job_id':os.environ.get('SLURM_ARRAY_JOB_ID'),'array_task_id':os.environ.get('SLURM_ARRAY_TASK_ID'),
  'started_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n')
PY
finish() {
  RC=$?
  export RC
  "$PYTHON" - <<'PY'
import json,os
from datetime import datetime,timezone
from pathlib import Path
p=Path(os.environ['STATE']);d=json.loads(p.read_text())
d.update(status='completed' if int(os.environ['RC'])==0 else 'failed',exit_code=int(os.environ['RC']),finished_utc=datetime.now(timezone.utc).isoformat())
p.write_text(json.dumps(d,indent=2)+'\n')
PY
  exit "$RC"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
test -f "$SECRET" && test "$(stat -c '%a' "$SECRET")" = 600
unset WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT WANDB_NAME WANDB_RUN_GROUP WANDB_RUN_ID WANDB_RESUME PYTHONPATH LD_LIBRARY_PATH
IFS= read -r WANDB_API_KEY < "$SECRET"
test -n "$WANDB_API_KEY"
export WANDB_API_KEY WANDB_ENTITY=models WANDB_PROJECT=OptiQ-MuJoCo-Spline-Energy
export WANDB_RUN_GROUP=spline_energy WANDB_MODE=online
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS=1
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
cd "$EXP"
"$PYTHON" -u train.py --env "$ENV_NAME" --seed "$SEED" --output "$OUTPUT" --require-gpu
