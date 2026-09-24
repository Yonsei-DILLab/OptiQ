"""Provision isolated launcher paths on the approved added 4090 host.

Prerequisite: copy the native AntMaze overlay and MuJoCo210 from the control
host to the same absolute paths. Existing DIPO/NM environments are read-only
parents, never edited. No training is launched by this script.
"""
from pathlib import Path
import json
import subprocess
import time

HOME_ROOT=Path('/home/heechan')
OPS=HOME_ROOT/'OptiQ-ops'
RUNTIME=HOME_ROOT/'.venv-ddiffpg-native'
PARENTS=(Path('/workspace/dipo-vast-20260914/venv/lib/python3.11/site-packages'),
         Path('/workspace/optiq-v2-transfer-20260911/venv/lib/python3.11/site-packages'))


def main():
    for p in PARENTS:assert p.is_dir(),p
    assert (RUNTIME/'bin/python').exists()
    for p in ['locks','supervisor/jobs','supervisor/logs','sources','reviews','cache']:
        (OPS/p).mkdir(parents=True,exist_ok=True)
    (HOME_ROOT/'optiq-experiments').mkdir(exist_ok=True)
    (RUNTIME/'lib/python3.11/site-packages/optiq_shared.pth').write_text(''.join(str(p)+'\n' for p in PARENTS))
    wrapper=OPS/'run-gpu.sh'
    assert not wrapper.exists(), 'Do not overwrite a pre-existing launcher'
    wrapper.write_text('''#!/usr/bin/env bash
set -euo pipefail
gpu=${1:?GPU required}; shift
if [[ "${1:-}" = --branch ]]; then shift 2; fi
[[ "$gpu" =~ ^[0-3]$ ]]
exec 9>"/home/heechan/OptiQ-ops/locks/gpu-${gpu}.lock"
flock -n 9 || exit 3
export CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu"
export MUJOCO_PY_MUJOCO_PATH=/home/heechan/.mujoco/mujoco210
export JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/home/heechan/OptiQ-ops/cache/jax
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
# Reuse existing W&B authentication without copying it into the repository.
if [[ -f /workspace/optiq-clean-seed3to7-20260909/wandb_api_key ]]; then
    export WANDB_API_KEY="$(cat /workspace/optiq-clean-seed3to7-20260909/wandb_api_key)"
fi
cd "${OPTIQ_SOURCE_DIR:?Frozen source required}"
exec "$@"
''')
    wrapper.chmod(0o755)
    conf=OPS/'supervisor/supervisord.conf'
    assert not conf.exists(), 'Do not overwrite an existing supervisor'
    conf.write_text('''[unix_http_server]
file=/home/heechan/OptiQ-ops/supervisor/supervisor.sock
chmod=0700
[supervisord]
logfile=/home/heechan/OptiQ-ops/supervisor/supervisord.log
pidfile=/home/heechan/OptiQ-ops/supervisor/supervisord.pid
childlogdir=/home/heechan/OptiQ-ops/supervisor/logs
[rpcinterface:supervisor]
supervisor.rpcinterface_factory=supervisor.rpcinterface:make_main_rpcinterface
[supervisorctl]
serverurl=unix:///home/heechan/OptiQ-ops/supervisor/supervisor.sock
[include]
files=/home/heechan/OptiQ-ops/supervisor/jobs/*.conf
''')
    subprocess.run(['/usr/local/bin/supervisord','-c',str(conf)],check=True)
    (OPS/'reviews/progress100-bootstrap.json').write_text(json.dumps(dict(
        time=time.time(),runtime=str(RUNTIME),read_only_parents=list(map(str,PARENTS)),
        training_launched=False),indent=2)+'\n')

if __name__=='__main__':main()
