"""Register one bounded, CPU-only inference job; never schedule training."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

from .register_horizon_temperature import CORE_FILES, PARENT

CAMPAIGN = 'antmaze-v3-startnorm-paired-horizon-diagnostic-20260925'
TRAINING_SOURCE = 'db4ca0a446f5fe8df1dda02e261fa9154c66d9c9'
RUN = Path('/home/heechan/optiq-experiments/antmaze-optiq-v3-startnorm-geodesic-250k-s0-20260925/runs/v3-optiq-startnorm-geodesic-gamma999-T3-H0.7-i500-250k-s0')


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--host', choices=['vast-heechan-180'], required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    assert source == Path('/home/heechan/OptiQ-ops/sources') / sha
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=source, text=True).strip()
    subprocess.run(['git', 'diff', '--exit-code', PARENT, '--', *CORE_FILES], cwd=source, check=True)
    cfg = json.loads((RUN / 'config.json').read_text())
    assert cfg['source_commit'] == TRAINING_SOURCE and cfg['temperature'] == 3.
    assert cfg['discount'] == .999 and cfg['task'] == 'v3'
    checkpoint = RUN / 'checkpoint-final.pt'
    proof = json.loads((RUN / 'checkpoint-verification.json').read_text())
    assert proof['readback_verified'] and proof['steps'] == 258304
    with checkpoint.open('rb') as f:
        assert hashlib.file_digest(f, 'sha256').hexdigest() == proof['sha256']
    root = Path('/home/heechan/optiq-experiments') / CAMPAIGN
    environment = dict(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', XLA_PYTHON_CLIENT_PREALLOCATE='false',
        PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', PYTHONWARNINGS='ignore',
        MPLBACKEND='Agg', WANDB_MODE='online', USE_FLAX='0', USE_TORCH='1',
        D4RL_SUPPRESS_IMPORT_ERROR='1', MUJOCO_PY_FORCE_CPU='1',
        LD_LIBRARY_PATH='/home/heechan/.mujoco/mujoco210/bin',
        OPTIQ_SOURCE_DIR='/home/heechan/OptiQ-ops/sources/' + TRAINING_SOURCE)
    command = ['/usr/bin/nice', '-n', '19', '/home/heechan/.venv-ddiffpg-native/bin/python', '-u',
        str(source / 'antmaze_experiments/diagnose_rollout_horizon.py'), '--run', str(RUN),
        '--checkpoint', str(checkpoint), '--output', str(root / 'evaluation'),
        '--episodes', '100', '--extended-limit', '1400', '--cpu-offset', '48']
    manifest = dict(time=time.time(), host=args.host, service=CAMPAIGN, command=command,
        environment=environment, evaluation_source=sha, training_source=TRAINING_SOURCE,
        checkpoint_sha256=proof['sha256'], inference_only=True, primary_evaluation_preserved=True,
        native_limit=700, extended_limit=1400, episodes_per_condition=100,
        shared199='pending SSH recovery', credential_activation='/home/heechan/OptiQ-ops/activate.sh v5-direct-gmm',
        protocol='antmaze_experiments/ROLLOUT_HORIZON_DIAGNOSTIC_PROTOCOL.md')
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    conf = Path('/home/heechan/OptiQ-ops/supervisor/jobs') / (CAMPAIGN + '.conf')
    assert not root.exists() and not conf.exists()
    root.mkdir()
    launcher = root / 'launch.sh'
    launcher.write_text('#!/bin/bash\nset -euo pipefail\n'
        'source /home/heechan/OptiQ-ops/activate.sh v5-direct-gmm >/dev/null\n'
        + '\n'.join('export ' + k + '=' + shlex.quote(v) for k, v in environment.items())
        + '\ncd ' + shlex.quote(str(source)) + '\nexec ' + shlex.join(command) + '\n')
    conf.write_text(f'''[program:{CAMPAIGN}]
command=/bin/bash {launcher}
directory={source}
environment={','.join(k+'="'+v+'"' for k,v in environment.items())}
autostart=false
autorestart=false
startsecs=2
stopasgroup=true
killasgroup=true
stopwaitsecs=30
stdout_logfile={root}/evaluation.log
stderr_logfile={root}/evaluation.err
stdout_logfile_maxbytes=0
stderr_logfile_maxbytes=0
''')
    (root / 'registration.json').write_text(json.dumps(manifest, indent=2) + '\n')
    ctl = ['/usr/local/bin/supervisorctl', '-c', '/home/heechan/OptiQ-ops/supervisor/supervisord.conf']
    subprocess.run(ctl + ['reread'], check=True)
    subprocess.run(ctl + ['update', CAMPAIGN], check=True)
    subprocess.run(ctl + ['start', CAMPAIGN], check=True)
    print(json.dumps(dict(root=str(root), evaluation_source=sha, training_source=TRAINING_SOURCE)))


if __name__ == '__main__':
    main()
