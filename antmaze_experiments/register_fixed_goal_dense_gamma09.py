"""Run the fixed-start nearest-goal dense OptiQ comparison at gamma=0.9."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from .register_fixed_goal_dense import campaign_manifest as control_manifest, HOSTS, TASKS, GPU


CAMPAIGN = 'antmaze-optiq-fixed-start-nearest-dense-gamma09-v34-1m-s0-20260925'
CONTROL_CAMPAIGN = 'antmaze-optiq-fixed-start-nearest-dense-v34-1m-s0-20260925-r2'
CONTROL_SOURCE = 'c3a2668cc180d4713204181ec09b87aecbe7e52e'


def campaign_manifest(source, sha, shard):
    manifest = control_manifest(source, sha, shard)
    task = TASKS[shard]
    job = manifest['jobs'][0]
    job['id'] = f'{task}-optiq-fixed-start-nearest-dense-gamma09-T1-nm64-s0'
    job['discount'] = 0.9
    manifest.update(
        campaign=CAMPAIGN,
        protocol='antmaze_experiments/FIXED_START_NEAREST_DENSE_GAMMA09_PROTOCOL.md',
        control_campaign=CONTROL_CAMPAIGN,
        control_source_commit=CONTROL_SOURCE,
        control_discount=0.99,
        discount=0.9,
        comparison='Only OptiQ discount gamma changes from 0.99 to 0.9; '
                   'fresh seed0 training, not a checkpoint resume.',
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--shard', type=int, choices=tuple(HOSTS), required=True)
    parser.add_argument('--host', choices=tuple(HOSTS.values()), required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    assert HOSTS[args.shard] == args.host
    source = Path(__file__).resolve().parents[1]
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    assert source == Path('/home/heechan/OptiQ-ops/sources') / sha
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                       cwd=source, text=True).strip()
    manifest = campaign_manifest(source, sha, args.shard)
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return

    root = Path('/home/heechan/optiq-experiments') / CAMPAIGN
    conf_root = Path('/home/heechan/OptiQ-ops/supervisor/jobs')
    service = CAMPAIGN
    assert not root.exists(), root
    assert not (conf_root / f'{service}.conf').exists(), service
    root.mkdir()
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (conf_root / f'{service}.conf').write_text(f'''[program:{service}]
command=/home/heechan/.venv-ddiffpg-native/bin/python -m antmaze_experiments.controller --root {root}
directory={source}
environment=PYTHONDONTWRITEBYTECODE="1",WANDB_MODE="online",OPTIQ_CAMPAIGN="{CAMPAIGN}"
autostart=false
autorestart=false
startsecs=2
stopasgroup=true
killasgroup=true
stopwaitsecs=30
stdout_logfile={root}/controller.log
stderr_logfile={root}/controller.err
stdout_logfile_maxbytes=0
stderr_logfile_maxbytes=0
''')
    ctl = ['/usr/local/bin/supervisorctl', '-c',
           '/home/heechan/OptiQ-ops/supervisor/supervisord.conf']
    subprocess.run(ctl + ['reread'], check=True)
    subprocess.run(ctl + ['update', service], check=True)
    subprocess.run(ctl + ['start', service], check=True)
    (root / 'registration.json').write_text(json.dumps(dict(
        time=time.time(), host=args.host, service=service, source_commit=sha,
        jobs=[job['id'] for job in manifest['jobs']], eligible_gpus=[GPU]), indent=2) + '\n')
    print(json.dumps(dict(root=str(root), host=args.host, service=service,
                          source_commit=sha, jobs=[job['id'] for job in manifest['jobs']],
                          eligible_gpus=[GPU])))


if __name__ == '__main__':
    main()
