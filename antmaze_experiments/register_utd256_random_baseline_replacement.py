"""Re-run accidentally stopped random-start 100Δd v3/v4 controls from seed 0."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from .register_utd256_random_starts import campaign_manifest as original_manifest


CAMPAIGN = 'antmaze-optiq-utd1-random-progress100-baseline-replacement-v34-s0-20260925'
HOSTS = ('vast-heechan-180', 'vast-heechan-199')


def campaign_manifest(source, sha, host):
    if host not in HOSTS:
        raise ValueError(host)
    manifest = original_manifest(source, sha, host)
    job = dict(manifest['jobs'][0])
    job['id'] += '-replacement'
    assert job['temperature'] == 1. and job['optiq_config_profile'] == 'basic'
    assert job['train_starts'] == job['eval_starts'] == 'random'
    assert job['reward_specification']['formula'] == '100*(d(current)-d(next))'
    assert job['reward_specification']['step_cost'] == 0.
    assert job['collection_profile'] == 'env256-update256'
    manifest.update(
        campaign=CAMPAIGN, host=host, jobs=[job],
        protocol='antmaze_experiments/RANDOM_BASELINE_REPLACEMENT_PROTOCOL.md',
        comparison='Exact learning and evaluation configuration of the stopped random-start 100Δd control; fresh seed-0 restart because no complete replay/optimizer/RNG checkpoint exists.',
        replaces_incomplete_campaign='antmaze-optiq-utd1-basic-euclidean-random-starts-v34-s0-20260925',
        resumed_from_checkpoint=False, fresh_training=True,
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--host', choices=HOSTS, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    assert source == Path('/home/heechan/OptiQ-ops/sources') / sha
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                       cwd=source, text=True).strip()
    manifest = campaign_manifest(source, sha, args.host)
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    root = Path('/home/heechan/optiq-experiments') / CAMPAIGN
    conf_root = Path('/home/heechan/OptiQ-ops/supervisor/jobs')
    services = [(CAMPAIGN, 'controller'), (CAMPAIGN + '-wandb-sync', 'sync_wandb')]
    assert not root.exists(), root
    for name, _ in services:
        assert not (conf_root / (name + '.conf')).exists(), name
    root.mkdir()
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for name, module in services:
        (conf_root / (name + '.conf')).write_text(f'''[program:{name}]
command=/home/heechan/.venv-ddiffpg-native/bin/python -m antmaze_experiments.{module} --root {root}
directory={source}
environment=PYTHONDONTWRITEBYTECODE="1",WANDB_MODE="online",OPTIQ_CAMPAIGN="{CAMPAIGN}"
autostart=false
autorestart=false
startsecs=2
stopasgroup=true
killasgroup=true
stopwaitsecs=30
stdout_logfile={root}/{module}.log
stderr_logfile={root}/{module}.err
stdout_logfile_maxbytes=0
stderr_logfile_maxbytes=0
''')
    ctl = ['/usr/local/bin/supervisorctl', '-c',
           '/home/heechan/OptiQ-ops/supervisor/supervisord.conf']
    subprocess.run(ctl + ['reread'], check=True)
    for name, _ in services:
        subprocess.run(ctl + ['update', name], check=True)
        subprocess.run(ctl + ['start', name], check=True)
    (root / 'registration.json').write_text(json.dumps(dict(
        time=time.time(), host=args.host, services=[name for name, _ in services],
        source_commit=sha, jobs=[job['id'] for job in manifest['jobs']]), indent=2) + '\n')
    print(json.dumps(dict(root=str(root), host=args.host, source_commit=sha,
                          jobs=[job['id'] for job in manifest['jobs']])))


if __name__ == '__main__':
    main()
