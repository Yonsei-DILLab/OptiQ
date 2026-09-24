"""Register the approved temperature-only DACER-off OptiQ comparison."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import time

from .register_dacer_off import campaign_manifest as control_manifest
from .register_dense_off_16 import HOSTS

CAMPAIGN = 'antmaze-optiq-dense-dacer-off-T3-T5-T10-s0-20260924'
CONTROL_SOURCE = '484f92e7d6d34c964d85b4493ff17c5a9ebcf32e'
TEMPERATURES = (3., 5., 10.)
TASKS = ('v1', 'v3', 'v4')
# Both hosts receive 18M native transitions. Start eight independent jobs;
# the ninth v1 job backfills the first available local slot on host180.
SHARDS = {
    0: (('v4', 3.), ('v3', 5.), ('v1', 3.), ('v1', 5.), ('v1', 10.)),
    1: (('v4', 5.), ('v4', 10.), ('v3', 3.), ('v3', 10.)),
}


def campaign_manifest(source, sha, shard):
    if shard not in SHARDS:
        raise ValueError('shard must be 0 or 1')
    manifest = control_manifest(source, sha, shard)
    prototypes = {entry['task']: entry for part in (0, 1)
                  for entry in control_manifest(source, sha, part)['jobs']}
    jobs = []
    for task, temperature in SHARDS[shard]:
        entry = copy.deepcopy(prototypes[task])
        entry.update(id=f'{task}-optiq-dacer-off-T{temperature:g}-s0',
                     temperature=temperature)
        jobs.append(entry)
    manifest.update(
        campaign=CAMPAIGN, jobs=jobs,
        protocol='antmaze_experiments/DACER_OFF_TEMPERATURE_PROTOCOL.md',
        parent_source=CONTROL_SOURCE,
        comparison_campaign='antmaze-optiq-dense-dacer-off-T1-s0-20260924',
        comparison_source=CONTROL_SOURCE,
        optiq_temperatures=list(TEMPERATURES),
        changed_learning_setting={'temperature': {'previous': 1., 'current': list(TEMPERATURES)}},
        replay_capacity=1000000, beta=1., gradient_clipping=None,
        policy_temperature_schedule=None,
        priority='Independent local GPU backfill every 2 seconds; no maze/temperature barrier',
    )
    manifest.pop('optiq_temperature', None)
    manifest.pop('cancelled_campaign', None)
    return manifest


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--shard', type=int, choices=[0, 1], required=True)
    parser.add_argument('--host', choices=list(HOSTS.values()), required=True)
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
    specs = [(CAMPAIGN, 'controller'), (CAMPAIGN + '-wandb-sync', 'sync_wandb')]
    conf_root = Path('/home/heechan/OptiQ-ops/supervisor/jobs')
    assert not root.exists(), root
    for service, _ in specs:
        assert not (conf_root / (service + '.conf')).exists(), service
    root.mkdir()
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for service, module in specs:
        (conf_root / (service + '.conf')).write_text(f'''[program:{service}]
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
    for service, _ in specs:
        subprocess.run(ctl + ['update', service], check=True)
        subprocess.run(ctl + ['start', service], check=True)
    (root / 'registration.json').write_text(json.dumps(dict(
        time=time.time(), services=[s for s, _ in specs], manifest=manifest), indent=2) + '\n')
    print(json.dumps(dict(root=str(root), host=args.host, source_commit=sha,
                         jobs=[entry['id'] for entry in manifest['jobs']])))


if __name__ == '__main__':
    main()
