"""Register approved OptiQ distance x success-bonus factorial, v1-v4 seed0."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import time

from .register_dacer_off import campaign_manifest as control_manifest
from .register_dense_off_16 import HOSTS

CAMPAIGN = 'antmaze-optiq-progress-2x2-T1-s0-20260924'
CONTROL_SOURCE = '484f92e7d6d34c964d85b4493ff17c5a9ebcf32e'
TASKS = ('v1', 'v2', 'v3', 'v4')
PROFILES = ('progress_euclidean', 'progress_euclidean_no_bonus',
            'progress_geodesic', 'progress_geodesic_no_bonus')
# Each host has all four conditions: 32M/28M native steps. Long mazes first.
SHARDS = {0: ('v4', 'v1'), 1: ('v3', 'v2')}


def campaign_manifest(source, sha, shard):
    from .progress_reward import specification
    if shard not in SHARDS: raise ValueError('shard must be 0 or 1')
    manifest = control_manifest(source, sha, shard)
    prototypes = {entry['task']: entry for part in (0, 1)
                  for entry in control_manifest(source, sha, part)['jobs']}
    jobs = []
    for task in SHARDS[shard]:
        for profile in PROFILES:
            entry = copy.deepcopy(prototypes[task])
            entry.update(id=f'{task}-optiq-{profile}-T1-s0', reward_profile=profile,
                         reward_specification=specification(task,profile), temperature=1.,
                         dacer='off', noveld='off')
            jobs.append(entry)
    manifest.update(campaign=CAMPAIGN,jobs=jobs,
        protocol='antmaze_experiments/PROGRESS_FACTORIAL_PROTOCOL.md',
        parent_source='85d2de8c447da84600b6a06ecfac451ee2117d51',
        comparison_campaign='antmaze-optiq-dense-dacer-off-T1-s0-20260924',
        comparison_source=CONTROL_SOURCE,
        changed_learning_setting={'reward_profile': list(PROFILES)},
        reward_profiles=list(PROFILES), reward='distance progress -0.01; 2x2 distance/bonus ablation',
        replay_capacity=1000000,beta=1.,gradient_clipping=None,
        priority='Independent GPU backfill every 2 seconds; no maze/condition barrier')
    manifest.pop('cancelled_campaign',None)
    manifest.pop('reward_profile',None)
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
