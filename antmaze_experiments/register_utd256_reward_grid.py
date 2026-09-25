"""Queue four reward-only AntMaze comparisons behind the running UTD=1 control."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import time

from .progress_reward import (
    EUCLIDEAN_SCALE10_PROFILE,
    EUCLIDEAN_SCALE100_COST01_PROFILE,
    EUCLIDEAN_SCALE10_COST01_PROFILE,
    specification,
)
from .register_utd256 import campaign_manifest as control_manifest


CAMPAIGN = 'antmaze-optiq-utd1-reward-grid-v1234-s0-20260925'
PREDECESSOR = 'antmaze-optiq-utd1-basic-euclidean-v1234-s0-20260925'
REWARDS = (
    ('progress10', EUCLIDEAN_SCALE10_PROFILE),
    ('progress100_cost01', EUCLIDEAN_SCALE100_COST01_PROFILE),
    ('progress10_cost01', EUCLIDEAN_SCALE10_COST01_PROFILE),
    ('negative_distance', 'dense'),
)
HOSTS = {0: 'vast-heechan-180', 1: 'vast-heechan-199', 2: 'vast1'}


def campaign_manifest(source, sha, shard):
    if shard not in HOSTS:
        raise ValueError(shard)
    base = control_manifest(source, sha, 0 if shard == 2 else shard, 'basic_euclidean')
    originals = {job['task']: job for control_shard in (0, 1)
                 for job in control_manifest(source, sha, control_shard, 'basic_euclidean')['jobs']}
    base.pop('jobs')
    jobs = []
    for label, reward in REWARDS:
        if (shard == 2) != (reward == 'dense'):
            continue
        tasks = ('v1', 'v2', 'v3', 'v4') if shard == 2 else (('v1', 'v3') if shard == 0 else ('v2', 'v4'))
        for task in tasks:
            original = originals[task]
            entry = copy.deepcopy(original)
            entry['id'] = f"{entry['task']}-optiq-utd1-{label}-s0"
            entry['reward_profile'] = reward
            entry.pop('reward_specification', None)
            if reward != 'dense':
                entry['reward_specification'] = specification(entry['task'], reward)
            jobs.append(entry)
    base.update(campaign=CAMPAIGN, jobs=jobs, shard=shard, host=HOSTS[shard],
                condition='reward_grid',
                comparison='Reward-only variants of the running basic_euclidean UTD=1 control',
                protocol='antmaze_experiments/UTD256_REWARD_GRID_PROTOCOL.md',
                reward_profiles={label: reward for label, reward in REWARDS},
                priority_campaign=(f'/home/heechan/optiq-experiments/{PREDECESSOR}' if shard != 2 else None),
                launch_policy='Protect running control GPUs; fill each eligible idle GPU immediately, then backfill independently')
    base.pop('reward_profile', None)
    base.pop('excluded_hosts', None)
    base['vast1_exception'] = 'User explicitly approved eligible idle 4090 GPUs for this reward queue; GMM40 workers and locks stay intact'
    assert len(jobs) == (4 if shard == 2 else 6) and len({j['id'] for j in jobs}) == len(jobs)
    return base


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
        time=time.time(), host=args.host, services=[n for n, _ in services],
        source_commit=sha, jobs=[j['id'] for j in manifest['jobs']]), indent=2) + '\n')
    print(json.dumps(dict(root=str(root), host=args.host, source_commit=sha,
                          jobs=[j['id'] for j in manifest['jobs']])))


if __name__ == '__main__':
    main()
