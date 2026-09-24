"""Register four Euclidean progress-only OptiQ runs after cancelling geodesic jobs."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import time
from .register_dacer_off import campaign_manifest as control_manifest
from .progress_reward import EUCLIDEAN_NO_COST_PROFILE as NO_COST_PROFILE, specification

HOSTS = {0: 'vast-heechan-180', 1: 'vast-heechan-199'}
CAMPAIGN = 'antmaze-optiq-euclidean-no-step-B0-T1-s0-20260925-r2'
SHARDS = {0: ('v3', 'v4'), 1: ('v1', 'v2')}


def campaign_manifest(source, sha, shard):
    if shard not in SHARDS: raise ValueError(shard)
    manifest = control_manifest(source, sha, shard)
    prototypes = {j['task']:j for part in (0,1)
                  for j in control_manifest(source,sha,part)['jobs']}
    jobs=[]
    for task in SHARDS[shard]:
        entry=copy.deepcopy(prototypes[task])
        entry.update(id=f'{task}-optiq-euclidean-no-step-B0-T1-s0',
                     reward_profile=NO_COST_PROFILE,reward_specification=specification(task,NO_COST_PROFILE),
                     temperature=1.,dacer='off',noveld='off')
        jobs.append(entry)
    manifest.update(campaign=CAMPAIGN,host=HOSTS[shard],shard=shard,jobs=jobs,
        protocol='antmaze_experiments/EUCLIDEAN_NO_STEP_PROTOCOL.md',
        parent_source='cab48faf07d41a86886355bfd6bef538fff29432',
        supersedes_failed_preflight_campaign='antmaze-optiq-euclidean-no-step-B0-T1-s0-20260925',
        comparison_source='4d757159ef84e849250a6d5111f9931b2b392b21',
        comparison_campaign='antmaze-optiq-geodesic-no-step-B0-T1-s0-20260924',
        changed_learning_setting={'distance': {'previous':'geodesic','current':'Euclidean'}},
        reward='100 * nearest-goal Euclidean distance decrease; B=0; step penalty=0',
        evaluation_starts='upstream: v1 random XY[-2,2]; v2-v4 fixed original full state at XY[0,0]',
        reward_profile=NO_COST_PROFILE,replay_capacity=1000000,beta=1.,gradient_clipping=None,
        priority='v3, v4, v1, v2; independent GPU slots; no cross-maze barrier')
    manifest.pop('cancelled_campaign',None)
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
