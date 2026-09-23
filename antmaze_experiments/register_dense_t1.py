"""Register four fresh OptiQ T=1 dense/NovelD-off runs without changing defaults."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from .register_dense_off_16 import campaign_manifest as previous_manifest, HOSTS
from .dependencies import verify_dependencies

CAMPAIGN = 'antmaze-optiq-dense-off-T1-s0-20260924'


def campaign_manifest(source, sha, shard):
    manifest = previous_manifest(source, sha, shard)
    manifest['jobs'] = [entry for entry in manifest['jobs'] if entry['method'] == 'optiq']
    for entry in manifest['jobs']:
        entry['temperature'] = 1.0
    manifest.update(campaign=CAMPAIGN, optiq_temperature=1.0,
        protocol='antmaze_experiments/DENSE_T1_PROTOCOL.md',
        comparison_campaign='antmaze-dense-off-16-current-s0-20260924',
        comparison_source='a70e6bb1c59407200f7ff2a65cd09b21fff0c1f3',
        parent_source='a91bdd9eb77983d8c35c7ed2412018e69eb1bc1e',
        changed_learning_setting={'temperature': {'previous': .01, 'current': 1.0}},
        source_dependencies=verify_dependencies(source, ('optiq',)))
    manifest.pop('dipo_dense_value_support')
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
    conf = Path('/home/heechan/OptiQ-ops/supervisor/jobs') / (CAMPAIGN + '.conf')
    assert not root.exists(), root
    assert not conf.exists(), conf
    root.mkdir()
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    conf.write_text(f'''[program:{CAMPAIGN}]
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
    for command in (['reread'], ['update', CAMPAIGN], ['start', CAMPAIGN]):
        subprocess.run(ctl + command, check=True)
    (root / 'registration.json').write_text(json.dumps(dict(time=time.time(),
        supervisor=CAMPAIGN, manifest=manifest, config=str(conf)), indent=2) + '\n')
    print(json.dumps(dict(root=str(root), host=args.host, source_commit=sha,
                          jobs=[j['id'] for j in manifest['jobs']])))


if __name__ == '__main__':
    main()
