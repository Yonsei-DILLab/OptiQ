"""Register the approved 15-policy signed DACER entropy sweep at fixed T=1."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import time

from .dependencies import verify_dependencies
from .register_dense_off_16 import campaign_manifest as previous_manifest,HOSTS

CAMPAIGN='antmaze-dense-dacer-entropy-T1-s0-20260924'
TARGETS=(-1.,-.8,-.5,-.3,-.1)
TASKS=('v1','v3','v4')


def campaign_manifest(source,sha,shard,wandb_mode='online'):
    base=previous_manifest(source,sha,shard)
    prototypes={entry['task']:entry for part in (0,1)
                for entry in previous_manifest(source,sha,part)['jobs'] if entry['method']=='optiq'}
    jobs=[]
    for entropy in TARGETS:
        for task in TASKS:
            entry=copy.deepcopy(prototypes[task])
            entry.update(id=f'{task}-optiq-H{entropy:g}-T1-s0',temperature=1.,
                         dacer_target_entropy_per_dim=entropy)
            jobs.append(entry)
    # Round-robin across equal four-5090 hosts: 8/7 jobs, 32M/28M transitions.
    # Each local slot backfills independently; no task or entropy barrier.
    base.update(campaign=CAMPAIGN,jobs=jobs[shard::2],wandb_mode=wandb_mode,
        protocol='antmaze_experiments/DACER_ENTROPY_PROTOCOL.md',
        optiq_temperature=1.,dacer_target_entropy_per_dim=list(TARGETS),
        dacer_target_entropy=[8*h for h in TARGETS],dacer_action_dim=8,
        dacer_enabled=True,dacer_behavior_only=True,
        parent_source='05fc113aeb093f9f83208ca540dd2aadf8d934bb',
        comparison_campaign='antmaze-optiq-dense-off-T1-s0-20260924',
        comparison_source='7af193833466f5bc41853948d6f417fc6aaa6803',
        comparison_target_entropy_per_dim=-.9,
        changed_learning_setting='Only DACER target_entropy_per_dim; fixed temperature1',
        warmup_transitions=8192,base_total_budgets_preserved=True,
        priority='independent per-slot backfill in manifest order',
        source_dependencies=verify_dependencies(source,('optiq',)))
    base.pop('dipo_dense_value_support')
    return base


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--shard',type=int,choices=[0,1],required=True)
    p.add_argument('--host',choices=list(HOSTS.values()),required=True)
    p.add_argument('--wandb-mode',choices=['online','offline'],default='online')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();assert HOSTS[a.shard]==a.host
    source=Path(__file__).resolve().parents[1]
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    assert source==Path('/home/heechan/OptiQ-ops/sources')/sha
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=source,text=True).strip()
    manifest=campaign_manifest(source,sha,a.shard,a.wandb_mode)
    if a.dry_run:print(json.dumps(manifest,indent=2));return
    root=Path('/home/heechan/optiq-experiments')/CAMPAIGN
    assert not root.exists();root.mkdir()
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    ctl=['/usr/local/bin/supervisorctl','-c','/home/heechan/OptiQ-ops/supervisor/supervisord.conf']
    services=[]
    for suffix,module in [('', 'controller'),('-wandb-sync','sync_wandb')]:
        service=CAMPAIGN+suffix
        conf=Path('/home/heechan/OptiQ-ops/supervisor/jobs')/(service+'.conf')
        assert not conf.exists()
        mode='online' if suffix else a.wandb_mode
        conf.write_text(f'''[program:{service}]
command=/home/heechan/.venv-ddiffpg-native/bin/python -m antmaze_experiments.{module} --root {root}
directory={source}
environment=PYTHONDONTWRITEBYTECODE="1",WANDB_MODE="{mode}",OPTIQ_CAMPAIGN="{CAMPAIGN}"
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
        services.append(service)
    subprocess.run(ctl+['reread'],check=True)
    for service in services:
        subprocess.run(ctl+['update',service],check=True)
        subprocess.run(ctl+['start',service],check=True)
    (root/'registration.json').write_text(json.dumps(dict(time=time.time(),services=services,manifest=manifest),indent=2)+'\n')
    print(json.dumps(dict(root=str(root),jobs=[j['id'] for j in manifest['jobs']],source_commit=sha)))


if __name__=='__main__':main()
