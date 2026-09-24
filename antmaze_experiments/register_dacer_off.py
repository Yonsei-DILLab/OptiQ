"""Register four fresh T=1 OptiQ policies with behavior DACER disabled."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from .register_dense_t1 import campaign_manifest as fixed_t1
from .register_dense_off_16 import HOSTS

CAMPAIGN='antmaze-optiq-dense-dacer-off-T1-s0-20260924'


def campaign_manifest(source,sha,shard):
    base=fixed_t1(source,sha,shard)
    for entry in base['jobs']:
        entry.update(id=f'{entry["task"]}-optiq-dacer-off-T1-s0',dacer='off')
    base.update(campaign=CAMPAIGN,protocol='antmaze_experiments/DACER_OFF_PROTOCOL.md',
        dacer_enabled=False,dacer_behavior_noise_std=0.,
        parent_source='4a33d13f0a8de3be508537fd38879ac8ef8018c0',
        comparison_campaign='antmaze-optiq-dense-off-T1-s0-20260924',
        comparison_source='7af193833466f5bc41853948d6f417fc6aaa6803',
        changed_learning_setting={'dacer.enabled':{'previous':True,'current':False}},
        cancelled_campaign='antmaze-dense-dacer-entropy-T1-s0-20260924')
    return base


def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--shard',type=int,choices=[0,1],required=True)
    p.add_argument('--host',choices=list(HOSTS.values()),required=True)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();assert HOSTS[a.shard]==a.host
    source=Path(__file__).resolve().parents[1]
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    assert source==Path('/home/heechan/OptiQ-ops/sources')/sha
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=source,text=True).strip()
    manifest=campaign_manifest(source,sha,a.shard)
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
        conf.write_text(f'''[program:{service}]
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
        services.append(service)
    subprocess.run(ctl+['reread'],check=True)
    for service in services:
        subprocess.run(ctl+['update',service],check=True)
        subprocess.run(ctl+['start',service],check=True)
    (root/'registration.json').write_text(json.dumps(dict(time=time.time(),services=services,manifest=manifest),indent=2)+'\n')
    print(json.dumps(dict(root=str(root),jobs=[j['id'] for j in manifest['jobs']],source_commit=sha)))


if __name__=='__main__':main()
