"""Queue 12 OptiQ annealing jobs before verified missing dense baselines."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from .dependencies import prepare_dependencies,verify_dependencies
from .register_dense_off_16 import campaign_manifest as previous_manifest,HOSTS
from .settings import expected_updates,total_budget

CAMPAIGN='antmaze-dense-anneal-baselines-s0-20260924'
REFERENCE='antmaze-dense-off-16-current-s0-20260924'


def baseline_inventory(reference,tasks):
    status=json.loads((reference/'status.json').read_text())
    running={j['id'] for j in status['running']}
    inventory={}
    for task in tasks:
        for method in ('sac','dipo','mfpo'):
            key=f'{task}-{method}-s0';run=reference/'runs'/key
            result_path=run/'result.json';state='missing_or_incomplete'
            if result_path.exists():
                result=json.loads(result_path.read_text())
                config=json.loads((run/'config.json').read_text())
                proof=json.loads((run/'checkpoint-verification.json').read_text())
                assert result['completed'] and result['steps']==total_budget(task)
                assert result['updates']==expected_updates(total_budget(task))
                assert config['reward_profile']=='dense' and not config['noveld_enabled']
                assert proof['readback_verified'] and proof['environment_reward_verified']
                assert (run/'checkpoint-final.pt').stat().st_size==proof['bytes']
                assert len(result['summaries'])==4
                assert all(s['episodes']==100 for s in result['summaries'].values())
                state='completed'
            elif key in running:
                live=next(j for j in status['running'] if j['id']==key)
                assert Path(f'/proc/{live["pid"]}').exists(),f'Stale running status: {key}'
                state='already_running'
            inventory[key]=dict(task=task,method=method,status=state,reference=str(run),
                                result=str(result_path) if result_path.exists() else None)
    return inventory


def campaign_manifest(source,sha,shard,inventory):
    base=previous_manifest(source,sha,shard)
    prototypes={j['id']:j for j in base['jobs']}
    tasks=('v1','v3') if shard==0 else ('v2','v4')
    jobs=[]
    for target in (1.,.25,.5):
        for task in tasks:
            entry=dict(prototypes[f'{task}-optiq-s0'])
            entry.update(id=f'{task}-optiq-10to{target:g}-s0',temperature=10.,
                temperature_schedule=dict(enabled=True,final_temperature=target,
                    anneal_steps=1000000,decay='linear'),queue_class='annealing')
            jobs.append(entry)
    # Native budgets/settings; earlier completed and currently running policies
    # are never queued again. A cancelled run without full state starts fresh.
    for method in ('mfpo','sac','dipo'):
        for task in tasks:
            key=f'{task}-{method}-s0'
            if inventory[key]['status']=='missing_or_incomplete':
                jobs.append(dict(prototypes[key],queue_class='missing_baseline',
                    supersedes_incomplete=inventory[key]['reference']))
    base.update(campaign=CAMPAIGN,jobs=jobs,
        protocol='antmaze_experiments/ANNEAL_BASELINES_PROTOCOL.md',
        parent_source='a9f6a7b9e982a31d217d0697746262dfd024a1b6',
        optiq_temperature=10.,temperature_schedule=dict(initial=10.,targets=[1.,.25,.5],
            decay='linear',anneal_steps=1000000,exclude_warmup=True,hold_after_anneal=True),
        baseline_inventory=inventory,priority='annealing first, then missing baselines; independent per-slot backfill',
        wandb_mode='online' if shard==0 else 'offline',
        warmup_transitions=8192,base_total_budgets_preserved=True,
        existing_training_preserved=True)
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
    tasks=('v1','v3') if a.shard==0 else ('v2','v4')
    inventory=baseline_inventory(Path('/home/heechan/optiq-experiments')/REFERENCE,tasks)
    manifest=campaign_manifest(source,sha,a.shard,inventory)
    manifest['source_dependencies']=verify_dependencies(source,('mfpo',)) if a.dry_run else prepare_dependencies(source,('mfpo',))
    if a.dry_run:print(json.dumps(manifest,indent=2));return
    root=Path('/home/heechan/optiq-experiments')/CAMPAIGN
    assert not root.exists();root.mkdir()
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    ctl=['/usr/local/bin/supervisorctl','-c','/home/heechan/OptiQ-ops/supervisor/supervisord.conf']
    services=[]
    for suffix,module in [('', 'controller'),('-wandb-sync','sync_wandb')]:
        service=CAMPAIGN+suffix;conf=Path('/home/heechan/OptiQ-ops/supervisor/jobs')/(service+'.conf')
        assert not conf.exists()
        mode='online' if suffix else manifest['wandb_mode']
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
