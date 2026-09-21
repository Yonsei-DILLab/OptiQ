"""DACER first, then beta, with immediate per-GPU backfill and no stage barrier."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import time
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CAMPAIGN = 'trg-dacer-priority-20260921'
ROOT = Path('/home/heechan/optiq-experiments')/CAMPAIGN
OLD = ROOT.parent/'trg-temp-beta-20260921'
OLD_SHA = '84f1e0a884349d6c4b0dae521839a8d4e5f46437'
OLD_SOURCE = Path('/home/heechan/OptiQ-ops/sources')/OLD_SHA
PYTHON = '/home/heechan/.venv-optiq-mujoco/bin/python'
CTL = ['supervisorctl','-c','/home/heechan/OptiQ-ops/supervisor/supervisord.conf']
spec = importlib.util.spec_from_file_location('old_campaign', HERE.parent/'20260921_temp_beta/campaign.py')
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
save = old.save


def verified(job, root):
    paths = list((root/'outputs').glob(job['name']+'_*'))
    assert len(paths) == 1, (job['name'], paths)
    run = paths[0]; cfg = json.loads((run/'config.json').read_text())
    a = cfg['alg']['actor']
    assert cfg['seed'] == job['seed'] and cfg['task'] == job['task']
    assert a['temperature'] == job['temperature'] and a['density_beta'] == job['beta']
    assert cfg['runtime']['git_commit'] == job['commit']
    assert bool(cfg.get('dacer',{}).get('enabled',False)) == (job['stage']=='dacer')
    metrics = old.summarize_run(run)
    for kind in ('actor','critic'):
        files = list(run.rglob(f'{kind}_state_1000000.msgpack'))
        assert len(files)==1 and files[0].stat().st_size > 1000
    completion = run/'completed.json'
    warning = None
    if completion.exists():
        record = json.loads(completion.read_text())
        assert record['timesteps']==1_000_000 and record['updates']==995_000
        url = record['wandb_url']
    else:
        log = (root/'logs'/(job['name']+'.log')).read_text()
        assert 'run.log_artifact(artifact)' in log and 'Failed to verify credentials' in log
        assert 'service process is busy' in log
        matches = re.findall(r'https://wandb.ai/OptiQ/gmm-trg/runs/[A-Za-z0-9]+', log)
        assert matches
        url = matches[-1]
        warning = 'Full 1M evaluation/checkpoints verified; W&B artifact upload failed after training. Original failure retained.'
    return dict(job, status='completed', run_dir=str(run), metrics=metrics, wandb_url=url,
                logging_warning=warning, verified_at=time.time())


def select_completed(jobs, task):
    temps = old.TEMPERATURES[task]
    cohorts = [{j['seed'] for j in jobs if j['temperature']==t} for t in temps]
    seeds = sorted(set.intersection(*cohorts)); assert seeds, 'No matched completed seeds'
    rows=[]
    for t in temps:
        group=sorted([j for j in jobs if j['temperature']==t and j['seed'] in seeds],key=lambda j:j['seed'])
        assert len(group)==len(seeds)
        values=[j['metrics']['stochastic_z_last_100k_mean'] for j in group]
        rows.append(dict(temperature=t, seeds=seeds, stochastic_z_mean=float(np.mean(values)),
            seed_std=float(np.std(values,ddof=1)) if len(values)>1 else None,
            per_seed=values, zero_z_mean=float(np.mean([j['metrics']['zero_z_last_100k_mean'] for j in group])),
            runs=[j['wandb_url'] for j in group]))
    rows.sort(key=lambda r:(-r['stochastic_z_mean'],-r['zero_z_mean'],r['temperature']))
    return dict(selected_temperature=rows[0]['temperature'], ranking=rows, matched_seeds=seeds,
        metric='stochastic_z_last_100k_mean', window='900000 < steps <= 1000000; 20 evaluations x 10 episodes',
        authorization='User canceled unfinished temperature sweep; select from already completed runs only.')


def ordered_pending(jobs):
    return sorted([j for j in jobs if j['status']=='queued'],key=lambda j:(j['priority'],j['seed'],j['beta']))


def setup(task):
    sha=old.source()
    old_service=f'trg-temp-beta-20260921-{task}'
    state=subprocess.run(CTL+['status',old_service],text=True,capture_output=True).stdout
    assert 'STOPPED' in state or 'EXITED' in state, state
    ROOT.mkdir(exist_ok=True)
    for folder in ('jobs','logs','outputs'): (ROOT/folder).mkdir(exist_ok=True)
    assert not (ROOT/f'manifest-{task}.json').exists()
    completed=[]; interrupted=[]
    for path in sorted((OLD/'jobs').glob(task+'-*.json')):
        job=json.loads(path.read_text())
        try: completed.append(verified(job,OLD))
        except (AssertionError,FileNotFoundError,KeyError,ValueError) as error:
            if job['status']=='completed': raise
            interrupted.append(dict(job, migration_status='canceled' if job['stage']=='temperature' else 'restart_after_dacer',
                                    verification_error=str(error)))
    temperatures=[j for j in completed if j['stage']=='temperature']
    selection=select_completed(temperatures,task)
    save(ROOT/f'selection-{task}.json',selection)
    chosen=selection['selected_temperature']
    jobs=old.job_plan(task,'dacer',(.25,),(1.,))+old.job_plan(task,'beta',(chosen,),(.5,.9))
    done={j['name']:j for j in completed if j['stage']=='beta'}
    for job in jobs:
        job.update(status='queued',priority=0 if job['stage']=='dacer' else 1,
                   commit=sha if job['stage']=='dacer' else OLD_SHA)
        if job['name'] in done: job.update(done[job['name']], imported=True)
        prior=[j for j in interrupted if j['name']==job['name']]
        if prior: job['interrupted_attempt']=prior[0]
        save(ROOT/'jobs'/(job['name']+'.json'),job)
    manifest=dict(task=task, commit=sha, source=str(REPO), beta_source=str(OLD_SOURCE), beta_commit=OLD_SHA,
        campaign=CAMPAIGN, project='OptiQ/gmm-trg', selection=selection,
        imported_temperature=temperatures, preserved_old_attempts=interrupted,
        jobs=[j['name'] for j in jobs], workers=4, order='DACER seeds 0..4 then beta .5/.9 seeds 0..4; no completion barrier',
        restart_policy='Partial beta attempts preserved, restart fresh 1M; no checkpoint resume', created=time.time())
    save(ROOT/f'manifest-{task}.json',manifest)
    confdir=Path('/home/heechan/OptiQ-ops/supervisor/jobs')
    prior=confdir/(old_service+'.conf')
    backup=ROOT/(old_service+'.conf.original'); backup.write_text(prior.read_text())
    prior.write_text(prior.read_text().replace('autostart=true','autostart=false'))
    name=f'{CAMPAIGN}-{task}'; conf=confdir/(name+'.conf'); assert not conf.exists()
    conf.write_text(f'''[program:{name}]
command={PYTHON} -u {HERE}/controller.py run {task}
directory={REPO}
autostart=true
autorestart=false
startsecs=3
startretries=0
stopasgroup=true
killasgroup=true
stopwaitsecs=60
redirect_stderr=true
stdout_logfile={ROOT}/controller-{task}.log
stdout_logfile_maxbytes=10MB
stdout_logfile_backups=2
environment=PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2"
''')
    subprocess.run(CTL+['reread'],check=True)
    subprocess.run(CTL+['update',old_service],check=True)
    subprocess.run(CTL+['update',name],check=True)
    print(json.dumps(dict(service=name,selection=selection,imported_beta=list(done))),flush=True)


def run(task):
    lock=(ROOT/f'controller-{task}.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads((ROOT/f'manifest-{task}.json').read_text())
    assert old.source()==manifest['commit']
    jobs=[json.loads((ROOT/'jobs'/(n+'.json')).read_text()) for n in manifest['jobs']]
    assert not any(j['status']=='running' for j in jobs), 'Interrupted runs require review; refusing duplicate launch'
    pending=ordered_pending(jobs); active={}
    while pending or active:
        for gpu,(process,job,log) in list(active.items()):
            code=process.poll()
            if code is None: continue
            log.close(); del active[gpu]
            job.update(exit_code=code,finished=time.time(),status='failed')
            try: job.update(verified(job,ROOT))
            except Exception as error: job['error']=repr(error)
            save(ROOT/'jobs'/(job['name']+'.json'),job)
            print(json.dumps(dict(event='finished',name=job['name'],status=job['status'],gpu=gpu)),flush=True)
        for gpu in range(4):
            if gpu in active or not pending: continue
            job=pending.pop(0); dacer=job['stage']=='dacer'
            source=REPO if dacer else OLD_SOURCE
            args=['/home/heechan/OptiQ-ops/run-gpu.sh',str(gpu),'--branch','v5-direct-gmm',PYTHON,'-u']
            if dacer: args += [str(HERE/'train_dacer.py'),task,str(job['seed']),str(ROOT/'outputs')]
            else: args += [str(OLD_SOURCE/'analysis_tools/experiments/20260921_temp_beta/train_sweep.py'),
                         task,str(job['temperature']),str(job['beta']),str(job['seed']),'beta',str(ROOT/'outputs')]
            env=os.environ.copy(); env.update(OPTIQ_SOURCE_DIR=str(source),CAMPAIGN_GPU=str(gpu),
                PYTHONDONTWRITEBYTECODE='1',WANDB_MODE='online',PYTHONPATH=str(ROOT/'deps'))
            log=(ROOT/'logs'/(job['name']+'.log')).open('x')
            process=subprocess.Popen(args,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
            job.update(status='running',pid=process.pid,gpu=gpu,started=time.time(),command=args)
            save(ROOT/'jobs'/(job['name']+'.json'),job); active[gpu]=(process,job,log)
            print(json.dumps(dict(event='started',name=job['name'],gpu=gpu)),flush=True)
        save(ROOT/f'status-{task}.json',dict(task=task,commit=manifest['commit'],updated=time.time(),
            phase='running' if active or pending else 'failed' if any(j['status']=='failed' for j in jobs) else 'completed',
            **{state:[j['name'] for j in jobs if j['status']==state] for state in ('running','queued','completed','failed')}))
        if any(j['status']=='failed' for j in jobs):
            save(ROOT/f'failure-{task}.json',dict(failed=[j for j in jobs if j['status']=='failed'],updated=time.time()))
        if pending or active: time.sleep(2)
    save(ROOT/f'result-{task}.json',dict(selection=manifest['selection'],jobs=jobs,
        imported_temperature=manifest['imported_temperature'],
        comparison_note='Report full five-seed beta/DACER and matched completed-temperature cohort for beta=1 comparisons.'))
    assert all(j['status']=='completed' for j in jobs), 'Some runs failed; no automatic retry'


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=('setup','run'))
    parser.add_argument('task',choices=('ant','humanoid')); args=parser.parse_args()
    (setup if args.command=='setup' else run)(args.task)
