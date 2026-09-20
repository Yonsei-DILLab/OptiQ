"""Supervisor-managed per-environment queues with a complete temperature gate."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CAMPAIGN = 'trg-temp-beta-20260921'
ROOT = Path('/home/heechan/optiq-experiments')/CAMPAIGN
TEMPERATURES = {'ant': (0.1, 0.05), 'humanoid': (0.5, 0.1)}
TOTAL = 1_000_000
WINDOW = 100_000
PYTHON = '/home/heechan/.venv-optiq-mujoco/bin/python'


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def source():
    sha = subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
    assert subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip() == 'direct-gmm-trg'
    return sha


def summarize_run(run_dir):
    result = {}
    expected = np.arange(TOTAL-WINDOW+5000,TOTAL+1,5000)
    for mode in ('stochastic_z','zero_z'):
        paths = list(run_dir.rglob(f'evaluations_{mode}.npz'))
        assert len(paths) == 1, (run_dir,mode,paths)
        with np.load(paths[0],allow_pickle=False) as d:
            steps = d['timesteps']; rewards = d['results']
            assert steps[-1] == TOTAL and rewards.shape == (len(steps),10)
            assert np.isfinite(rewards).all()
            mask = (steps > TOTAL-WINDOW) & (steps <= TOTAL)
            np.testing.assert_array_equal(steps[mask],expected)
            result[f'{mode}_last_100k_mean'] = float(rewards[mask].mean())
            result[f'{mode}_final_mean'] = float(rewards[-1].mean())
            result[f'{mode}_learning_curve_mean'] = float(np.trapz(rewards.mean(axis=1),steps)/(steps[-1]-steps[0]))
    result.update(window_start_exclusive=TOTAL-WINDOW,window_end_inclusive=TOTAL,evaluations=20,episodes_per_evaluation=10)
    return result


def job_name(task, stage, temp, beta, seed):
    return f'{task}-trg-{stage}-T{temp:g}-b{beta:g}-s{seed}'


def job_plan(task,stage,temps,betas):
    return [dict(task=task,stage=stage,temperature=t,beta=b,seed=s,
                 name=job_name(task,stage,t,b,s))
            for s in range(5) for t in temps for b in betas]


def run_stage(task, stage, temps, betas, sha):
    plan=job_plan(task,stage,temps,betas)
    pending=[]; complete=[]; failed=[]; active={}
    for job in plan:
        path=ROOT/'jobs'/(job['name']+'.json')
        if path.exists():
            prior=json.loads(path.read_text())
            if prior['status']=='completed':
                assert prior['commit']==sha
                prior['metrics']=summarize_run(Path(prior['run_dir']))
                complete.append(prior)
            else:
                failed.append(prior)  # Never duplicate or silently restart a partial run.
        else:
            pending.append(job)
    if failed:
        raise RuntimeError(f'Incomplete/failed records require review: {[j["name"] for j in failed]}')
    while pending or active:
        for gpu,entry in list(active.items()):
            process,job,log=entry
            code=process.poll()
            if code is None:
                continue
            log.close();del active[gpu]
            job.update(exit_code=code,finished=time.time(),status='failed')
            try:
                paths=list((ROOT/'outputs').glob(job['name']+'_*/completed.json'))
                assert code==0 and len(paths)==1, (code,paths)
                record=json.loads(paths[0].read_text())
                assert record['timesteps']==TOTAL and record['updates']==TOTAL-5000
                job.update(record,run_dir=str(paths[0].parent),metrics=summarize_run(paths[0].parent),status='completed')
                complete.append(job)
            except Exception as error:
                job['error']=repr(error);failed.append(job)
            save(ROOT/'jobs'/(job['name']+'.json'),job)
            print(json.dumps({'event':'finished','name':job['name'],'status':job['status'],'gpu':gpu}),flush=True)
        for gpu in range(4):
            if gpu in active or not pending:
                continue
            job=pending.pop(0)
            args=['/home/heechan/OptiQ-ops/run-gpu.sh',str(gpu),'--branch','v5-direct-gmm',
                  PYTHON,'-u',str(HERE/'train_sweep.py'),task,str(job['temperature']),str(job['beta']),
                  str(job['seed']),stage,str(ROOT/'outputs')]
            env=os.environ.copy()
            env.update(OPTIQ_SOURCE_DIR=str(REPO),CAMPAIGN_GPU=str(gpu),PYTHONDONTWRITEBYTECODE='1',WANDB_MODE='online')
            log=(ROOT/'logs'/(job['name']+'.log')).open('x')
            process=subprocess.Popen(args,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
            job.update(status='running',pid=process.pid,gpu=gpu,commit=sha,started=time.time(),command=args)
            save(ROOT/'jobs'/(job['name']+'.json'),job)
            active[gpu]=(process,job,log)
            print(json.dumps({'event':'started','name':job['name'],'gpu':gpu}),flush=True)
        save(ROOT/f'status-{task}.json',dict(task=task,phase=stage,updated=time.time(),commit=sha,
             running=[j[1]['name'] for j in active.values()],pending=[j['name'] for j in pending],
             completed=[j['name'] for j in complete],failed=[j['name'] for j in failed]))
        if pending or active:
            time.sleep(10)
    if failed:
        raise RuntimeError(f'{stage} failed; beta progression is blocked: {[j["name"] for j in failed]}')
    assert len(complete)==len(plan)
    return complete


def rank_temperature(jobs):
    rows=[]
    for temp in sorted({j['temperature'] for j in jobs}):
        group=sorted([j for j in jobs if j['temperature']==temp],key=lambda j:j['seed'])
        assert [j['seed'] for j in group]==list(range(5))
        assert all(j['status']=='completed' and j['beta']==1 for j in group)
        primary=[j['metrics']['stochastic_z_last_100k_mean'] for j in group]
        secondary=[j['metrics']['zero_z_last_100k_mean'] for j in group]
        rows.append(dict(temperature=temp,stochastic_z_mean=float(np.mean(primary)),
                         stochastic_z_seed_std=float(np.std(primary,ddof=1)),per_seed_stochastic_z=primary,
                         zero_z_mean=float(np.mean(secondary)),runs=[j['wandb_url'] for j in group]))
    assert len(rows)==2
    rows.sort(key=lambda row:(-row['stochastic_z_mean'],-row['zero_z_mean'],row['temperature']))
    return dict(metric='stochastic_z_last_100k_mean',window='900000 < env_steps <= 1000000',
                selected_temperature=rows[0]['temperature'],ranking=rows,
                tie_break='zero_z last-100k mean, then smaller temperature; only exact primary ties',
                scope='best among the two tested temperatures; same seeds reused for beta comparisons')


def run(task):
    sha=source()
    manifest=json.loads((ROOT/f'manifest-{task}.json').read_text())
    assert manifest['commit']==sha and manifest['task']==task
    try:
        temperatures=run_stage(task,'temperature',TEMPERATURES[task],(1.,),sha)
        selection=rank_temperature(temperatures)
        save(ROOT/f'selection-{task}.json',selection)
        chosen=selection['selected_temperature']
        print(json.dumps({'event':'selected','task':task,**selection}),flush=True)
        betas=run_stage(task,'beta',(chosen,),(.5,.9),sha)
        rows=[]
        for beta in (1.,.5,.9):
            group=[j for j in temperatures+betas if j['temperature']==chosen and j['beta']==beta]
            group.sort(key=lambda j:j['seed']);assert [j['seed'] for j in group]==list(range(5))
            values=[j['metrics']['stochastic_z_last_100k_mean'] for j in group]
            rows.append(dict(beta=beta,stochastic_z_mean=float(np.mean(values)),seed_std=float(np.std(values,ddof=1)),
                             per_seed=values,zero_z_mean=float(np.mean([j['metrics']['zero_z_last_100k_mean'] for j in group])),
                             runs=[j['wandb_url'] for j in group]))
        save(ROOT/f'result-{task}.json',dict(task=task,commit=sha,selection=selection,beta_comparison=rows))
        save(ROOT/f'status-{task}.json',dict(task=task,phase='completed',updated=time.time(),commit=sha,total_runs=20))
    except Exception as error:
        save(ROOT/f'failure-{task}.json',dict(error=repr(error),time=time.time(),commit=sha))
        raise


def setup(task):
    sha=source()
    ROOT.mkdir(exist_ok=True)
    for folder in ('jobs','logs','outputs'):
        (ROOT/folder).mkdir(exist_ok=True)
    path=ROOT/f'manifest-{task}.json';assert not path.exists()
    save(path,dict(campaign=CAMPAIGN,task=task,source=str(REPO),commit=sha,branch='direct-gmm-trg',
        project='OptiQ/gmm-trg',seeds=list(range(5)),temperatures=TEMPERATURES[task],temperature_stage_beta=1,
        beta_stage=[.5,.9],selection_metric='stochastic_z_last_100k_mean',selection_window=[900000,1000000],
        selection_start_exclusive=True,total_steps=TOTAL,workers=4,algorithm_defaults='unchanged except temperature and beta',
        deferred=['DACER','NM ablation','density correction removal','other environments'],
        phase1=job_plan(task,'temperature',TEMPERATURES[task],(1.,))))
    name=f'{CAMPAIGN}-{task}'
    conf=Path('/home/heechan/OptiQ-ops/supervisor/jobs')/(name+'.conf')
    assert not conf.exists()
    conf.write_text(f'''[program:{name}]
command={PYTHON} -u {HERE}/campaign.py run {task}
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
    ctl=['supervisorctl','-c','/home/heechan/OptiQ-ops/supervisor/supervisord.conf']
    subprocess.run(ctl+['reread'],check=True)
    subprocess.run(ctl+['update',name],check=True)
    print(json.dumps({'service':name,'root':str(ROOT),'commit':sha}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['setup','run'])
    parser.add_argument('task',choices=TEMPERATURES);args=parser.parse_args()
    (setup if args.command=='setup' else run)(args.task)
