"""MFPO entropy/initial-temperature sweep with preservation of other Fetch jobs."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from .campaign import environment, gpu_memory, memory_available_mib, process_live
from .her_campaign import TASKS, completed_task as original_completed_task
from .her_campaign import run_preflight as original_preflight
from .run import guard, save_json

COEFFICIENTS = (-.5, -.25, 0.)
INITIAL_TEMPERATURES = (.01, .1, 1.)
ACTIVE = ('initializing', 'running')


def name(task, coeff, initial, seed, phase):
    return f'{task}-mfpo-{phase}-H{coeff:g}-A{initial:g}-s{seed}'


def make_jobs(folder, source):
    jobs = []
    for seed in (0, 1):
        for coeff in COEFFICIENTS:
            for initial in INITIAL_TEMPERATURES:
                ident = name(TASKS[0], coeff, initial, seed, 'grid')
                jobs.append(dict(id=ident, method='mfpo', task=TASKS[0], seed=seed,
                    target_entropy_coeff=coeff, initial_temperature=initial, temperature=initial,
                    phase='mfpo-hyperparameter-sweep', source=str(source),
                    gpu=len(jobs)%4, output=str(folder/'runs'/ident), status='queued'))
    for task in TASKS:
        for seed in range(5):
            ident = f'{task}-mfpo-selected-s{seed}'
            jobs.append(dict(id=ident, method='mfpo', task=task, seed=seed,
                phase='mfpo-selected-5seed', source=str(source), temperature=None,
                target_entropy_coeff=None, initial_temperature=None, gpu=seed%4,
                output=str(folder/'runs'/ident), status='waiting_for_selection'))
    return jobs


def command(job, python, smoke=False):
    cmd = [python, '-u', '-m', 'fetch_rebuttal.run', '--method', job['method'],
        '--task', job['task'], '--seed', str(job['seed']), '--temperature', str(job['temperature']),
        '--batch-size', '256', '--normalize-observations', '--her', '--output', job['output']]
    if job['method'] == 'mfpo':
        cmd += ['--mfpo-target-entropy-coeff', str(job['target_entropy_coeff']),
                '--mfpo-initial-temperature', str(job['initial_temperature'])]
    return cmd + (['--smoke'] if smoke else [])


def preflight(plan, folder, job, gpu, cpus):
    key = f'{job["task"]}-mfpo-H{job["target_entropy_coeff"]:g}-A{job["initial_temperature"]:g}'
    if plan['mfpo_preflights'].get(key, {}).get('status') == 'completed':
        return
    probe = dict(job, seed=100, output=str(folder/'preflights'/key))
    output = Path(probe['output'])
    if not (output/'completed.json').exists():
        with (folder/'logs'/('preflight-'+key+'.log')).open('a') as stream:
            subprocess.run(['taskset', '-c', ','.join(map(str, cpus))]+command(probe,plan['python'],True),
                cwd=job['source'],env=environment(job['source'],gpu),stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,check=True)
    result=json.loads((output/'completed.json').read_text())
    cfg=json.loads((output/'config.json').read_text())
    native=json.loads((output/'native-config.json').read_text())
    progress=json.loads((output/'progress.json').read_text())
    assert result['updates']==4 and result['runtime']['backend']=='gpu'
    assert cfg['batch_size']==256 and cfg['HER'] and cfg['normalize_observations']
    assert native['temp']==job['initial_temperature']
    assert native['target_entropy_coeff']==job['target_entropy_coeff']
    her=result['runtime']['her_replay_last_sample']
    assert her['relabeled_samples']==204 and her['sampled_future_within_episode']
    metrics=[json.loads(line) for line in (output/'learner-metrics.jsonl').read_text().splitlines()]
    assert metrics[-1]['updates']==4
    assert abs(metrics[-1]['temperature']/job['initial_temperature']-1)<.01
    assert all(math.isfinite(v) for row in metrics for v in row.values())
    plan['mfpo_preflights'][key]=dict(status='completed',gpu=gpu,result=result,
        initial_temperature=native['temp'],target_entropy_coeff=native['target_entropy_coeff'],
        last_learner_metrics=metrics[-1])


def validate_grid(plan, folder):
    probes=[j for j in plan['jobs'] if j['phase']=='mfpo-hyperparameter-sweep' and j['seed']==0]
    def lane(gpu):
        for index, job in enumerate(probes):
            if index%4==gpu:
                preflight(plan,folder,job,gpu,plan['cpu_lanes'][gpu])
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lane,range(4)))
    save_json(folder/'grid-ready.json',dict(checked_unix=time.time(),preflights=plan['mfpo_preflights']))


def choose(plan, folder):
    if plan.get('mfpo_selection'):
        return
    jobs=[j for j in plan['jobs'] if j['phase']=='mfpo-hyperparameter-sweep']
    assert len(jobs)==18
    if not all(j['status']=='completed' for j in jobs):
        return
    scores=[]
    for coeff in COEFFICIENTS:
        for initial in INITIAL_TEMPERATURES:
            members=[j for j in jobs if j['target_entropy_coeff']==coeff and j['initial_temperature']==initial]
            assert {j['seed'] for j in members}=={0,1}
            seeds=[]
            for job in members:
                records=[json.loads(line) for line in (Path(job['output'])/'evaluations.jsonl').read_text().splitlines()]
                records=sorted((r for r in records if 0<r['steps']<1000000),key=lambda r:r['steps'])
                assert [r['steps'] for r in records]==list(range(5000,1000000,5000))
                values=[r[r['primary_mode']]['success_rate'] for r in records]
                late=records[-10:]
                seeds.append(dict(seed=job['seed'],late_success=sum(values[-10:])/10,
                    success_auc=sum(values)/len(values),late_new_final_rate=sum(
                        r[r['primary_mode']]['new_final_success_count']/r[r['primary_mode']]['episodes'] for r in late)/10))
            scores.append(dict(target_entropy_coeff=coeff,initial_temperature=initial,seeds=seeds,
                late_success=sum(s['late_success'] for s in seeds)/2,
                late_new_final_rate=sum(s['late_new_final_rate'] for s in seeds)/2,
                success_auc=sum(s['success_auc'] for s in seeds)/2))
    best=max(scores,key=lambda s:(s['late_success'],s['late_new_final_rate'],s['success_auc'],
        -abs(s['target_entropy_coeff']+.5),-abs(math.log(s['initial_temperature']/.01))))
    selection=dict(status='selected',task=TASKS[0],**best,scores=scores,selected_unix=time.time(),
        validated_new_goal_success=best['late_new_final_rate']>0,
        rerun_seeds=list(range(5)),fresh_initialization=True,
        limitation='Seeds 0/1 participated in tuning; fresh reruns are not independent validation seeds. If all settings fail, the best rank remains unvalidated.',
        transfer='Use the PickAndPlace-selected setting for Push, Slide, Reach; no per-task MFPO tuning authorized.')
    plan['mfpo_selection']=selection
    save_json(folder/'mfpo-hyperparameter-selection.json',selection)
    for job in plan['jobs']:
        if job['phase']!='mfpo-selected-5seed':continue
        job.update(target_entropy_coeff=best['target_entropy_coeff'],initial_temperature=best['initial_temperature'],
            temperature=best['initial_temperature'],
            status='queued' if job['task']==TASKS[0] else 'waiting_for_previous_task')
    print('Selected MFPO '+json.dumps(best),flush=True)


def completed_task(plan, method, task):
    if method!='mfpo':return original_completed_task(plan,method,task)
    jobs=[j for j in plan['jobs'] if j['method']=='mfpo' and j['task']==task and j['phase']=='mfpo-selected-5seed']
    return len(jobs)==5 and {j['seed'] for j in jobs}==set(range(5)) and all(j['status']=='completed' for j in jobs)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
    path=args.plan.resolve();folder=path.parent;guard()
    lock=(folder/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for directory in ('logs','runs','preflights'):(folder/directory).mkdir(exist_ok=True)
    plan=json.loads(path.read_text());plan.update(controller_pid=os.getpid(),status='validating_gpu')
    save_json(path,plan)
    if args.preflight_only:
        validate_grid(plan,folder);plan['status']='prepared';save_json(path,plan);return
    def shutdown(signum, _frame):
        # Adopted jobs retain their old process group. Stop only verified job PIDs.
        for job in plan['jobs']:
            if job['status'] in ACTIVE and process_live(job):
                try:os.kill(job['pid'],signal.SIGTERM)
                except ProcessLookupError:pass
        plan['status']='stopped';plan['stopped_unix']=time.time();save_json(path,plan)
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,shutdown);signal.signal(signal.SIGINT,shutdown)
    children={}
    try:
        validate_grid(plan,folder);plan['status']='running'
        while True:
            for job in plan['jobs']:
                if job['status'] not in ACTIVE:continue
                output=Path(job['output'])
                if (output/'completed.json').exists():
                    result=json.loads((output/'completed.json').read_text())
                    assert result['completed'] and result['updates']==995000 and result['runtime']['backend']=='gpu'
                    job.update(status='completed',finished_unix=result['finished_unix'])
                elif (output/'failure.json').exists() or not process_live(job):
                    job.update(status='failed',finished_unix=time.time(),error=(output/'failure.json').read_text()
                        if (output/'failure.json').exists() else 'Exited without completion')
                elif (output/'progress.json').exists():
                    job['progress']=json.loads((output/'progress.json').read_text())
                    if job['progress'].get('updates',0)>0:job['status']='running'
            for pid, child in list(children.items()):
                if child.poll() is not None:child.wait();del children[pid]
            choose(plan,folder)
            for method in plan['methods']:
                for previous,task in zip(TASKS,TASKS[1:]):
                    waiting=[j for j in plan['jobs'] if j['method']==method and j['task']==task
                             and j['status']=='waiting_for_previous_task']
                    if not waiting or not completed_task(plan,method,previous):continue
                    memory=gpu_memory();gpu=max(memory,key=lambda g:memory[g][1]-memory[g][0])
                    if memory[gpu][1]-memory[gpu][0]<plan['reserves_mib'][method]+plan['gpu_headroom_mib']:continue
                    active=[j for j in plan['jobs'] if j['status'] in ACTIVE]
                    cpu=min(plan['cpu_lanes'],key=lambda c:sum(j.get('cpus')==c for j in active))
                    if method=='mfpo':preflight(plan,folder,waiting[0],gpu,cpu)
                    else:original_preflight(plan,folder,method,task,gpu,cpu)
                    for job in waiting:job.update(status='queued',released_unix=time.time())
            memory=gpu_memory();available=memory_available_mib()
            active=[j for j in plan['jobs'] if j['status'] in ACTIVE]
            for job in plan['jobs']:
                if job['status']!='queued':continue
                gpu=job['gpu'];used,total=memory[gpu]
                pending=sum(plan['reserves_mib'][j['method']] for j in active if j['gpu']==gpu and j['status']=='initializing')
                if used+pending+plan['reserves_mib'][job['method']]>total-plan['gpu_headroom_mib'] or available<plan['host_reserve_mib']:
                    job['waiting_reason']='memory headroom';continue
                if sum(j['status']=='initializing' and j['gpu']==gpu for j in active)>=plan['max_initializing_per_gpu']:
                    job['waiting_reason']='initialization admission';continue
                cpu=min(plan['cpu_lanes'],key=lambda c:sum(j.get('cpus')==c for j in active))
                source=job.get('source',plan['source'])
                cmd=['taskset','-c',','.join(map(str,cpu))]+command(job,plan['python'])
                logfolder=Path(job.get('log_folder',str(folder/'logs')));logfolder.mkdir(exist_ok=True)
                with (logfolder/(job['id']+'.log')).open('x') as stream:
                    child=subprocess.Popen(cmd,cwd=source,env=environment(source,gpu),stdin=subprocess.DEVNULL,
                        stdout=stream,stderr=subprocess.STDOUT)
                fields=Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()
                job.update(status='initializing',pid=child.pid,start_ticks=int(fields[19]),started_unix=time.time(),
                    cpus=cpu,command=cmd);job.pop('waiting_reason',None)
                children[child.pid]=child;active.append(job);available-=2048
                print('Started '+job['id']+' GPU'+str(gpu),flush=True)
            counts=dict(Counter(j['status'] for j in plan['jobs']))
            plan.update(counts=counts,checked_unix=time.time(),gpu_memory=memory)
            save_json(path,plan)
            if not any(counts.get(s,0) for s in (*ACTIVE,'queued')):
                blocked=any(counts.get(s,0) for s in ('waiting_for_selection','waiting_for_previous_task'))
                plan['status']='blocked_followups' if blocked else 'completed_with_failures' if counts.get('failed') else 'completed'
                save_json(path,plan);break
            time.sleep(10)
    except Exception as error:
        plan.update(status='controller_error',error=repr(error));save_json(path,plan);raise


if __name__=='__main__':main()
