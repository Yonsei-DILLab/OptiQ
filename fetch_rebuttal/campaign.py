"""Supervised parallel Fetch campaign with memory admission and temperature selection."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from .run import METHODS, save_json, guard

TEMPERATURES = (.05, .1, .25, .5, 1., 5., 10.)

def process_live(job):
    pid = job.get('pid')
    if not pid:
        return False
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[0] != 'Z' and int(fields[19]) == job['start_ticks']
    except FileNotFoundError:
        return False

def gpu_memory():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.total',
                                      '--format=csv,noheader,nounits'], text=True)
    return {int(i): (int(used), int(total)) for i, used, total in
            (line.split(',') for line in output.strip().splitlines())}

def memory_available_mib():
    values = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(values['MemAvailable'].strip().split()[0]) // 1024
    maximum = Path('/sys/fs/cgroup/memory.max')
    if maximum.exists() and maximum.read_text().strip() != 'max':
        current = int(Path('/sys/fs/cgroup/memory.current').read_text())
        available = min(available, (int(maximum.read_text())-current)//1024**2)
    return available

def environment(source, gpu):
    return dict(os.environ, PYTHONPATH=str(source), CUDA_VISIBLE_DEVICES=str(gpu),
        XLA_PYTHON_CLIENT_PREALLOCATE='false', MUJOCO_GL='egl', WANDB_MODE='disabled',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
        TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='1')

def command(job, python, smoke=False):
    result = [python, '-u', '-m', 'fetch_rebuttal.run', '--method', job['method'],
        '--task', job['task'], '--seed', str(job['seed']), '--temperature', str(job['temperature']),
        '--output', job['output']]
    if smoke:
        result.append('--smoke')
    return result

def make_plan(folder, source, python, host):
    tasks = ['pick-and-place']
    # Keep all 14 sweep runs on host 1 so automatic selection requires no
    # cross-host credentials or a local laptop remaining awake.
    specs = [(task, method, seed, 1.) for seed in range(5) for task in tasks
             for method in METHODS if method != 'ibolt'
             and ((method == 'sac') if host == 1 else (method != 'sac'))]
    # Interleave method/temperature families so every method starts promptly.
    if host == 1:
        specs += [(task, 'ibolt', seed, temperature) for seed in range(2)
                  for temperature in TEMPERATURES for task in tasks]
    specs.sort(key=lambda s: (s[2], METHODS.index(s[1]), s[3], tasks.index(s[0])))
    jobs = []
    for index, (task, method, seed, temperature) in enumerate(specs):
        name = f'{task}-{method}' + (f'-T{temperature:g}' if method == 'ibolt' else '') + f'-s{seed}'
        jobs.append(dict(id=name, task=task, method=method, seed=seed, temperature=temperature,
            phase='temperature-sweep' if method == 'ibolt' else 'baseline', gpu=index%4,
            output=str(folder/'runs'/name), status='queued'))
    initial_count=len(jobs)
    for method in METHODS:
        if method=='ibolt' or ((method=='sac') != (host==1)):
            continue
        previous='pick-and-place'
        for task in ('push','slide','reach'):
            for seed in range(5):
                name=f'{task}-{method}-s{seed}'
                jobs.append(dict(id=name,task=task,method=method,seed=seed,temperature=1.,
                    phase='fetch-followup',previous_task=previous,gpu=(METHODS.index(method)+seed)%4,
                    output=str(folder/'runs'/name),status='waiting_for_previous_task'))
            previous=task
    return dict(host=host, tasks=tasks, source=str(source), python=str(python), jobs=jobs,
        status='created', temperatures=list(TEMPERATURES), training_steps=1000000,
        initial_job_count=initial_count,followup_job_count=len(jobs)-initial_count,
        followup_order=['pick-and-place','push','slide','reach'],followup_preflights={},
        seeds_baseline=list(range(5)), seeds_temperature=[0, 1], final_seeds=list(range(5)),
        selections={}, created_unix=time.time(), gpu_headroom_mib=3072,
        selection_rule='Per task, mean final-step success over last 10 pre-final evaluations (950k-995k) and seeds 0/1; tie: success AUC; tie: closest to T=1. Require nonzero observed success. Selected seeds 0/1 reused, append 2/3/4. Label tuning overlap in final report.',
        isolated_environment=True, common_mujoco_defaults=True, her=False)

def preflight(plan, folder):
    source, python = Path(plan['source']), plan['python']
    jobs = []
    for task in ('pick-and-place',):
        for method in METHODS:
            name = f'{task}-{method}'
            jobs.append(dict(id=name, task=task, method=method, seed=100,
                temperature=1., gpu=len(jobs)%4, output=str(folder/'preflights'/name)))
    results = {}
    def lane(gpu):
        lane_results = []
        for job in (j for j in jobs if j['gpu'] == gpu):
            output = Path(job['output'])
            if not (output/'completed.json').exists():
                log = folder/'logs'/f'preflight-{job["id"]}.log'
                with log.open('w') as stream:
                    # One physical core + its sibling limits native thread pools.
                    cpu = plan['cpu_lanes'][gpu]
                    cmd = ['taskset','-c',','.join(map(str,cpu))] + command(job, python, smoke=True)
                    result = subprocess.run(cmd, cwd=source, env=environment(source, gpu),
                        stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f'GPU preflight failed: {job["id"]}: {log}')
            result = json.loads((output/'completed.json').read_text())
            if result['updates'] != 4 or result['runtime']['backend'] != 'gpu':
                raise RuntimeError('Preflight did not execute four GPU optimizer updates')
            lane_results.append((job['id'], result))
        return lane_results
    with ThreadPoolExecutor(max_workers=4) as workers:
        for lane_results in workers.map(lane, range(4)):
            results.update(lane_results)
    save_json(folder/'ready.json', dict(all_6_gpu_preflights_passed=True, results=results,
        checked_unix=time.time()))

def select_temperatures(plan, folder):
    for task in plan['tasks']:
        if task in plan['selections']:
            continue
        jobs = [j for j in plan['jobs'] if j['phase'] == 'temperature-sweep' and j['task'] == task]
        if len(jobs) != 14 or not all(j['status'] == 'completed' for j in jobs):
            continue
        scores = []
        for temperature in TEMPERATURES:
            rows = []
            for job in (j for j in jobs if j['temperature'] == temperature):
                records = [json.loads(line) for line in (Path(job['output'])/'evaluations.jsonl').read_text().splitlines()]
                # Exclude the 100-episode final evaluation from the equal-size
                # selection window. Ten checkpoints each contain ten episodes.
                records = [r for r in records if 0 < r['steps'] < 1000000]
                values = [r[r['primary_mode']]['success_rate'] for r in records]
                if len(values) != 199:
                    raise RuntimeError('Temperature selection has incomplete evaluations')
                rows.append(dict(seed=job['seed'], late_success=sum(values[-10:])/10,
                                 success_auc=sum(values)/len(values)))
            scores.append(dict(temperature=temperature, late_success=sum(r['late_success'] for r in rows)/2,
                               success_auc=sum(r['success_auc'] for r in rows)/2, seeds=rows))
        best = max(scores, key=lambda r: (r['late_success'],r['success_auc'],-abs(math.log(r['temperature']))))
        if best['late_success'] <= 0:
            plan['selections'][task] = dict(status='needs_review_no_late_success', scores=scores,
                reason='No observed late-window success; do not describe a temperature as good or promote automatically')
            continue
        plan['selections'][task] = dict(status='selected', temperature=best['temperature'],
            scores=scores, selected_unix=time.time(), reused_sweep_seeds=[0,1], added_seeds=[2,3,4],
            limitation='Seeds 0/1 participated in temperature selection; report this overlap.')
        for seed in (2,3,4):
            name = f'{task}-ibolt-T{best["temperature"]:g}-s{seed}'
            plan['jobs'].append(dict(id=name,task=task,method='ibolt',seed=seed,
                temperature=best['temperature'], phase='selected-5seed', gpu=(seed+plan['tasks'].index(task))%4,
                output=str(folder/'runs'/name),status='queued'))
        save_json(folder/f'{task}-temperature-selection.json', plan['selections'][task])

def summarize(plan, folder):
    groups = {}
    for job in plan['jobs']:
        if job['status'] != 'completed':
            continue
        records = [json.loads(line) for line in (Path(job['output'])/'evaluations.jsonl').read_text().splitlines()]
        record = next(r for r in records if r['steps'] == 1000000)
        key = job['task']+'-'+job['method']+(f'-T{job["temperature"]:g}' if job['method']=='ibolt' else '')
        groups.setdefault(key,[]).append(dict(seed=job['seed'],
            success_rate=record[record['primary_mode']]['success_rate'],
            any_success_rate=record[record['primary_mode']]['any_success_rate'],
            mean_return=record[record['primary_mode']]['mean_return'], output=job['output']))
    summary = {}
    for key, rows in groups.items():
        values = [r['success_rate'] for r in rows]
        mean = sum(values)/len(values)
        summary[key] = dict(n_seeds=len(rows), mean_success=mean,
            sample_sd_success=(sum((x-mean)**2 for x in values)/(len(values)-1))**.5 if len(values)>1 else None,
            per_seed=rows)
    save_json(folder/'metrics.json', dict(groups=summary,selections=plan['selections'],
        note='Seed-level means/sample SD. Selected iBOLT seeds 0/1 were also tuning seeds.',updated_unix=time.time()))

def release_followups(plan,folder):
    groups={}
    for job in plan['jobs']:
        if job['status']=='waiting_for_previous_task':
            groups.setdefault((job['task'],job['method'],job['previous_task']),[]).append(job)
    for (task,method,previous),jobs in groups.items():
        upstream=[j for j in plan['jobs'] if j['task']==previous and j['method']==method]
        if len(upstream)!=5 or not all(j['status']=='completed' for j in upstream):
            continue
        key=task+'-'+method
        prior=plan['followup_preflights'].get(key,{})
        if prior.get('status')=='failed':
            continue
        memory=gpu_memory()
        gpu=max(memory,key=lambda g:memory[g][1]-memory[g][0])
        if memory[gpu][1]-memory[gpu][0]<plan['reserves_mib'][method]+3072:
            continue
        smoke=dict(id=key,task=task,method=method,seed=100,temperature=1.,gpu=gpu,
            output=str(folder/'preflights'/('followup-'+key)))
        result_path=Path(smoke['output'])/'completed.json'
        try:
            if not result_path.exists():
                log=folder/'logs'/('preflight-followup-'+key+'.log')
                active=[j for j in plan['jobs'] if j['status'] in ('initializing','running')]
                lane=min(plan['cpu_lanes'],key=lambda c:sum(j.get('cpus')==c for j in active))
                cmd=['taskset','-c',','.join(map(str,lane))]+command(smoke,plan['python'],smoke=True)
                with log.open('x') as stream:
                    subprocess.run(cmd,cwd=plan['source'],env=environment(plan['source'],gpu),
                        stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,check=True)
            result=json.loads(result_path.read_text())
            if result['updates']!=4 or result['runtime']['backend']!='gpu':
                raise RuntimeError('Followup GPU preflight failed validation')
            plan['followup_preflights'][key]=dict(status='completed',result=result)
            for job in jobs:
                job.update(status='queued',released_unix=time.time())
            print('Released five seeds: '+key+' after '+previous,flush=True)
        except Exception as error:
            plan['followup_preflights'][key]=dict(status='failed',error=repr(error))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args()
    guard()
    path=args.plan.resolve();folder=path.parent
    lock=(folder/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads(path.read_text())
    source=Path(plan['source'])
    for name in ('logs','runs','preflights'):
        (folder/name).mkdir(exist_ok=True)
    plan.update(status='validating_gpu',controller_pid=os.getpid())
    save_json(path,plan)
    try:
        if not (folder/'ready.json').exists():
            preflight(plan,folder)
        plan['status']='running'
        children={}
        last_summary=0
        while True:
            for job in plan['jobs']:
                output=Path(job['output'])
                if job['status'] not in ('running','initializing'):
                    continue
                if (output/'completed.json').exists():
                    result=json.loads((output/'completed.json').read_text())
                    if not result['completed'] or result['updates']!=995000 or result['runtime']['backend']!='gpu':
                        raise RuntimeError('Invalid completed training result: '+job['id'])
                    job.update(status='completed',finished_unix=result['finished_unix'])
                elif (output/'failure.json').exists() or not process_live(job):
                    job.update(status='failed',finished_unix=time.time(),
                        error=(output/'failure.json').read_text() if (output/'failure.json').exists() else 'Child exited without completion')
                elif (output/'progress.json').exists():
                    job['progress']=json.loads((output/'progress.json').read_text())
                    if job['progress'].get('updates',0)>=1:
                        job['status']='running'
            for pid,child in list(children.items()):
                if child.poll() is not None:
                    child.wait(); del children[pid]
            select_temperatures(plan,folder)
            release_followups(plan,folder)
            memory=gpu_memory();available=memory_available_mib()
            active=[j for j in plan['jobs'] if j['status'] in ('running','initializing')]
            for job in plan['jobs']:
                if job['status']!='queued':
                    continue
                gpu=job['gpu'];used,total=memory[gpu]
                # Account for compilation and real-batch allocations that an
                # initializing process has not yet made visible to nvidia-smi.
                pending=sum(plan['reserves_mib'][j['method']] for j in active
                    if j['gpu']==gpu and j['status']=='initializing')
                reserve=plan['reserves_mib'][job['method']]
                if used+pending+reserve>total-plan['gpu_headroom_mib'] or available<4096:
                    job['waiting_reason']='memory headroom';continue
                # Bound simultaneous native compiler starts, not training runs.
                if sum(j['status']=='initializing' and j['gpu']==gpu for j in active)>=3:
                    job['waiting_reason']='compilation admission';continue
                cpu=min(plan['cpu_lanes'],key=lambda c:sum(j.get('cpus')==c for j in active))
                cmd=['taskset','-c',','.join(map(str,cpu))]+command(job,plan['python'])
                log=folder/'logs'/f'{job["id"]}.log'
                with log.open('x') as stream:
                    child=subprocess.Popen(cmd,cwd=source,env=environment(source,gpu),
                        stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                fields=Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()
                job.update(status='initializing',pid=child.pid,start_ticks=int(fields[19]),
                    started_unix=time.time(),cpus=cpu,command=cmd)
                children[child.pid]=child;active.append(job);available-=2048
                print('Started '+job['id']+' GPU'+str(gpu),flush=True)
            counts={s:sum(j['status']==s for j in plan['jobs']) for s in ('waiting_for_previous_task','queued','initializing','running','completed','failed')}
            plan.update(counts=counts,checked_unix=time.time(),gpu_memory=memory)
            if time.time()-last_summary>=60:
                summarize(plan,folder);last_summary=time.time()
            save_json(path,plan)
            if not counts['queued'] and not counts['initializing'] and not counts['running']:
                plan['status']=('blocked_followups' if counts['waiting_for_previous_task'] else
                    'completed' if not counts['failed'] else 'completed_with_failures')
                save_json(path,plan);summarize(plan,folder);break
            time.sleep(10)
    except BaseException as error:
        plan.update(status='controller_error',error=repr(error));save_json(path,plan)
        raise

if __name__=='__main__':
    main()
