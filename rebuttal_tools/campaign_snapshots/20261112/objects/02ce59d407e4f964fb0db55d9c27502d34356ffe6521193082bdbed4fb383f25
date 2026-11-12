"""Per-method sequential Fetch tasks with shared future HER and batch 256."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import time
from .campaign import environment, gpu_memory, memory_available_mib, process_live, summarize
from .run import save_json, guard

TASKS = ('pick-and-place', 'push', 'slide', 'reach')
TEMPERATURES = (.05, .1, .25, .5, 1., 5., 10.)


def command(job, python, smoke=False):
    cmd = [python, '-u', '-m', 'fetch_rebuttal.run', '--method', job['method'],
           '--task', job['task'], '--seed', str(job['seed']), '--temperature', str(job['temperature']),
           '--batch-size', '256', '--normalize-observations', '--her', '--output', job['output']]
    return cmd + (['--smoke'] if smoke else [])


def make_plan(folder, source, python, host):
    methods = ['ibolt', 'sac'] if host == 1 else ['idac', 'idac-nodgn', 'mfpo', 'meow']
    jobs = []
    for task in TASKS:
        specs = [(m, s, t) for m in methods for s in (range(2) if m == 'ibolt' else range(5))
                 for t in (TEMPERATURES if m == 'ibolt' else (1.,))]
        specs.sort(key=lambda x: (x[1], methods.index(x[0]), x[2]))
        for index, (method, seed, temperature) in enumerate(specs):
            name = f'{task}-{method}' + (f'-T{temperature:g}' if method == 'ibolt' else '') + f'-s{seed}'
            gpu = (index % 4 if host == 1 else (seed + methods.index(method)) % 4)
            jobs.append(dict(id=name, task=task, method=method, seed=seed, temperature=temperature,
                phase='temperature-sweep' if method == 'ibolt' else 'baseline', gpu=gpu,
                output=str(folder/'runs'/name),
                status='queued' if task == TASKS[0] else 'waiting_for_previous_task'))
    return dict(host=host, methods=methods, tasks=list(TASKS), source=str(source), python=str(python),
        jobs=jobs, status='prepared', selections={}, preflights={}, created_unix=time.time(),
        training_steps=1000000, batch_size=256, utd=1, warmup=5000, her=True,
        normalize_observations=True, normalize_reward=False, final_seeds=list(range(5)),
        temperatures=list(TEMPERATURES), seeds_temperature=[0,1], followup_order=list(TASKS),
        gpu_headroom_mib=3072, host_reserve_mib=4096, max_initializing_per_gpu=2,
        reserves_mib=dict(ibolt=3072, sac=1536, idac=2560, **{'idac-nodgn':2048}, mfpo=2048, meow=2048),
        selection_rule='Per environment: maximize mean final-step success across last 10 pre-final evaluations and sweep seeds 0/1; ties: AUC then temperature closest to 1. Reuse selected seeds 0/1 and add 2/3/4. Complete all five before next task. If all candidates fail, label the tied selection unvalidated and still finish the requested five-seed benchmark.',
        failed_run_policy='Preserve failure evidence and block only the affected method task chain; no hidden parameter changes or automatic seed replacement.')


def run_preflight(plan, folder, method, task, gpu, cpus):
    key = task+'-'+method
    if plan['preflights'].get(key, {}).get('status') == 'completed':
        return
    job = dict(id=key, method=method, task=task, seed=100, temperature=1., gpu=gpu,
               output=str(folder/'preflights'/key))
    output = Path(job['output'])
    if not (output/'completed.json').exists():
        log = folder/'logs'/('preflight-'+key+'.log')
        with log.open('a') as stream:
            subprocess.run(['taskset','-c',','.join(map(str,cpus))]+command(job,plan['python'],True),
                cwd=plan['source'], env=environment(plan['source'],gpu), stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, check=True)
    result = json.loads((output/'completed.json').read_text())
    cfg = json.loads((output/'config.json').read_text())
    if result['updates'] != 4 or result['runtime']['backend'] != 'gpu':
        raise RuntimeError('Preflight must execute four actual GPU updates: '+key)
    if cfg['batch_size'] != 256 or not cfg['normalize_observations'] or not cfg['HER']:
        raise RuntimeError('Wrong preflight configuration: '+key)
    norm = json.loads((output/'checkpoints'/f'normalizer_{cfg["steps"]:09d}.json').read_text())
    if norm['count'] < cfg['steps'] or norm['settings']['norm_reward']:
        raise RuntimeError('Missing or wrong normalization statistics: '+key)
    her = result['runtime']['her_replay_last_sample']
    assert her['relabeled_samples'] == 204 and her['sampled_future_within_episode']
    plan['preflights'][key] = dict(status='completed', result=result, normalization_count=norm['count'])


def select_temperatures(plan, folder):
    if 'ibolt' not in plan['methods']:
        return
    for task in TASKS:
        if task in plan['selections']:
            continue
        jobs = [j for j in plan['jobs'] if j['task']==task and j['phase']=='temperature-sweep']
        if not all(j['status']=='completed' for j in jobs):
            continue
        scores = []
        for temperature in TEMPERATURES:
            per_seed = []
            for job in (j for j in jobs if j['temperature']==temperature):
                records = [json.loads(s) for s in (Path(job['output'])/'evaluations.jsonl').read_text().splitlines()]
                records = [r for r in records if 0 < r['steps'] < 1000000]
                assert len(records)==199
                values = [r[r['primary_mode']]['success_rate'] for r in records]
                per_seed.append(dict(seed=job['seed'], late_success=sum(values[-10:])/10,
                    success_auc=sum(values)/len(values),
                    late_new_final_success_count=sum(r[r['primary_mode']]['new_final_success_count'] for r in records[-10:])))
            scores.append(dict(temperature=temperature, late_success=sum(x['late_success'] for x in per_seed)/2,
                success_auc=sum(x['success_auc'] for x in per_seed)/2,
                late_new_final_success_count=sum(x['late_new_final_success_count'] for x in per_seed), seeds=per_seed))
        best = max(scores, key=lambda x:(x['late_success'],x['success_auc'],-abs(math.log(x['temperature']))))
        plan['selections'][task] = dict(status='selected', temperature=best['temperature'], scores=scores,
            selected_unix=time.time(), reused_sweep_seeds=[0,1], added_seeds=[2,3,4],
            evidence_of_new_late_success=best['late_new_final_success_count']>0,
            limitation='Seeds 0/1 participated in tuning. Selection is a ranking, not proof of adequate success.')
        for seed in (2,3,4):
            name = f'{task}-ibolt-T{best["temperature"]:g}-s{seed}'
            plan['jobs'].append(dict(id=name,task=task,method='ibolt',seed=seed,temperature=best['temperature'],
                phase='selected-5seed',gpu=seed%4,output=str(folder/'runs'/name),status='queued'))
        save_json(folder/f'{task}-temperature-selection.json',plan['selections'][task])


def completed_task(plan, method, task):
    jobs = [j for j in plan['jobs'] if j['method']==method and j['task']==task]
    if method=='ibolt':
        selection=plan['selections'].get(task)
        if not selection:
            return False
        jobs=[j for j in jobs if j['temperature']==selection['temperature']]
    return len(jobs)==5 and {j['seed'] for j in jobs}==set(range(5)) and all(j['status']=='completed' for j in jobs)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True)
    path=parser.parse_args().plan.resolve();folder=path.parent
    guard();lock=(folder/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads(path.read_text());plan.update(controller_pid=os.getpid(),status='validating_gpu')
    for name in ('logs','runs','preflights'):(folder/name).mkdir(exist_ok=True)
    save_json(path,plan)
    try:
        def preflight_lane(gpu):
            for i,method in enumerate(plan['methods']):
                if i%4==gpu:
                    run_preflight(plan,folder,method,TASKS[0],gpu,plan['cpu_lanes'][gpu])
        with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(preflight_lane,range(4)))
        save_json(folder/'ready.json',dict(preflights=plan['preflights'],checked_unix=time.time()))
        plan['status']='running';children={};last_summary=0
        while True:
            for job in plan['jobs']:
                if job['status'] not in ('initializing','running'):continue
                output=Path(job['output'])
                if (output/'completed.json').exists():
                    result=json.loads((output/'completed.json').read_text())
                    assert result['completed'] and result['updates']==995000 and result['runtime']['backend']=='gpu'
                    job.update(status='completed',finished_unix=result['finished_unix'])
                elif (output/'failure.json').exists() or not process_live(job):
                    job.update(status='failed',finished_unix=time.time(),error=(output/'failure.json').read_text() if (output/'failure.json').exists() else 'Exited without completion')
                elif (output/'progress.json').exists():
                    job['progress']=json.loads((output/'progress.json').read_text())
                    if job['progress'].get('updates',0)>0:job['status']='running'
            for pid,child in list(children.items()):
                if child.poll() is not None:child.wait();del children[pid]
            select_temperatures(plan,folder)
            for method in plan['methods']:
                for previous,task in zip(TASKS,TASKS[1:]):
                    waiting=[j for j in plan['jobs'] if j['method']==method and j['task']==task and j['status']=='waiting_for_previous_task']
                    if waiting and completed_task(plan,method,previous):
                        memory=gpu_memory();gpu=max(memory,key=lambda g:memory[g][1]-memory[g][0])
                        if memory[gpu][1]-memory[gpu][0] < plan['reserves_mib'][method]+plan['gpu_headroom_mib']:continue
                        active=[j for j in plan['jobs'] if j['status'] in ('running','initializing')]
                        lane=min(plan['cpu_lanes'],key=lambda c:sum(j.get('cpus')==c for j in active))
                        run_preflight(plan,folder,method,task,gpu,lane)
                        for job in waiting:job.update(status='queued',released_unix=time.time())
                        print('Released '+method+' '+task,flush=True)
            memory=gpu_memory();available=memory_available_mib()
            active=[j for j in plan['jobs'] if j['status'] in ('initializing','running')]
            for job in plan['jobs']:
                if job['status']!='queued':continue
                gpu=job['gpu'];used,total=memory[gpu]
                pending=sum(plan['reserves_mib'][x['method']] for x in active if x['gpu']==gpu and x['status']=='initializing')
                if used+pending+plan['reserves_mib'][job['method']]>total-plan['gpu_headroom_mib'] or available<plan['host_reserve_mib']:
                    job['waiting_reason']='memory headroom';continue
                if sum(x['status']=='initializing' and x['gpu']==gpu for x in active)>=plan['max_initializing_per_gpu']:
                    job['waiting_reason']='initialization admission';continue
                cpu=min(plan['cpu_lanes'],key=lambda c:sum(x.get('cpus')==c for x in active))
                cmd=['taskset','-c',','.join(map(str,cpu))]+command(job,plan['python'])
                with (folder/'logs'/(job['id']+'.log')).open('x') as stream:
                    child=subprocess.Popen(cmd,cwd=plan['source'],env=environment(plan['source'],gpu),
                        stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                fields=Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()
                job.update(status='initializing',pid=child.pid,start_ticks=int(fields[19]),started_unix=time.time(),cpus=cpu,command=cmd)
                job.pop('waiting_reason',None);children[child.pid]=child;active.append(job);available-=2048
                print('Started '+job['id']+' GPU'+str(gpu),flush=True)
            counts=dict(Counter(j['status'] for j in plan['jobs']))
            plan.update(counts=counts,checked_unix=time.time(),gpu_memory=memory)
            if time.time()-last_summary>=60:summarize(plan,folder);last_summary=time.time()
            save_json(path,plan)
            if not any(counts.get(s,0) for s in ('queued','running','initializing')):
                plan['status']='blocked_followups' if counts.get('waiting_for_previous_task') else 'completed_with_failures' if counts.get('failed') else 'completed'
                save_json(path,plan);summarize(plan,folder);break
            time.sleep(10)
    except BaseException as error:
        plan.update(status='controller_error',error=repr(error));save_json(path,plan);raise


if __name__=='__main__':main()
