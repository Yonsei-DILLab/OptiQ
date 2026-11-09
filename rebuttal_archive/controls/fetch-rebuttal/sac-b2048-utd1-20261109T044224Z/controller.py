"""SAC-only batch-size comparison, admitted to spare GPU capacity."""
from collections import deque
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

folder = Path(__file__).resolve().parent
path = folder / 'plan.json'
plan = json.loads(path.read_text())
sys.path.insert(0, plan['source'])
from fetch_rebuttal.run import guard, save_json
from fetch_rebuttal.campaign import environment, process_live, memory_available_mib


def sample_gpus():
    raw = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,utilization.gpu,memory.used,memory.total',
        '--format=csv,noheader,nounits'], text=True, timeout=10)
    return {int(g): dict(util=int(u), used=int(m), total=int(t))
            for g, u, m, t in (line.split(',') for line in raw.strip().splitlines())}


def refresh(job):
    if job['status'] not in ('initializing', 'running'):
        return
    output = Path(job['output'])
    if (output/'completed.json').exists():
        result = json.loads((output/'completed.json').read_text())
        expected = 4 if job.get('smoke') else 995000
        if result['updates'] != expected or result['runtime']['backend'] != 'gpu':
            raise RuntimeError('Invalid completion: ' + job['id'])
        job.update(status='completed', result=result)
    elif (output/'failure.json').exists() or not process_live(job):
        failure = output/'failure.json'
        job.update(status='failed', error=json.loads(failure.read_text()) if failure.exists()
                   else 'Child exited without completion', finished_unix=time.time())
    elif (output/'progress.json').exists():
        job['progress'] = json.loads((output/'progress.json').read_text())
        if job['progress'].get('updates', 0) > 0:
            job['status'] = 'running'


def launch(job, gpu, cpus):
    cmd = ['taskset', '-c', ','.join(map(str, cpus)), plan['python'], '-u', '-m',
           'fetch_rebuttal.run', '--method', 'sac', '--task', 'pick-and-place',
           '--seed', str(job['seed']), '--batch-size', '2048', '--output', job['output']]
    if job.get('smoke'):
        cmd.append('--smoke')
    with (folder/'logs'/f'{job["id"]}.log').open('x') as stream:
        child = subprocess.Popen(cmd, cwd=plan['source'], env=environment(plan['source'], gpu),
                                 stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
    fields = Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()
    job.update(status='initializing', gpu=gpu, cpus=cpus, pid=child.pid,
               start_ticks=int(fields[19]), command=cmd, started_unix=time.time())
    children[child.pid] = child
    print(json.dumps(dict(started=job['id'], gpu=gpu, cpus=cpus)), flush=True)
    save_json(path, plan)


guard()
lock = (folder/'controller.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
history = {g: deque(maxlen=6) for g in plan['allowed_gpus']}
children = {}
plan.update(status='waiting_for_capacity', controller_pid=os.getpid())
try:
    while True:
        for job in [plan['preflight'], *plan['jobs']]:
            refresh(job)
        for pid, child in list(children.items()):
            if child.poll() is not None:
                child.wait()
                del children[pid]
        stats = sample_gpus()
        for g in history:
            history[g].append(stats[g]['util'])
            stats[g]['mean_util'] = sum(history[g])/len(history[g])
        plan['gpu_samples'] = stats
        active = [j for j in plan['jobs'] if j['status'] in ('initializing', 'running')]
        candidates = [g for g in history if len(history[g]) >= 3
                      and stats[g]['mean_util'] <= plan['max_admission_util']
                      and stats[g]['total']-stats[g]['used'] >= 4096
                      and not any(j.get('gpu') == g for j in active)]
        candidates.sort(key=lambda g: stats[g]['mean_util'])
        preflight = plan['preflight']
        if preflight['status'] == 'failed':
            plan['status'] = 'preflight_failed'
            save_json(path, plan)
            raise RuntimeError('SAC batch-2048 GPU preflight failed')
        elif preflight['status'] == 'queued':
            if candidates and memory_available_mib() >= 8192:
                launch(preflight, candidates[0], plan['cpu_lanes'][0])
                plan['status'] = 'validating_gpu'
        elif preflight['status'] == 'completed':
            plan['status'] = 'running'
            pending = [j for j in plan['jobs'] if j['status'] == 'queued']
            # One new start at a time. Let allocations and utilization settle.
            if pending and candidates and not any(j['status'] == 'initializing' for j in active):
                if memory_available_mib() >= 8192:
                    job = pending[0]
                    launch(job, candidates[0], plan['cpu_lanes'][job['seed']])
        counts = {s: sum(j['status'] == s for j in plan['jobs'])
                  for s in ('queued', 'initializing', 'running', 'completed', 'failed')}
        plan.update(counts=counts, checked_unix=time.time())
        if counts['completed']+counts['failed'] == len(plan['jobs']):
            plan['status'] = 'completed_with_failures' if counts['failed'] else 'completed'
            save_json(path, plan)
            break
        save_json(path, plan)
        time.sleep(5)
except BaseException as error:
    plan.update(status='controller_error', error=repr(error), checked_unix=time.time())
    save_json(path, plan)
    raise
