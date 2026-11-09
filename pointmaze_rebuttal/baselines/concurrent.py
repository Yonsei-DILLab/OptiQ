"""Share GPUs across all methods, adopting live SAC processes without restart."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

from ..guards import assert_work_branch, assert_output_boundary
from .profiles import METHODS
from .queue import atomic_json, job_commands, run_logged_command

GIB = 1024**3


def process_info(pid):
    if not pid:
        return None
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return dict(state=fields[0], ppid=int(fields[1]), start_ticks=int(fields[19]),
                    rss=int(fields[21])*os.sysconf('SC_PAGE_SIZE'))
    except (FileNotFoundError, ProcessLookupError):
        return None


def gpu_assignment(method, seed):
    return (METHODS.index(method) + seed) % 4


def source_hashes(source):
    return {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((source/'pointmaze_rebuttal').rglob('*'))
        if p.is_file() and p.suffix in ('.py', '.json', '.yaml', '.xml')
        and '__pycache__' not in p.parts}


def validate_completed(folder, expected):
    c = json.loads((Path(folder)/'completed.json').read_text())
    if not c['completed'] or c['backend'] != 'gpu' or c['updates'] != expected:
        raise RuntimeError('Invalid completed result: ' + str(folder))
    return c


def wait_adopted(job, interval=5):
    while True:
        current = process_info(job['pid'])
        live = current and current['start_ticks'] == job['start_ticks'] and current['state'] != 'Z'
        failure = Path(job['output'])/'failure.json'
        if failure.exists():
            raise RuntimeError(failure.read_text())
        if not live:
            return validate_completed(job['output'], job['settings']['expected_updates'])
        time.sleep(interval)


def resources():
    query = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,memory.used,memory.total',
        '--format=csv,noheader,nounits'], text=True)
    gpus = {}
    for line in query.strip().splitlines():
        index, uuid, used, total = (s.strip() for s in line.split(','))
        gpus[int(index)] = dict(uuid=uuid, used=int(used)*1024**2, total=int(total)*1024**2)
    query = subprocess.check_output(['nvidia-smi',
        '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
        '--format=csv,noheader,nounits'], text=True)
    apps = {}
    for line in query.strip().splitlines():
        uuid, pid, used = (s.strip() for s in line.split(','))
        apps[int(pid)] = dict(uuid=uuid, used=int(used)*1024**2)
    limit = Path('/sys/fs/cgroup/memory.max').read_text().strip()
    maximum = int(limit) if limit != 'max' else os.sysconf('SC_PHYS_PAGES')*os.sysconf('SC_PAGE_SIZE')
    return dict(gpus=gpus, apps=apps, host_used=int(Path('/sys/fs/cgroup/memory.current').read_text()),
                host_max=maximum)


def admission(job, jobs, current, gpu_reserve=3*GIB, host_reserve=8*GIB):
    owned = {j.get('pid') for j in jobs if j.get('reserved')}
    gpu = current['gpus'][job['gpu']]
    if any(pid not in owned and a['uuid'] == gpu['uuid'] for pid, a in current['apps'].items()):
        return False, 'unrelated GPU process'
    reserved = [j for j in jobs if j.get('reserved')]
    gpu_future = sum(max(0, gpu_reserve-current['apps'].get(j.get('pid'), {}).get('used', 0))
                     for j in reserved if j['gpu'] == job['gpu'])
    own_gpu = 0 if job.get('reserved') else gpu_reserve
    if gpu['used']+gpu_future+own_gpu > gpu['total']-3*GIB:
        return False, 'GPU memory headroom'
    host_future = sum(max(0, host_reserve-(process_info(j.get('pid')) or {}).get('rss', 0))
                      for j in reserved)
    own_host = 0 if job.get('reserved') else host_reserve
    if current['host_used']+host_future+own_host > current['host_max']-32*GIB:
        return False, 'host memory headroom'
    return True, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    assert_work_branch()
    assert_output_boundary(args.plan)
    plan = json.loads(args.plan.read_text())
    output = args.plan.resolve().parent
    handle = (output/'controller.lock').open('a')
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock, launch_lock = threading.RLock(), threading.Lock()
    source = Path(plan['source'])
    if source_hashes(source) != plan['learner_hashes']:
        raise RuntimeError('Frozen learner source changed')
    def persist():
        with lock:
            atomic_json(args.plan, plan)
    def update(job, **values):
        with lock:
            job.update(values)
            persist()
    def retire_legacy():
        legacy = plan['legacy_controller']
        adopted = [j for j in plan['jobs'] if j.get('adopted')]
        if all(j['status'] in ('completed', 'failed') for j in adopted) and not legacy.get('retired_unix'):
            current = process_info(legacy['pid'])
            if current and current['start_ticks'] == legacy['start_ticks']:
                if current['state'] not in ('T', 't', 'Z'):
                    raise RuntimeError('Legacy controller unexpectedly resumed')
                os.kill(legacy['pid'], signal.SIGKILL)
            legacy['retired_unix'] = time.time()
            persist()

    def execute(job):
        try:
            if job.get('adopted'):
                wait_adopted(job)
            else:
                common_env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(job['gpu']),
                    PYTHONPATH=str(source), XLA_PYTHON_CLIENT_PREALLOCATE='false',
                    MUJOCO_GL='egl', WANDB_MODE='offline', OMP_NUM_THREADS='1',
                    OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
                preflight, train = job_commands(job, plan['python'], plan['profile'])
                for phase, command, destination, expected in (
                    ('preflight', preflight, job['preflight'], job['settings']['updates_per_collect']),
                    ('training', train, job['output'], job['settings']['expected_updates'])):
                    while True:
                        with launch_lock:
                            if plan.get('abort_pending'):
                                update(job, status='blocked_by_failure')
                                return
                            allowed, reason = admission(job, plan['jobs'], resources())
                            if allowed:
                                assert_work_branch()
                                if source_hashes(source) != plan['learner_hashes']:
                                    raise RuntimeError('Frozen learner source changed')
                                actual = ['taskset', '-c', ','.join(map(str, job['cpu_affinity']))] + command
                                log = output/'logs'/f'{job["name"]}-{phase}.log'
                                log.parent.mkdir(parents=True, exist_ok=True)
                                stream = log.open('x')
                                child = subprocess.Popen(actual, cwd=source, env=common_env,
                                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                                info = process_info(child.pid)
                                update(job, status=phase, phase=phase, pid=child.pid,
                                    start_ticks=info['start_ticks'] if info else None,
                                    reserved=True, wait_reason=None, command=actual,
                                    phase_started_unix=time.time(), log=str(log))
                                break
                            update(job, status='waiting_for_resources', wait_reason=reason)
                        time.sleep(5)
                    try:
                        code = child.wait()
                    finally:
                        stream.close()
                    if code != 0:
                        raise RuntimeError(f'{phase} exited {code}: {log}')
                    validate_completed(destination, expected)
            update(job, status='completed', finished_unix=time.time(), reserved=False)
        except BaseException as error:
            with lock:
                job.update(status='failed', error=repr(error), reserved=False, finished_unix=time.time())
                plan['abort_pending'] = True
                persist()
        finally:
            with lock:
                retire_legacy()

    plan.update(status='running_concurrently', controller_pid=os.getpid(), started_unix=time.time())
    persist()
    with ThreadPoolExecutor(max_workers=len(plan['jobs'])) as executor:
        list(executor.map(execute, plan['jobs']))
    plan.update(status='completed' if all(j['status']=='completed' for j in plan['jobs'])
                else 'blocked_by_failure', finished_unix=time.time())
    persist()


if __name__ == '__main__':
    main()
