"""Adopt the original campaign and advance queued Hammer/Pen seeds unchanged.

Run as a scheduler sidecar with the frozen learner source on PYTHONPATH.
The authoritative 40-run plan and completion markers remain in their original
locations, so monitoring and the baseline dependency gate keep working.
"""
import argparse
from collections import Counter, deque
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from adroit_rebuttal.run import atomic, guard


def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return int(fields[19]) if fields[0] != 'Z' else None
    except OSError:
        return None


def pin_tree(pid, cpus):
    """Move existing threads and collectors without restarting a learner."""
    try:
        children = list(map(int, Path(f'/proc/{pid}/task/{pid}/children').read_text().split()))
        for _ in range(2):
            for thread in Path(f'/proc/{pid}/task').iterdir():
                try:
                    os.sched_setaffinity(int(thread.name), cpus)
                except ProcessLookupError:
                    pass
        for child in children:
            pin_tree(child, cpus)
    except FileNotFoundError:
        pass


def priority(job):
    return job['task'] in ('hammer', 'pen') and job['seed'] in range(5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    guard()
    path, folder = args.plan, args.plan.parent
    controller_lock = (folder/'controller.lock').open('a')
    fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads(path.read_text())
    assert len(plan['jobs']) == 40 and not plan['smoke']
    verification = json.loads(Path(plan['verification']).read_text())
    assert verification['status'] == 'completed' and len(verification['jobs']) == 8
    assert all(j['status'] == 'completed' for j in verification['jobs'])
    assert json.loads(Path('/workspace/heechan-adroit/datasets/verified-demonstrations.json').read_text())['passed']
    assert json.loads(Path('/workspace/heechan-adroit/environment-checks.json').read_text())['passed']
    groups = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        base = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key = (base.joinpath('physical_package_id').read_text(), base.joinpath('core_id').read_text())
        groups.setdefault(key, []).append(cpu)
    lanes = [sum(list(groups.values())[i:i+2], []) for i in range(4, len(groups)-1, 2)]
    assert len(lanes) >= 24, 'Need 48 physical cores plus service headroom'
    from adroit_rebuttal.verified_demos import resolve
    for task, spec in plan['verified_demonstrations'].items():
        resolve(task, spec)
    children, locks, history = {}, {}, {}
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for j in plan['jobs']:
        if j['status'] == 'running':
            j['start_ticks'] = j.get('start_ticks') or identity(j['pid'])
            if identity(j['pid']) == j['start_ticks'] and j['start_ticks'] is not None:
                j['scheduler_cpus'] = lanes.pop(0)
                pin_tree(j['pid'], j['scheduler_cpus'])
                print('Adopted '+j['id']+' PID '+str(j['pid']), flush=True)
    plan['parallel_policy'] = dict(priority_tasks=['hammer', 'pen'], priority_seeds=[0, 1, 2, 3, 4],
        max_active=24, max_per_gpu=6, max_initializing_per_gpu=2,
        max_other_active=4, learner_source_unchanged=True, controller_pid=os.getpid())
    try:
        while not stopping:
            for j in plan['jobs']:
                if j['status'] != 'running':
                    continue
                marker = Path(j['output'])/'completed.json'
                failure = Path(j['output'])/'failure.json'
                if marker.exists():
                    done = json.loads(marker.read_text())
                    assert done['completed'] and not done['smoke']
                    assert (done['method'], done['task'], done['seed']) == (j['method'], j['task'], j['seed'])
                    assert done['steps'] == 5000192 and done['updates'] == 623424
                    assert done['num_envs'] == 256 and done['observed_utd'] == .125
                    j.update(status='completed', result=done)
                elif failure.exists() or identity(j['pid']) != j['start_ticks']:
                    j.update(status='failed', error=failure.read_text() if failure.exists() else 'Exited without completion marker')
                else:
                    continue
                if j.get('scheduler_cpus'):
                    lanes.append(j.pop('scheduler_cpus'))
            for pid, child in list(children.items()):
                if child.poll() is not None:
                    child.wait()
                    del children[pid]
            active = [j for j in plan['jobs'] if j['status'] == 'running']
            raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                '--format=csv,noheader,nounits'], text=True, timeout=15)
            gpu_info = {}
            for line in raw.splitlines():
                gpu, uuid, memory, util = [v.strip() for v in line.split(',')]
                gpu = int(gpu)
                history.setdefault(gpu, deque(maxlen=3)).append(int(util))
                gpu_info[gpu] = dict(uuid=uuid, memory=int(memory), util=sum(history[gpu])/len(history[gpu]))
                if gpu not in locks:
                    lock = (Path('/workspace/.gmm40-gpu-queue-locks')/(uuid+'.lock')).open('a')
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        lock.close()
                    else:
                        locks[gpu] = lock
            candidates = sorted((j for j in plan['jobs'] if j['status'] == 'queued'),
                key=lambda j: (not priority(j), j['seed'], j['task'], j['method']))
            for j in candidates:
                if len(active) >= 24 or not lanes:
                    break
                if not priority(j) and len(active) >= 4:
                    continue
                counts = Counter(x['gpu'] for x in active)
                initializing = Counter()
                for x in active:
                    p = Path(x['output'])/'progress.json'
                    progress = json.loads(p.read_text()) if p.exists() else {}
                    if progress.get('updates', 0) == 0:
                        initializing[x['gpu']] += 1
                eligible = [g for g, info in gpu_info.items() if g in locks and counts[g] < 6
                    and initializing[g] < 2 and info['memory'] < 24000 and info['util'] < 85]
                if not eligible:
                    break
                gpu = min(eligible, key=lambda g: (counts[g], gpu_info[g]['util'], gpu_info[g]['memory'], g))
                out = Path(j['output'])
                assert not out.exists(), 'Refusing to overwrite '+str(out)
                cpus = lanes.pop(0)
                spec = folder/'specs'/(j['id']+'.json')
                atomic(spec, j)
                command = ['taskset', '-c', ','.join(map(str, cpus)), plan['python'], '-u', '-m',
                           'adroit_rebuttal.run', '--spec', str(spec)]
                learner_source = j.get('source', plan['source'])
                env = dict(os.environ, PYTHONPATH=learner_source, CUDA_VISIBLE_DEVICES=str(gpu), XLA_PYTHON_CLIENT_PREALLOCATE='false',
                    MUJOCO_GL='egl', WANDB_MODE='disabled', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
                with (folder/'logs'/(j['id']+'.log')).open('x') as stream:
                    child = subprocess.Popen(command, cwd=learner_source, env=env, stdout=stream,
                                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                j.update(status='running', pid=child.pid, start_ticks=identity(child.pid), gpu=gpu,
                         started_unix=time.time(), scheduler_cpus=cpus)
                children[child.pid] = child
                active.append(j)
                gpu_info[gpu]['memory'] += 2048
                print('Started '+j['id']+' GPU'+str(gpu), flush=True)
            for gpu in list(locks):
                if not any(j['gpu'] == gpu for j in active):
                    locks.pop(gpu).close()
            counts = dict(Counter(j['status'] for j in plan['jobs']))
            finished = all(j['status'] in ('completed', 'failed') for j in plan['jobs'])
            plan.update(status=('completed_with_failures' if counts.get('failed') else 'completed') if finished else 'running',
                        counts=counts, observed_unix=time.time())
            atomic(path, plan)
            if finished:
                return
            time.sleep(5)
    except BaseException as error:
        plan.update(status='controller_error', error=repr(error))
        atomic(path, plan)
        raise
    finally:
        for lock in locks.values():
            lock.close()


if __name__ == '__main__':
    main()
