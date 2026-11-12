"""Private Door data-only comparison; preserve other campaigns and learner code."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from .run import atomic, guard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    guard()
    plan = json.loads(args.plan.read_text())
    folder = args.plan.parent
    lock = (folder/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    children, gpu_locks = {}, {}
    admission = Path('/workspace/.ibolt-capacity-admission.lock').open('a')
    for dataset in plan['datasets'].values():
        assert hashlib.sha256(Path(dataset['path']).read_bytes()).hexdigest() == dataset['sha256']

    def terminate(_sig, _frame):
        for child in children.values():
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        plan['status'] = 'stopped'
        atomic(args.plan, plan)
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)

    def cpu_lanes():
        reserved = set()
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():
                continue
            try:
                affinity = os.sched_getaffinity(int(p.name))
            except (ProcessLookupError, PermissionError):
                continue
            if len(affinity) <= 16:
                reserved.update(affinity)
        # Reserve CPU pools allocated to other not-yet-started workers.
        for path in plan.get('cpu_reserve_plans', []):
            other = json.loads(Path(path).read_text())
            reserved.update(c for lane in other.get('cpu_lanes', []) for c in lane)
        topology = {}
        for c in sorted(os.sched_getaffinity(0)):
            p = Path(f'/sys/devices/system/cpu/cpu{c}/topology')
            key = (p.joinpath('physical_package_id').read_text(), p.joinpath('core_id').read_text())
            topology.setdefault(key, []).append(c)
        return [cs for cs in topology.values() if not set(cs) & reserved]

    try:
        assert not any(j['status'] == 'running' for j in plan['jobs']), 'No automatic replay resume'
        while True:
            for pid, child in list(children.items()):
                if child.poll() is None:
                    continue
                job = next(j for j in plan['jobs'] if j.get('pid') == pid)
                marker = Path(job['output'])/'completed.json'
                if child.returncode == 0 and marker.exists():
                    done = json.loads(marker.read_text())
                    assert done['completed'] and done['smoke'] == (job['phase'] == 'smoke')
                    assert done['steps'] == (13568 if done['smoke'] else 5000192)
                    assert done['updates'] == (96 if done['smoke'] else 623424)
                    job.update(status='completed', result=done)
                else:
                    failure = Path(job['output'])/'failure.json'
                    job.update(status='failed', error=failure.read_text() if failure.exists() else f'exit {child.returncode}')
                gpu_locks.pop(pid).close()
                child.wait()
                del children[pid]
            if any(j['status'] == 'failed' for j in plan['jobs']):
                plan['status'] = 'held_for_failure_review'
                atomic(args.plan, plan)
                # Keep watching other already-running runs; no new admission.
                if children:
                    time.sleep(10)
                    continue
                return
            for job in plan['jobs']:
                progress = Path(job['output'])/'progress.json'
                if job['status'] == 'running' and progress.exists():
                    job['progress'] = json.loads(progress.read_text())
            smokes = [j for j in plan['jobs'] if j['phase'] == 'smoke']
            smoke_passed = all(j['status'] == 'completed' for j in smokes)
            plan['gpu_smokes_passed'] = smoke_passed
            collection = [j for j in plan['jobs'] if j['phase'] == ('full' if smoke_passed else 'smoke')]
            fcntl.flock(admission, fcntl.LOCK_EX)
            try:
                raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu',
                    '--format=csv,noheader,nounits'], text=True)
                lanes = cpu_lanes()
                for row in raw.splitlines():
                    gpu, uuid, used, free, util = (s.strip() for s in row.split(','))
                    # Four short smoke runs share only one GPU at a time. Full
                    # experiments wait for an idle GPU and never stop another job.
                    if len(children) >= (2 if smoke_passed else 1) or len(lanes) < 6:
                        break
                    if int(free) < 12288 or int(util) > (5 if smoke_passed else 97) or (smoke_passed and int(used) > 512):
                        continue
                    job = next((j for j in collection if j['status'] == 'queued'), None)
                    if job is None:
                        break
                    path = Path('/workspace/.gmm40-gpu-queue-locks')/(uuid+'.lock')
                    path.parent.mkdir(exist_ok=True)
                    glock = path.open('a')
                    try:
                        fcntl.flock(glock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        glock.close()
                        continue
                    assert not Path(job['output']).exists()
                    dataset = plan['datasets'][job['dataset']]
                    job.update(demo_path=dataset['path'], demo_sha256=dataset['sha256'],
                               worker_cpus=lanes[2:6], learner_cpus=lanes[0]+lanes[1])
                    spec = folder/'specs'/(job['id']+'.json')
                    atomic(spec, job)
                    command = ['nice', '-n', '10', 'taskset', '-c', ','.join(map(str, job['learner_cpus'])),
                               plan['python'], '-u', '-m', 'adroit_rebuttal.run', '--spec', str(spec)]
                    if job['phase'] == 'smoke':
                        command.append('--smoke')
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONPATH=plan['source']+os.pathsep+plan['robotics_overlay'],
                               MUJOCO_GL='egl', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                               XLA_PYTHON_CLIENT_PREALLOCATE='false', WANDB_MODE='disabled',
                               MPLCONFIGDIR=str(folder/'mpl-cache'), JAX_COMPILATION_CACHE_DIR=str(folder/'jax-cache'))
                    with (folder/'logs'/(job['id']+'.log')).open('x') as stream:
                        child = subprocess.Popen(command, cwd=plan['source'], env=env, start_new_session=True,
                                                 stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                    children[child.pid], gpu_locks[child.pid] = child, glock
                    job.update(status='running', pid=child.pid, gpu=int(gpu), started_unix=time.time())
                    lanes = lanes[6:]
                    print('Started '+job['id']+' on GPU'+gpu, flush=True)
            finally:
                fcntl.flock(admission, fcntl.LOCK_UN)
            plan.update(status='running' if children else 'waiting_for_idle_gpu',
                        counts=dict(Counter(j['status'] for j in plan['jobs'])), observed_unix=time.time())
            if all(j['status'] == 'completed' for j in plan['jobs']):
                plan['status'] = 'completed'
            atomic(args.plan, plan)
            if plan['status'] == 'completed':
                return
            time.sleep(10)
    except BaseException as error:
        if not isinstance(error, SystemExit):
            for child in children.values():
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
            plan.update(status='controller_error', error=repr(error))
            atomic(args.plan, plan)
        raise


if __name__ == '__main__':
    main()
