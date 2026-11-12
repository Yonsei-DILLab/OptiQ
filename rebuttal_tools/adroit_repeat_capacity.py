"""Finite, resource-admitted repeat queue. Never discard trials or modify learners."""
import argparse
from collections import Counter, deque
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from adroit_gpu_repeat import atomic, alive, worker


def free_cores():
    topology = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        p = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key = (p.joinpath('physical_package_id').read_text(), p.joinpath('core_id').read_text())
        topology.setdefault(key, []).append(cpu)
    reserved = {c for g in list(topology.values())[:4] for c in g}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or not alive(int(p.name)):
            continue
        try:
            cpus = os.sched_getaffinity(int(p.name))
        except (ProcessLookupError, PermissionError):
            continue
        if len(cpus) <= 16:
            reserved.update(cpus)
    return [g for g in topology.values() if not set(g) & reserved]


def campaign(path):
    root = path.parent
    lock = (root/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads(path.read_text())
    children = {}
    history = {}
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while not stopping:
        for j in plan['jobs']:
            if j['status'] != 'running':
                continue
            child = children.get(j['pid'])
            if (child and child.poll() is None) or (not child and alive(j['pid'])):
                continue
            if child:
                child.wait()
                del children[j['pid']]
            marker = Path(j['output'])/'completed.json'
            try:
                result = json.loads(marker.read_text())
                proof = json.loads((Path(j['output'])/'demonstration-validation.json').read_text())
                final = json.loads((Path(j['output'])/'evaluations.jsonl').read_text().splitlines()[-1])
                assert result['completed'] and result['steps'] == final['steps'] == 5000192
                assert result['updates'] == 623424
                assert proof['passed'] and proof['manifest_sha256'] == j['demonstration_manifest_sha256']
                assert len(final['zero_z']['episodes']) == 100
                j.update(status='completed', final_zero_z=final['zero_z']['success_rate'],
                         final_stochastic_z=final['stochastic_z']['success_rate'],
                         final_full_policy=final['stochastic_z_sigma']['success_rate'])
            except Exception as exc:
                j.update(status='failed', error=repr(exc))
            j['finished_unix'] = time.time()
        raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
        gpus = {}
        for line in raw.splitlines():
            i, uuid, used, total, util = [s.strip() for s in line.split(',')]
            i = int(i)
            gpus[i] = dict(uuid=uuid, used=int(used), total=int(total), util=int(util))
            history.setdefault(i, deque(maxlen=7)).append((time.time(), int(util)))
        active = [j for j in plan['jobs'] if j['status'] == 'running']
        free = free_cores()
        available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
        reasons = []
        if not (root/'PAUSE').exists() and shutil.disk_usage(root).free > 20*1024**3 and available > 16*1024**3:
            for gpu, info in gpus.items():
                h = history[gpu]
                avg = sum(u for _, u in h)/len(h)
                if len(h) < 7 or h[-1][0]-h[0][0] < 55 or avg >= 75:
                    reasons.append(f'GPU{gpu}: utilization/window {avg:.1f}%/{len(h)}')
                    continue
                if sum(j['gpu'] == gpu for j in active) >= 4 or info['total']-info['used'] < 6144 or len(free) < 2:
                    reasons.append(f'GPU{gpu}: memory/CPU/concurrency admission')
                    continue
                j = next((j for j in plan['jobs'] if j['status'] == 'queued'), None)
                if not j:
                    break
                cpus = free.pop(0)+free.pop(0)
                j.update(gpu=gpu, gpu_uuid=info['uuid'], cpu_affinity=cpus)
                assert not Path(j['output']).exists(), 'Never overwrite an earlier trial'
                spec = root/'specs'/(j['id']+'.json')
                atomic(spec, j)
                env = dict(os.environ, PYTHONPATH=plan['source'], CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1',
                           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', XLA_PYTHON_CLIENT_PREALLOCATE='false',
                           MUJOCO_GL='egl', WANDB_MODE='disabled', JAX_COMPILATION_CACHE_DIR=str(root/'jax-cache'),
                           MPLCONFIGDIR=str(root/'mpl-cache'))
                cmd = ['taskset', '-c', ','.join(map(str, cpus)), plan['python'], '-u', str(Path(__file__).resolve()), '--worker', '--spec', str(spec)]
                with (root/'logs'/(j['id']+'.log')).open('x') as log:
                    child = subprocess.Popen(cmd, cwd=plan['source'], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                j.update(status='running', pid=child.pid, started_unix=time.time())
                children[child.pid] = child
                active.append(j)
                history[gpu].clear()  # Wait for a fresh minute before admitting another learner.
                print('Started', j['id'], 'GPU', gpu, flush=True)
        plan.update(counts=dict(Counter(j['status'] for j in plan['jobs'])), observed_unix=time.time(), admission=reasons)
        atomic(path, plan)
        atomic(root/'all-trial-results.json', [dict(j) for j in plan['jobs']])
        candidates = {}
        for seed in (0, 2, 3, 4):
            done = [j for j in plan['jobs'] if j['seed'] == seed and j['status'] == 'completed']
            if done:
                best = max(done, key=lambda j: j['final_zero_z'])
                candidates[str(seed)] = {k: best[k] for k in ('id', 'seed', 'gpu', 'final_zero_z', 'final_stochastic_z')}
        atomic(root/'selected-candidates.json', dict(selection='Exploratory best-of-repeats; all attempts retained; not an unbiased seed estimate', candidates=candidates))
        with (root/'gpu-load.jsonl').open('a') as log:
            log.write(json.dumps(dict(observed_unix=time.time(), gpus=gpus, active=[{k:j[k] for k in ('id', 'gpu', 'pid')} for j in active]))+'\n')
        if all(j['status'] in ('completed', 'failed') for j in plan['jobs']):
            return
        time.sleep(10)
    # Graceful controller shutdown does not terminate unrelated jobs or its learners.
    atomic(root/'controller-stopped.json', dict(observed_unix=time.time(), active=[j['id'] for j in active]))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path)
    p.add_argument('--worker', action='store_true')
    p.add_argument('--spec', type=Path)
    a = p.parse_args()
    worker(a.spec) if a.worker else campaign(a.plan)
