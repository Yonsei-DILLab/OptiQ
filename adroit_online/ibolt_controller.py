"""Independent fixed-temperature iBOLT jobs, confined to dilg-01 GPUs0/1."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from .config import IBOLT_TEMPERATURES, TASKS, configuration, run_id
from .ibolt_allocation import peer_state
from .controller import gpu_state
from .run import ROOT, atomic, source_provenance


def campaign(temperatures=(.25,)):
    # Four tasks and both seeds represented in the first wave; no completion gates.
    assert len(set(temperatures)) == len(temperatures) and temperatures
    return [configuration('ibolt', task, 4 + ((i + wave) % 2), 'e256-u025',
                          temperature=temperatures[(i + offset) % len(temperatures)])
            for wave in (0, 1) for offset in range(len(temperatures))
            for i, task in enumerate(TASKS)]


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--temperatures', type=float, nargs='+', choices=IBOLT_TEMPERATURES, default=[.25])
    p.add_argument('--peer-root', type=Path)
    a = p.parse_args(); proof = source_provenance()
    assert socket.gethostname().split('.')[0] in ('dilg01', 'dilg-01')
    a.root.mkdir(parents=True, exist_ok=True)
    lock = (a.root / 'controller.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    receipt = json.loads((a.root / 'preflight' / 'passed.json').read_text())
    assert receipt['passed'] and receipt['source_commit'] == proof['commit'] and receipt['gpus'] == [0, 1]
    assert sorted(receipt['temperatures']) == sorted(a.temperatures)
    jobs = campaign(a.temperatures); hardware = gpu_state()
    cpus = sorted(os.sched_getaffinity(0)); assert len(cpus) >= 32
    slots = [(g, k) for k in range(2) for g in (0, 1)]
    atomic(a.root / 'plan.json', dict(source_commit=proof['commit'], hostname=socket.gethostname(),
        allowed_gpus=[0, 1], gpu_uuids={str(g): hardware[g]['uuid'] for g in (0, 1)},
        slots_per_gpu=2, cpus_per_job=8, temperatures=a.temperatures,
        peer_root=str(a.peer_root) if a.peer_root else None, jobs=jobs))
    (a.root / 'process-logs').mkdir(exist_ok=True)
    path = a.root / 'queue-state.json'
    states = json.loads(path.read_text())['jobs'] if path.exists() else {
        run_id(c): dict(status='pending') for c in jobs}
    for c in jobs:
        name = run_id(c); dest = a.root / 'runs' / name
        if (dest / 'completed.json').exists(): states[name]['status'] = 'completed'
        elif dest.exists() or states[name]['status'] == 'running': states[name]['status'] = 'requires_review'
    active = {}; stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    while True:
        for name, (proc, slot, log) in list(active.items()):
            code = proc.poll()
            if code is None: continue
            log.close()
            completed = (a.root / 'runs' / name / 'completed.json').exists()
            states[name].update(status='completed' if code == 0 and completed else 'failed',
                                returncode=code, finished_unix=time.time())
            del active[name]
        pending = [c for c in jobs if states[run_id(c)]['status'] == 'pending']
        if pending and not stop[0]:
            hardware = gpu_state(); occupied = {x[1] for x in active.values()}
            ours = {x[0].pid for x in active.values()}
            peers = peer_state(a.peer_root)
            for i, slot in enumerate(slots):
                gpu, _ = slot
                if slot in occupied or hardware[gpu]['free_mib'] < 12000: continue
                if peers['pending'] or set(hardware[gpu]['pids']) - ours - peers['pids'][gpu]: continue
                if len(hardware[gpu]['pids']) >= 4: continue
                c = pending[0]; name = run_id(c); allocation = cpus[i * 8:(i + 1) * 8]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), JAX_PLATFORMS='cuda',
                           XLA_PYTHON_CLIENT_PREALLOCATE='false', OMP_NUM_THREADS='1',
                           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                           MUJOCO_GL='egl', PYTHONUNBUFFERED='1')
                log = (a.root / 'process-logs' / (name + '.log')).open('a')
                cmd = ['taskset', '-c', ','.join(map(str, allocation)), sys.executable,
                       '-m', 'adroit_online.run', '--method', 'ibolt', '--task', c['task'],
                       '--seed', str(c['seed']), '--profile', c['profile'],
                       '--temperature', str(c['temperature']),
                       '--output', str(a.root / 'runs' / name)]
                proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                active[name] = (proc, slot, log)
                states[name] = dict(status='running', pid=proc.pid, gpu=gpu,
                    gpu_uuid=hardware[gpu]['uuid'], cpus=allocation, started_unix=time.time(),
                    source_commit=proof['commit'])
                print('started', name, 'gpu', gpu, 'pid', proc.pid, flush=True)
                break
        atomic(path, dict(source_commit=proof['commit'], jobs=states, updated_unix=time.time()))
        if not active and (not pending or stop[0]): break
        time.sleep(15)


if __name__ == '__main__':
    main()
