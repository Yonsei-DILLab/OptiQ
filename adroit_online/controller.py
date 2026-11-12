"""GPU-aware, non-retrying execution queue; run under Supervisor."""
import argparse
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .config import campaign, run_id
from .run import ROOT, atomic, source_provenance


def assigned(c, shard):
    # Each host sees all tasks and both algorithms; cross over the two seeds.
    return ((c['seed'] - 4) + int(c['method'] == 'td3')) % 2 == shard


def gpu_state():
    result = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
                                     '--format=csv,noheader,nounits'], text=True)
    rows = {}
    for row in csv.reader(result.splitlines()):
        index, uuid, free = [x.strip() for x in row]
        rows[int(index)] = dict(uuid=uuid, free_mib=int(free), pids=[])
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                   '--format=csv,noheader,nounits'], text=True)
    by_uuid = {v['uuid']: v for v in rows.values()}
    for row in csv.reader(apps.splitlines()):
        if len(row) == 2 and row[0].strip() in by_uuid:
            by_uuid[row[0].strip()]['pids'].append(int(row[1].strip()))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--shard', type=int, choices=(0, 1), required=True)
    p.add_argument('--slots-per-gpu', type=int, default=2)
    a = p.parse_args()
    proof = source_provenance()
    receipt = json.loads((a.root / 'preflight' / 'passed.json').read_text())
    assert receipt['source_commit'] == proof['commit'] and receipt['passed']
    assert sorted(receipt['gpus']) == [0, 1, 2, 3]
    a.root.mkdir(parents=True, exist_ok=True)
    (a.root / 'process-logs').mkdir(exist_ok=True)
    jobs = [c for c in campaign() if assigned(c, a.shard)]
    atomic(a.root / 'plan.json', dict(source_commit=proof['commit'], shard=a.shard,
                                    slots_per_gpu=a.slots_per_gpu, jobs=jobs))
    state_file = a.root / 'queue-state.json'
    if state_file.exists():
        states = json.loads(state_file.read_text())['jobs']
    else:
        states = {run_id(c): dict(status='pending') for c in jobs}
    # Restarts never relaunch an interrupted or failed run, nor overwrite outputs.
    for c in jobs:
        name = run_id(c)
        dest = a.root / 'runs' / name
        if (dest / 'completed.json').exists():
            states[name]['status'] = 'completed'
        elif dest.exists() or states[name]['status'] == 'running':
            states[name]['status'] = 'requires_review'
    active = {}
    cpus = sorted(os.sched_getaffinity(0))
    slots = [(g, k) for k in range(a.slots_per_gpu) for g in range(4)]
    assert len(cpus) >= len(slots) * 8
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    while True:
        for name, (proc, slot, log) in list(active.items()):
            code = proc.poll()
            if code is not None:
                log.close()
                complete = (a.root / 'runs' / name / 'completed.json').exists()
                states[name].update(status='completed' if code == 0 and complete else 'failed',
                                    returncode=code, finished_unix=time.time())
                del active[name]
        pending = [c for c in jobs if states[run_id(c)]['status'] == 'pending']
        if pending and not stop[0]:
            gpus = gpu_state()
            own_pids = {v[0].pid for v in active.values()}
            occupied = {v[1] for v in active.values()}
            for index, slot in enumerate(slots):
                gpu, _ = slot
                if slot in occupied or not pending:
                    continue
                info = gpus[gpu]
                if info['free_mib'] < 6000 or set(info['pids']) - own_pids:
                    continue
                c = pending.pop(0)
                name = run_id(c)
                allocation = cpus[index * 8:(index + 1) * 8]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), JAX_PLATFORMS='cuda',
                           XLA_PYTHON_CLIENT_PREALLOCATE='false', OMP_NUM_THREADS='1',
                           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                           MUJOCO_GL='egl', PYTHONUNBUFFERED='1')
                cmd = ['taskset', '-c', ','.join(map(str, allocation)), sys.executable,
                       '-m', 'adroit_online.run', '--method', c['method'], '--task', c['task'],
                       '--seed', str(c['seed']), '--profile', c['profile'],
                       '--output', str(a.root / 'runs' / name)]
                log = (a.root / 'process-logs' / (name + '.log')).open('a')
                proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                active[name] = (proc, slot, log)
                states[name] = dict(status='running', pid=proc.pid, gpu=gpu, cpus=allocation,
                                    started_unix=time.time(), source_commit=proof['commit'])
                atomic(state_file, dict(source_commit=proof['commit'], jobs=states, updated_unix=time.time()))
                print('started', name, 'gpu', gpu, 'pid', proc.pid, flush=True)
                # Stagger imports/compilation; do not occupy the same GPU with simultaneous compilers.
                break
        atomic(state_file, dict(source_commit=proof['commit'], jobs=states, updated_unix=time.time()))
        if not active and (not pending or stop[0]):
            break
        time.sleep(15)


if __name__ == '__main__':
    main()
