"""Separate, peer-aware queue for the authorized eight T0.01 online runs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .config import configuration, run_id
from .controller import gpu_state
from .priority_controller import alive, choose_slot, process_ticks, verify_adopted, verify_source
from .run import ROOT, atomic, source_provenance

HOSTNAMES = {0: '98073c6dde13', 1: 'ff5c11e94b3e'}
ALLOWED_GPUS = {0: (0, 2, 3), 1: (0, 1, 2, 3)}


def campaign(shard):
    assert shard in (0, 1)
    return [configuration('ibolt', task, 4 + shard, 'e256-u025', temperature=.01)
            for task in ('hammer', 'relocate', 'door', 'pen')]


def peers(root):
    """Observe other queues; never mutate their states or signal their learners."""
    q = json.loads((root / 'priority-queue-state.json').read_text())['jobs']
    recovery = root / 'recovery-queue-state.json'
    if recovery.exists():
        q.update(json.loads(recovery.read_text())['jobs'])
    result = []
    for name, state in q.items():
        if state['status'] not in ('running', 'preflight', 'diagnostic') or not alive(state):
            continue
        assert state['status'] == 'running', ('Unrecognized peer diagnostic', name)
        verify_adopted(root, state.get('attempt_id', name), state,
                       root / ('source-' + state['source_commit']))
        result.append(state)
    return result


def slot_for(c, running, hardware, cpus, shard, preferred=None):
    candidates = (preferred,) if preferred is not None else ALLOWED_GPUS[shard]
    for gpu in candidates:
        assert gpu in ALLOWED_GPUS[shard]
        slot = choose_slot(c, running, hardware, cpus, gpu)
        if slot is not None:
            return slot
    return None


def smoke_receipt(root, name, state, commit):
    folder = root / 'preflight' / 'temp001' / name
    out = folder / 'smoke'
    completed = json.loads((out / 'completed.json').read_text())
    progress = json.loads((out / 'progress.json').read_text())
    spec = json.loads((out / 'spec.json').read_text())
    environment = json.loads((folder / 'environment-and-evaluation.json').read_text())
    assert completed['source_commit'] == spec['source_commit'] == commit
    assert progress['status'] == 'completed'
    assert progress['steps'] == 53248 and progress['updates'] == 512
    assert spec['temperature'] == environment['temperature'] == .01
    assert spec['cuda_visible_devices'] == str(state['gpu'])
    assert environment['passed'] and environment['evaluation_rng_preserved']
    assert environment['empty_standard_replay']
    restores = list(out.rglob('restore-validation.json'))
    assert restores and all(json.loads(p.read_text())['passed'] for p in restores)
    receipt = dict(passed=True, source_commit=commit, gpu=state['gpu'],
                   gpu_uuid=state['gpu_uuid'], smoke_pid=state['pid'],
                   steps=53248, updates=512, completed_unix=time.time())
    atomic(root / ('temp001-preflight-' + name + '.json'), receipt)
    return receipt


def main(root, shard):
    proof = source_provenance()
    assert os.uname().nodename == HOSTNAMES[shard]
    verify_source(ROOT, proof['commit'])
    lock = (root / 'temp001-controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_file = root / 'temp001-queue-state.json'
    assert not state_file.exists(), 'Review any previous attempt; no automatic retry'
    configs = campaign(shard)
    for c in configs:
        assert not (root / 'runs' / run_id(c)).exists()
        assert not (root / 'preflight' / 'temp001' / run_id(c)).exists()
    cpus = sorted(os.sched_getaffinity(0)); assert len(cpus) >= 96
    hardware = gpu_state()
    initial_peers = peers(root)
    for s in initial_peers:
        verify_source(root / ('source-' + s['source_commit']), s['source_commit'])
    plan = dict(source_commit=proof['commit'], source=str(ROOT), source_manifest=proof,
                hostname=os.uname().nodename, shard=shard, allowed_gpus=ALLOWED_GPUS[shard],
                gpu_uuids={str(g): hardware[g]['uuid'] for g in range(4)},
                controller_pid=os.getpid(), created_unix=time.time(),
                jobs=[dict(run_id=run_id(c), config=c) for c in configs],
                preserved_peers={s['pid']: s['start_ticks'] for s in initial_peers})
    atomic(root / 'temp001-plan.json', plan)
    states = {run_id(c): dict(status='pending', source_commit=proof['commit']) for c in configs}
    active = {}
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))

    def save():
        atomic(state_file, dict(source_commit=proof['commit'], controller_pid=os.getpid(),
                               shard=shard, jobs=states, updated_unix=time.time()))

    save()
    while True:
        for name, (proc, log) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            log.close(); del active[name]
            state = states[name]
            if state['status'] == 'preflight' and code == 0:
                state['preflight'] = smoke_receipt(root, name, state, proof['commit'])
                state.update(status='ready', pid=None, start_ticks=None)
            else:
                completed = (root / 'runs' / name / 'completed.json').exists()
                state.update(status='completed' if code == 0 and completed else 'failed',
                             returncode=code, finished_unix=time.time())
        pending = [c for c in configs if states[run_id(c)]['status'] in ('pending', 'ready')]
        # Ready jobs retain the device checked by their smoke test.
        pending.sort(key=lambda c: states[run_id(c)]['status'] != 'ready')
        if pending and not stop[0]:
            running = peers(root) + [states[n] for n in active]
            hardware = gpu_state()
            assert {str(g): hardware[g]['uuid'] for g in range(4)} == plan['gpu_uuids']
            for c in pending:
                name = run_id(c); state = states[name]
                slot = slot_for(c, running, hardware, cpus, shard,
                                state['gpu'] if state['status'] == 'ready' else None)
                if slot is None:
                    continue
                gpu, allocation = slot
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), JAX_PLATFORMS='cuda',
                           XLA_PYTHON_CLIENT_PREALLOCATE='false', OMP_NUM_THREADS='1',
                           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                           MUJOCO_GL='egl', PYTHONUNBUFFERED='1', PYTHONFAULTHANDLER='1')
                env.pop('XLA_FLAGS', None); env.pop('ADROIT_JAX_SYNCHRONIZE_UPDATES', None)
                smoke = state['status'] == 'pending'
                if smoke:
                    dest = root / 'preflight' / 'temp001' / name
                    args = ['-m', 'adroit_online.ibolt_preflight', '--root', str(dest),
                            '--task', c['task'], '--temperature', '.01']
                else:
                    dest = root / 'runs' / name
                    args = ['-m', 'adroit_online.run', '--method', 'ibolt', '--task', c['task'],
                            '--seed', str(c['seed']), '--profile', c['profile'],
                            '--temperature', '.01', '--output', str(dest)]
                assert not dest.exists()
                log = (root / 'process-logs' / (name + ('.preflight' if smoke else '') + '.log')).open('x')
                cmd = ['taskset', '-c', ','.join(map(str, allocation)), sys.executable, *args]
                proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                state.update(status='preflight' if smoke else 'running', pid=proc.pid,
                             start_ticks=process_ticks(proc.pid), gpu=gpu, cpus=allocation,
                             gpu_uuid=hardware[gpu]['uuid'], output=str(dest), command=cmd,
                             started_unix=time.time())
                active[name] = (proc, log)
                print('started', name, state['status'], 'gpu', gpu, 'pid', proc.pid, flush=True)
                save()
                break
        save()
        if not active and (not pending or stop[0]):
            break
        time.sleep(5)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--shard', type=int, choices=(0, 1), required=True)
    a = p.parse_args()
    main(a.root, a.shard)
