"""Migrate unstarted iBOLT jobs; preserve learners while draining old queues.

The frozen plan is the scheduling authority. Old controllers receive SIGTERM
directly (their handlers drain children); never use Supervisor's group stop.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .config import configuration, run_id
from .controller import gpu_state
from .run import ROOT, atomic, source_provenance


def process_ticks(pid):
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(') ', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except FileNotFoundError:
        return None


def alive(state):
    return bool(state.get('pid') and process_ticks(state['pid']) == state.get('start_ticks'))


def verify_source(source, commit):
    proof = json.loads((source / 'source-provenance.json').read_text())
    assert proof['commit'] == commit and proof['branch'] == 'heejoon-rebuttal'
    for name, digest in proof['files'].items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest, name


def verify_adopted(root, name, state, source):
    proc = Path('/proc') / str(state['pid'])
    assert proc.stat().st_uid == os.getuid()
    assert (proc / 'cwd').resolve() == source
    cmd = (proc / 'cmdline').read_bytes().decode().split(chr(0))
    assert 'adroit_online.run' in cmd
    assert cmd[cmd.index('--output') + 1] == str(root / 'runs' / name)
    spec = json.loads((root / 'runs' / name / 'spec.json').read_text())
    assert spec['pid'] == state['pid'] and spec['source_commit'] == state['source_commit']
    assert spec['cuda_visible_devices'] == str(state['gpu'])
    assert spec['cpu_affinity'] == state['cpus']
    state['start_ticks'] = process_ticks(state['pid'])
    assert state['start_ticks'] is not None


def choose_slot(config, running, hardware, cpus, preferred_gpu=None):
    """Keep two baseline slots and one additional iBOLT slot per GPU."""
    known = {s['pid'] for s in running}
    used_cpus = {c for s in running for c in s['cpus']}
    candidates = [(g, 2) for g in range(4)] if config['method'] == 'ibolt' else [
        (g, k) for k in range(2) for g in range(4)]
    for gpu, slot in candidates:
        if preferred_gpu is not None and gpu != preferred_gpu:
            continue
        info = hardware[gpu]
        # Include learners that are importing and have not allocated CUDA yet.
        if sum(s['gpu'] == gpu for s in running) >= 3:
            continue
        if info['free_mib'] < 12000 or set(info['pids']) - known:
            continue
        allocation = cpus[(slot * 4 + gpu) * 8:(slot * 4 + gpu + 1) * 8]
        assert len(allocation) == 8
        if not set(allocation) & used_cpus:
            return gpu, allocation
    return None


def initial_states(root, plan):
    legacy = json.loads((root / 'queue-state.json').read_text())['jobs']
    states = {}
    for job in plan['jobs']:
        name, config = job['run_id'], job['config']
        assert name == run_id(config)
        assert config == configuration(config['method'], config['task'], config['seed'],
                                       config['profile'], temperature=config.get('temperature'))
        dest = root / 'runs' / name
        if job['origin'] == 'dilg-01':
            assert not dest.exists(), ('Already started; do not overwrite', name)
            state = dict(status='pending', source_commit=plan['operation_commit'])
        else:
            state = dict(legacy[name])
            state['source_commit'] = job['source_commit']
            if (dest / 'completed.json').exists() and state['status'] != 'running':
                state['status'] = 'completed'
            elif state['status'] == 'running':
                if process_ticks(state['pid']) is None:
                    assert (dest / 'completed.json').exists(), name
                    state['status'] = 'completed'
                else:
                    verify_adopted(root, name, state, Path(job['source']))
            elif state['status'] == 'pending':
                assert not dest.exists(), name
            else:
                assert state['status'] == 'completed', (name, state)
        state.update(origin=job['origin'], priority=job['priority'])
        states[name] = state
    return states


def smoke_receipt(root, name, state, commit):
    out = root / 'preflight' / 'migration-5090' / name / 'smoke'
    progress = json.loads((out / 'progress.json').read_text())
    assert (out / 'completed.json').exists() and progress['status'] == 'completed'
    assert progress['steps'] == 53248 and progress['updates'] == 512
    spec = json.loads((out / 'spec.json').read_text())
    assert spec['source_commit'] == commit
    restores = list(out.rglob('restore-validation.json'))
    assert restores and all(json.loads(p.read_text())['passed'] for p in restores)
    receipt = dict(passed=True, source_commit=commit, gpu=state['gpu'],
                   gpu_uuid=state['gpu_uuid'], smoke_pid=state['pid'],
                   steps=53248, updates=512, completed_unix=time.time())
    atomic(root / ('migration-preflight-' + name + '.json'), receipt)
    return receipt


def main(root):
    proof = source_provenance()
    lock = (root / 'priority-controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads((root / 'priority-plan.json').read_text())
    assert plan['operation_commit'] == proof['commit'] and plan['hostname'] == os.uname().nodename
    assert plan['max_jobs_per_gpu'] == 3 and plan['baseline_slots_per_gpu'] == 2
    assert len(plan['jobs']) == len({j['run_id'] for j in plan['jobs']}) == 28
    delegated_path = root / 'origin-delegation.json'
    assert hashlib.sha256(delegated_path.read_bytes()).hexdigest() == plan['origin_delegation_sha256']
    delegated = json.loads(delegated_path.read_text())['jobs']
    for job in plan['jobs']:
        if job['origin'] == 'dilg-01':
            target = delegated[job['run_id']]
            assert target['shard'] == plan['shard'] and target['root'] == str(root)
            assert target['source_commit'] == proof['commit']
    drain = json.loads((root / 'migration-drain.json').read_text())
    assert drain['signal'].startswith('SIGTERM to controller PID only')
    assert drain['controller']['pid'] == plan['drained_controller']['pid']
    assert drain['controller']['start_ticks'] == plan['drained_controller']['start_ticks']
    assert time.time() - drain['sent_unix'] > 15
    hardware = gpu_state()
    assert {str(g): hardware[g]['uuid'] for g in range(4)} == plan['gpu_uuids']
    cpus = sorted(os.sched_getaffinity(0)); assert len(cpus) >= 96
    for source, commit in {(j['source'], j['source_commit']) for j in plan['jobs']}:
        verify_source(Path(source), commit)
    state_file = root / 'priority-queue-state.json'
    assert not state_file.exists(), 'No automatic scheduler retry; review existing PID/output state first'
    states = initial_states(root, plan)
    active = {}
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    baseline_names = [j['run_id'] for j in plan['jobs'] if j['origin'] == 'baseline']
    while True:
        for name, state in states.items():
            if state['status'] not in ('running', 'preflight'):
                continue
            code = None
            if name in active:
                proc, log = active[name]
                code = proc.poll()
                if code is None:
                    continue
                log.close(); del active[name]
            elif alive(state):
                continue
            if state['status'] == 'preflight':
                if code == 0:
                    state['preflight'] = smoke_receipt(root, name, state, proof['commit'])
                    state.update(status='ready', pid=None, start_ticks=None)
                else:
                    state.update(status='failed', phase='preflight', returncode=code)
            else:
                complete = (root / 'runs' / name / 'completed.json').exists()
                state.update(status='completed' if complete and code in (None, 0) else 'failed',
                             returncode=code, finished_unix=time.time())
        pending = sorted((j for j in plan['jobs'] if states[j['run_id']]['status'] in ('pending', 'ready')),
                         key=lambda j: (j['priority'], j['order']))
        running = [s for s in states.values() if s['status'] in ('running', 'preflight')]
        if pending and not stop[0]:
            hardware = gpu_state()
            for job in pending:
                name, c = job['run_id'], job['config']; state = states[name]
                # Prefer all unstarted iBOLT jobs before lower-priority baselines.
                if c['method'] != 'ibolt' and any(j['config']['method'] == 'ibolt' for j in pending):
                    continue
                slot = choose_slot(c, running, hardware, cpus, job.get('preferred_gpu'))
                if slot is None:
                    continue
                gpu, allocation = slot
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), JAX_PLATFORMS='cuda',
                    XLA_PYTHON_CLIENT_PREALLOCATE='false', OMP_NUM_THREADS='1',
                    OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                    MUJOCO_GL='egl', PYTHONUNBUFFERED='1')
                smoke = c['method'] == 'ibolt' and state['status'] == 'pending'
                if smoke:
                    dest = root / 'preflight' / 'migration-5090' / name
                    assert not dest.exists()
                    args = ['-m', 'adroit_online.ibolt_preflight', '--root', str(dest),
                            '--task', c['task'], '--temperature', str(c['temperature'])]
                else:
                    dest = root / 'runs' / name; assert not dest.exists()
                    args = ['-m', 'adroit_online.run', '--method', c['method'], '--task', c['task'],
                            '--seed', str(c['seed']), '--profile', c['profile'], '--output', str(dest)]
                    if c['method'] == 'ibolt':
                        assert state['preflight']['passed']
                        args += ['--temperature', str(c['temperature'])]
                log = (root / 'process-logs' / (name + ('.preflight' if smoke else '') + '.log')).open('a')
                cmd = ['taskset', '-c', ','.join(map(str, allocation)), job['python'], *args]
                proc = subprocess.Popen(cmd, cwd=job['source'], env=env, stdout=log, stderr=subprocess.STDOUT)
                state.update(status='preflight' if smoke else 'running', pid=proc.pid,
                             start_ticks=process_ticks(proc.pid), gpu=gpu, gpu_uuid=hardware[gpu]['uuid'],
                             cpus=allocation, started_unix=time.time(), source_commit=job['source_commit'],
                             operation_commit=proof['commit'])
                active[name] = (proc, log)
                print('started', name, state['status'], 'gpu', gpu, 'pid', proc.pid, flush=True)
                break
        atomic(state_file, dict(source_commit=proof['commit'], controller_pid=os.getpid(),
                               authoritative=True, jobs=states, updated_unix=time.time()))
        # Old queue files are written only after their original controller exits.
        if not alive(plan['drained_controller']):
            atomic(root / 'queue-state.json', dict(source_commit=plan['baseline_source_commit'],
                managed_by='priority-queue-state.json', jobs={n: states[n] for n in baseline_names},
                updated_unix=time.time()))
        if not running and (not pending or stop[0]):
            break
        time.sleep(5)


def finalize_origin(root):
    delegation = json.loads((root / 'queue-delegation.json').read_text())
    drain = json.loads((root / 'migration-drain.json').read_text())
    while alive(drain['controller']):
        time.sleep(15)
    path = root / 'queue-state.json'; queue = json.loads(path.read_text())
    for name, destination in delegation['jobs'].items():
        assert queue['jobs'][name]['status'] == 'pending' and not (root / 'runs' / name).exists()
        queue['jobs'][name].update(status='migrated', destination=destination)
    queue['updated_unix'] = time.time(); atomic(path, queue)
    atomic(root / 'migration-finalized.json', dict(finished_unix=time.time(), jobs=delegation['jobs'],
                                                  no_further_origin_launches=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--finalize-origin', action='store_true'); args = parser.parse_args()
    finalize_origin(args.root) if args.finalize_origin else main(args.root)
