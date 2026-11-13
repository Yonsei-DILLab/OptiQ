"""One-shot, provenance-checked iBOLT env64/UTD1 queue on Vast60326900."""
import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .config import configuration, run_id
from .controller import gpu_state
from .priority_controller import process_ticks
from .run import ROOT, atomic, source_provenance

GPUS = {
    0: 'GPU-26e7ef72-6e3e-3a71-38ba-6227f918b5e1',
    1: 'GPU-6c338831-d993-1eaa-4c73-9ad26367f403',
    2: 'GPU-ad3246bf-02de-546f-2b07-087120a7b123',
    3: 'GPU-fccd2074-5595-0b9a-691f-b5dfd926b0e6',
}


def campaign():
    return [configuration('ibolt', task, seed, 'e64-u1', temperature=temp)
            for seed in (4, 5) for temp in (.05, .10, .25) for task in ('door', 'hammer')]


def slot_for(running, hardware, cpus, allowed, preferred=None):
    known = {s['pid'] for s in running}
    occupied = {tuple(s['cpus']) for s in running}
    for slot in range(2):
        for gpu in allowed:
            if preferred is not None and gpu != preferred:
                continue
            allocation = cpus[(slot * 4 + gpu) * 8:(slot * 4 + gpu + 1) * 8]
            assert len(allocation) == 8
            info = hardware[gpu]
            assert info['uuid'] == GPUS[gpu]
            if (tuple(allocation) not in occupied and info['free_mib'] >= 12288
                    and not set(info['pids']) - known):
                return gpu, allocation
    return None


def launch(args, gpu, cpus, log_path):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), JAX_PLATFORMS='cuda',
               XLA_PYTHON_CLIENT_PREALLOCATE='false', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               MUJOCO_GL='egl', PYTHONUNBUFFERED='1', PYTHONFAULTHANDLER='1')
    env.pop('XLA_FLAGS', None)
    env.pop('ADROIT_JAX_SYNCHRONIZE_UPDATES', None)
    cmd = ['taskset', '-c', ','.join(map(str, cpus)), sys.executable, '-m', *args]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open('x')
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    state = dict(pid=proc.pid, start_ticks=process_ticks(proc.pid), gpu=gpu,
                 gpu_uuid=GPUS[gpu], cpus=cpus, command=cmd, started_unix=time.time())
    print(json.dumps(dict(event='started', **state)), flush=True)
    return proc, log, state


def smoke_args(c, output):
    return ['adroit_online.ibolt_preflight', '--root', str(output), '--task', c['task'],
            '--temperature', str(c['temperature']), '--profile', c['profile']]


def verify_smoke(output, c, commit, gpu):
    def read(name):
        return json.loads((output / name).read_text())
    environment = read('environment-and-evaluation.json')
    assert environment['passed'] and environment['evaluation_rng_preserved']
    assert environment['empty_standard_replay'] and environment['jax_devices']
    assert environment['profile'] == 'e64-u1' and environment['num_envs'] == 64
    assert environment['task'] == c['task'] and environment['temperature'] == c['temperature']
    spec, progress, complete = (read('smoke/' + x + '.json') for x in ('spec', 'progress', 'completed'))
    expected = configuration('ibolt', c['task'], 4, 'e64-u1', smoke=True,
                             temperature=c['temperature'])
    assert all(spec[k] == v for k, v in expected.items())
    assert spec['source_commit'] == complete['source_commit'] == commit
    assert spec['cuda_visible_devices'] == str(gpu)
    assert progress['status'] == 'completed'
    assert progress['steps'] == complete['steps'] == expected['steps'] == 53248
    assert progress['updates'] == complete['updates'] == expected['expected_updates'] == 2048
    restored = read('smoke/checkpoints/restore-validation.json')
    assert restored['passed'] and restored['actor_and_critic_optimizer_exact']
    return dict(passed=True, source_commit=commit, gpu=gpu, gpu_uuid=GPUS[gpu],
                steps=53248, updates=2048, profile='e64-u1', output=str(output),
                checked_unix=time.time())


def device_checks(root, commit, cpus, stop):
    """Fresh arithmetic + identical native Hammer smoke on GPUs0/1.

    Admission is conditional; failures remain evidence, never auto-retried.
    """
    states = {str(g): dict(status='pending_hardware') for g in (0, 1)}
    active = {}
    c = configuration('ibolt', 'hammer', 4, 'e64-u1', temperature=.10)
    while True:
        for g, (proc, log) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            log.close(); del active[g]
            state = states[str(g)]
            phase = state['status']
            state.update(returncode=code, finished_unix=time.time())
            try:
                assert code == 0, code
                if phase == 'hardware':
                    r = json.loads((root / 'device-checks' / f'gpu{g}-hardware.json').read_text())
                    assert r['passed'] and r['source_commit'] == commit
                    state.update(status='pending_smoke', hardware=r)
                else:
                    r = verify_smoke(root / 'device-checks' / f'gpu{g}-smoke', c, commit, g)
                    state.update(status='passed', smoke=r)
            except Exception as error:
                state.update(status='failed', error=repr(error), failed_phase=phase)
        if not stop[0]:
            for g in (0, 1):
                state = states[str(g)]
                if not state['status'].startswith('pending_'):
                    continue
                hw = gpu_state()[g]
                assert hw['uuid'] == GPUS[g]
                if hw['pids'] or hw['free_mib'] < 12288:
                    continue
                phase = state['status'].removeprefix('pending_')
                args = (['adroit_online.hardware_probe', '--output',
                         str(root / 'device-checks' / f'gpu{g}-hardware.json')]
                        if phase == 'hardware' else smoke_args(c, root / 'device-checks' / f'gpu{g}-smoke'))
                proc, log, launched = launch(args, g, cpus[g * 8:(g + 1) * 8],
                                            root / 'process-logs' / f'device-gpu{g}-{phase}.log')
                state.setdefault('history', []).append(dict(phase=phase, **launched))
                state.update(status=phase, **launched)
                active[g] = (proc, log)
                break
        atomic(root / 'device-check-state.json', dict(source_commit=commit, devices=states,
                                                     updated_unix=time.time()))
        if not active and (stop[0] or all(s['status'] in ('passed', 'failed') for s in states.values())):
            break
        time.sleep(5)
    return states


def main(root):
    proof = source_provenance()
    assert os.uname().nodename == '98073c6dde13'
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (root / 'queue-state.json').exists(), 'No automatic restart/overwrite'
    cpus = sorted(os.sched_getaffinity(0)); assert len(cpus) >= 64
    hw = gpu_state()
    assert {g: h['uuid'] for g, h in hw.items()} == GPUS
    assert not any(h['pids'] for h in hw.values()), 'Review unrelated GPU work before starting'
    jobs = campaign()
    assert all(not (root / 'runs' / run_id(c)).exists() for c in jobs)
    atomic(root / 'plan.json', dict(source_commit=proof['commit'], source=str(ROOT),
           source_manifest=proof, hostname=os.uname().nodename, gpu_uuids=GPUS,
           maximum_jobs_per_gpu=2, controller_pid=os.getpid(), controller_start_ticks=process_ticks(os.getpid()),
           requested_gpu1_recheck=True, jobs=jobs, created_unix=time.time()))
    robotics = importlib.metadata.distribution('gymnasium-robotics')
    atomic(root / 'runtime.json', dict(python=sys.version, executable=sys.executable,
           packages={x.metadata['Name']: x.version for x in importlib.metadata.distributions()},
           robotics_direct_url=json.loads(robotics.read_text('direct_url.json'))))
    states = {run_id(c): dict(status='pending', source_commit=proof['commit']) for c in jobs}
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    atomic(root / 'queue-state.json', dict(source_commit=proof['commit'], phase='device_checks', jobs=states))
    checked = device_checks(root, proof['commit'], cpus, stop)
    allowed = [2, 3]
    if checked['0']['status'] == 'passed':
        allowed.insert(0, 0)
        if checked['1']['status'] == 'passed':
            allowed.insert(1, 1)
    atomic(root / 'gpu-admission.json', dict(source_commit=proof['commit'], allowed_gpus=allowed,
           device_checks=checked, limitation='Short validation cannot exclude intermittent recurrence.'))
    active, quarantined = {}, {}
    while True:
        for name, (proc, log) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            log.close(); del active[name]
            state = states[name]
            try:
                assert code == 0, code
                if state['status'] == 'preflight':
                    c = next(c for c in jobs if run_id(c) == name)
                    receipt = verify_smoke(root / 'preflight' / name, c, proof['commit'], state['gpu'])
                    state.update(status='ready', preflight=receipt)
                else:
                    c = next(c for c in jobs if run_id(c) == name)
                    finished = json.loads((root / 'runs' / name / 'completed.json').read_text())
                    assert finished['steps'] == c['steps'] and finished['updates'] == c['expected_updates']
                    assert finished['source_commit'] == proof['commit']
                    state.update(status='completed', completed=finished)
            except Exception as error:
                quarantined[state['gpu']] = dict(run_id=name, error=repr(error), time=time.time())
                state.update(failed_phase=state['status'], status='failed', error=repr(error))
            state.update(returncode=code, finished_unix=time.time())
        for state in states.values():
            if state['status'] == 'ready' and state['gpu'] in quarantined:
                state.update(status='requires_review', reason='Preflight GPU quarantined after peer failure')
        pending = sorted((c for c in jobs if states[run_id(c)]['status'] in ('pending', 'ready')),
                         key=lambda c: states[run_id(c)]['status'] != 'ready')
        eligible = [g for g in allowed if g not in quarantined]
        if pending and not stop[0]:
            hardware = gpu_state()
            running = [states[name] for name in active]
            for c in pending:
                name = run_id(c); state = states[name]
                preferred = state['gpu'] if state['status'] == 'ready' else None
                slot = slot_for(running, hardware, cpus, eligible, preferred)
                if slot is None:
                    continue
                gpu, allocation = slot
                phase = 'running' if state['status'] == 'ready' else 'preflight'
                output = root / ('runs' if phase == 'running' else 'preflight') / name
                assert not output.exists()
                args = (['adroit_online.run', '--method', 'ibolt', '--task', c['task'],
                         '--seed', str(c['seed']), '--profile', c['profile'],
                         '--temperature', str(c['temperature']), '--output', str(output)]
                        if phase == 'running' else smoke_args(c, output))
                proc, log, launched = launch(args, gpu, allocation,
                    root / 'process-logs' / f'{name}-{phase}.log')
                active[name] = (proc, log)
                state.setdefault('history', []).append(dict(phase=phase, **launched))
                state.update(status=phase, output=str(output), **launched)
                break
        atomic(root / 'queue-state.json', dict(source_commit=proof['commit'], phase='learning', jobs=states,
               allowed_gpus=allowed, quarantined=quarantined, draining=stop[0], updated_unix=time.time()))
        if not active and (not pending or stop[0] or not eligible):
            break
        time.sleep(5)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    main(p.parse_args().root)
