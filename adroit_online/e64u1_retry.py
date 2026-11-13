"""Reviewed fresh retries for the two affected GPU1 Hammer runs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import time

from .config import configuration
from .controller import gpu_state
from .e64u1_controller import GPUS, launch, smoke_args, verify_smoke
from .priority_controller import alive, process_ticks, verify_adopted, verify_source
from .run import ROOT, atomic, source_provenance

NAME = 'hammer-ibolt-s4-e64-u1-t025'
ATTEMPT = NAME + '-retry1'
ORIGINAL = '34aaa0a40ad2aab1b326bbc74c103215cebfa16b'


def main(root):
    proof = source_provenance()
    assert os.uname().nodename == '98073c6dde13'
    lock = (root / 'e64u1-retry.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = root / 'recovery-queue-state.json'; assert not path.exists()
    q = json.loads((root / 'queue-state.json').read_text())
    assert q['source_commit'] == ORIGINAL and '1' in q['quarantined']
    assert json.loads((root / 'gpu1-reviewed-stop.json').read_text())['stopped']
    error = json.loads((root / 'runs' / NAME / 'error.json').read_text())['traceback']
    assert 'CUDA_ERROR_ILLEGAL_ADDRESS' in error
    original_source = root / ('source-' + ORIGINAL)
    verify_source(original_source, ORIGINAL)
    old_manifest = json.loads((original_source / 'source-provenance.json').read_text())
    numerical = [n for n in old_manifest['files'] if n.startswith(('ibolt/', 'models/', 'configs/')) or n in
                 ('exploration.py', 'train.py', 'environment.py', 'adroit_online/config.py',
                  'adroit_online/run.py', 'adroit_online/ibolt.py', 'adroit_online/env.py', 'adroit_online/vector_env.py')]
    assert all(proof['files'][n] == old_manifest['files'][n] for n in numerical)
    affinity = sorted(os.sched_getaffinity(0)); assert len(affinity) >= 80
    states = {}
    for temp, gpu, offset in ((.25, 0, 64), (.05, 2, 72)):
        c = configuration('ibolt', 'hammer', 4, 'e64-u1', temperature=temp)
        name = f"hammer-ibolt-s4-e64-u1-t{round(temp*100):03d}"
        failed = root / 'runs' / name
        assert q['jobs'][name]['status'] == 'failed' and not alive(q['jobs'][name])
        old = json.loads((failed / 'spec.json').read_text())
        assert all(old[k] == v for k, v in c.items())
        attempt = name + '-retry1'
        assert not (root / 'runs' / attempt).exists()
        states[name] = dict(status='pending', source_commit=proof['commit'], original_source_commit=ORIGINAL,
            attempt_id=attempt, output=str(root / 'runs' / attempt), gpu=gpu, gpu_uuid=GPUS[gpu],
            cpus=affinity[offset:offset+8], config=c, numerical_files_verified=numerical,
            reviewed_reason='GPU1 memory-error evidence; fresh process on another validated GPU, unchanged science')
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    def save():
        atomic(path, dict(source_commit=proof['commit'], jobs=states, controller_pid=os.getpid(),
                         controller_start_ticks=process_ticks(os.getpid()), updated_unix=time.time()))
    active = {}; save()
    while True:
        for name, (proc, log) in list(active.items()):
            if proc.poll() is None: continue
            log.close(); del active[name]
            state = states[name]; phase = state['status']; c = state['config']
            state.update(returncode=proc.returncode, finished_unix=time.time())
            try:
                assert proc.returncode == 0, proc.returncode
                if phase == 'preflight':
                    state['preflight'] = verify_smoke(root / 'preflight' / state['attempt_id'], c,
                                                      proof['commit'], state['gpu'])
                    state['status'] = 'ready'
                else:
                    done = json.loads((Path(state['output']) / 'completed.json').read_text())
                    assert done['source_commit'] == proof['commit']
                    assert done['steps'] == c['steps'] and done['updates'] == c['expected_updates']
                    state.update(status='completed', completed=done)
            except Exception as error:
                state.update(status='failed', failed_phase=phase, error=repr(error))
            save()
        pending = [n for n,s in states.items() if s['status'] in ('pending', 'ready')]
        if pending and not stop[0]:
            q = json.loads((root / 'queue-state.json').read_text())
            peers = []
            for name, peer in q['jobs'].items():
                if peer['status'] == 'running' and alive(peer):
                    verify_adopted(root, name, peer, original_source); peers.append(peer)
            for name in pending:
                state = states[name]; gpu = state['gpu']; cpus = state['cpus']; c = state['config']
                hw = gpu_state()[gpu]; assert hw['uuid'] == GPUS[gpu]
                assert not (set(cpus) & {cpu for peer in peers for cpu in peer['cpus']})
                if (hw['free_mib'] < 12288 or set(hw['pids']) - {p['pid'] for p in peers}
                        or sum(p['gpu'] == gpu for p in peers) > 2): continue
                phase = 'running' if state['status'] == 'ready' else 'preflight'
                output = root / ('preflight' if phase == 'preflight' else 'runs') / state['attempt_id']
                assert not output.exists()
                args = smoke_args(c, output) if phase == 'preflight' else [
                    'adroit_online.run', '--method', 'ibolt', '--task', 'hammer', '--seed', '4',
                    '--profile', 'e64-u1', '--temperature', str(c['temperature']), '--output', str(output)]
                proc, log, launched = launch(args, gpu, cpus,
                    root / 'process-logs' / f"{state['attempt_id']}-{phase}.log")
                state.setdefault('history', []).append(dict(phase=phase, **launched))
                state.update(status=phase, **launched); active[name] = (proc, log); save()
                break
        if not active and (not pending or stop[0]): break
        time.sleep(5)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    main(p.parse_args().root)
