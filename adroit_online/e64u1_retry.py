"""Reviewed, single fresh retry after the GPU1 illegal-address recurrence."""
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
    failed = root / 'runs' / NAME
    assert q['source_commit'] == ORIGINAL and q['jobs'][NAME]['status'] == 'failed'
    assert q['jobs'][NAME]['gpu'] == 1 and not alive(q['jobs'][NAME])
    assert 'CUDA_ERROR_ILLEGAL_ADDRESS' in json.loads((failed / 'error.json').read_text())['traceback']
    assert '1' in q['quarantined']
    c = configuration('ibolt', 'hammer', 4, 'e64-u1', temperature=.25)
    old = json.loads((failed / 'spec.json').read_text())
    assert all(old[k] == v for k, v in c.items())
    original_source = root / ('source-' + ORIGINAL)
    verify_source(original_source, ORIGINAL)
    old_manifest = json.loads((original_source / 'source-provenance.json').read_text())
    numerical = [n for n in old_manifest['files'] if n.startswith(('ibolt/', 'models/', 'configs/')) or n in
                 ('exploration.py', 'train.py', 'environment.py', 'adroit_online/config.py',
                  'adroit_online/run.py', 'adroit_online/ibolt.py', 'adroit_online/env.py', 'adroit_online/vector_env.py')]
    assert all(proof['files'][n] == old_manifest['files'][n] for n in numerical)
    cpus = sorted(os.sched_getaffinity(0))[64:72]; assert len(cpus) == 8
    state = dict(status='pending', source_commit=proof['commit'], original_source_commit=ORIGINAL,
                 attempt_id=ATTEMPT, output=str(root / 'runs' / ATTEMPT), gpu=0,
                 gpu_uuid=GPUS[0], cpus=cpus, config=c, numerical_files_verified=numerical,
                 reviewed_reason='GPU1 CUDA illegal address; fresh process on GPU0, unchanged science',
                 controller_pid=os.getpid(), controller_start_ticks=process_ticks(os.getpid()))
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    def save():
        atomic(path, dict(source_commit=proof['commit'], jobs={NAME: state}, updated_unix=time.time()))
    save()
    for phase in ('preflight', 'running'):
        if stop[0]:
            state.update(status='requires_review', reason='Controller drained before next phase'); save(); return
        while True:
            q = json.loads((root / 'queue-state.json').read_text())
            peers = []
            for name, peer in q['jobs'].items():
                if peer['status'] == 'running' and alive(peer):
                    verify_adopted(root, name, peer, original_source); peers.append(peer)
            hw = gpu_state()[0]; assert hw['uuid'] == GPUS[0]
            assert not (set(cpus) & {cpu for peer in peers for cpu in peer['cpus']})
            if (hw['free_mib'] >= 12288 and not set(hw['pids']) - {p['pid'] for p in peers}
                    and sum(p['gpu'] == 0 for p in peers) <= 2):
                break
            if stop[0]:
                state.update(status='requires_review', reason='Controller drained waiting for admission'); save(); return
            time.sleep(5)
        output = root / ('preflight' if phase == 'preflight' else 'runs') / ATTEMPT
        assert not output.exists()
        args = smoke_args(c, output) if phase == 'preflight' else [
            'adroit_online.run', '--method', 'ibolt', '--task', 'hammer', '--seed', '4',
            '--profile', 'e64-u1', '--temperature', '.25', '--output', str(output)]
        proc, log, launched = launch(args, 0, cpus, root / 'process-logs' / f'{ATTEMPT}-{phase}.log')
        state.setdefault('history', []).append(dict(phase=phase, **launched))
        state.update(status=phase, **launched); save()
        while proc.poll() is None:
            time.sleep(5)
        log.close()
        state.update(returncode=proc.returncode, finished_unix=time.time())
        try:
            assert proc.returncode == 0, proc.returncode
            if phase == 'preflight':
                state['preflight'] = verify_smoke(output, c, proof['commit'], 0)
            else:
                done = json.loads((output / 'completed.json').read_text())
                assert done['source_commit'] == proof['commit']
                assert done['steps'] == c['steps'] and done['updates'] == c['expected_updates']
                state.update(status='completed', completed=done)
        except Exception as error:
            state.update(status='failed', failed_phase=phase, error=repr(error)); save(); return
        save()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    main(p.parse_args().root)
