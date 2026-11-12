"""Allow only explicitly verified existing T0.25 jobs to share GPUs0/1."""
import argparse
import json
import os
from pathlib import Path
import time

from .config import IBOLT_TEMPERATURES
from .controller import gpu_state
from .run import atomic, source_provenance

PEER_ROOT = Path('/mnt/data1/heejoonorm/OptiQ/runs/adroit-online-ibolt-20261112')
PEER_COMMIT = '86354642a0936688d3b268f1c8ba10e74d1480b0'


def peer_state(root):
    result = dict(pending=0, pids={0: set(), 1: set()})
    if root is None:
        return result
    root = Path(root).resolve()
    assert root == PEER_ROOT.resolve(), 'Only the existing authorized T0.25 campaign may share GPUs'
    queue = json.loads((root / 'queue-state.json').read_text())
    assert queue['source_commit'] == PEER_COMMIT and len(queue['jobs']) == 8
    result['pending'] = sum(v['status'] == 'pending' for v in queue['jobs'].values())
    for name, job in queue['jobs'].items():
        if job['status'] != 'running': continue
        assert job['gpu'] in (0, 1) and job['source_commit'] == PEER_COMMIT
        pid = job['pid']; proc = Path('/proc') / str(pid)
        try:
            spec = json.loads((root / 'runs' / name / 'spec.json').read_text())
            if proc.stat().st_uid != os.getuid(): continue
            command = proc.joinpath('cmdline').read_bytes().decode().split('\0')
            cwd = proc.joinpath('cwd').resolve(strict=True)
            if cwd != root / ('source-' + PEER_COMMIT): continue
            if spec['pid'] != pid or spec['source_commit'] != PEER_COMMIT: continue
            if spec['method'] != 'ibolt' or spec['temperature'] != .25: continue
            if spec['profile'] != 'e256-u025' or spec['cuda_visible_devices'] != str(job['gpu']): continue
            if 'adroit_online.run' not in command or '--output' not in command: continue
            if command[command.index('--output') + 1] != str(root / 'runs' / name): continue
        except (FileNotFoundError, ProcessLookupError, PermissionError, json.JSONDecodeError):
            continue
        result['pids'][job['gpu']].add(pid)
    return result


def wait_for_admission(peer_root, receipt):
    previous = None
    while True:
        peers = peer_state(peer_root); hardware = gpu_state()
        foreign = {g: sorted(set(hardware[g]['pids']) - peers['pids'][g]) for g in (0, 1)}
        ready = (not peers['pending'] and not any(foreign.values())
                 and all(hardware[g]['free_mib'] >= 12000 for g in (0, 1)))
        state = dict(ready=ready, waiting_for_peer_starts=peers['pending'], foreign_pids=foreign,
                     verified_peer_pids={g: sorted(peers['pids'][g]) for g in (0, 1)},
                     updated_unix=time.time())
        atomic(receipt, state)
        if ready: return
        message = (peers['pending'], str(foreign))
        if message != previous:
            print('Waiting for GPU admission:', message, flush=True); previous = message
        # The immutable old controller regards new jobs as foreign. Let its
        # already queued seeds START first so it is never starved by this queue.
        # We do not wait for the old runs to finish and never modify their PIDs.
        time.sleep(15)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--peer-root', type=Path)
    p.add_argument('--temperatures', type=float, nargs='+', choices=IBOLT_TEMPERATURES, default=[.25])
    a = p.parse_args(); source_provenance()
    wait_for_admission(a.peer_root, a.root / 'runtime' / 'gpu-admission.json')


if __name__ == '__main__':
    main()
