#!/usr/bin/env python3
"""Run fixed-checkpoint CFM checks after supervised fits finish; no training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def parent_ready(job):
    status = subprocess.run(
        ['supervisorctl', 'status', job['parent_program']],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    fields = status.split()
    if len(fields) < 2 or fields[1] != 'EXITED':
        return False, status
    folder = ROOT / job['cfm_run']
    required = ['config.json', 'summary.json', 'history.jsonl', 'best_cfm.pt']
    if not all((folder / name).is_file() for name in required):
        return False, 'Parent exited without all completed evaluation files'
    config = json.loads((folder / 'config.json').read_text())
    rows = [json.loads(line) for line in (folder / 'history.jsonl').read_text().splitlines() if line]
    if not rows or max(row['cfm_update'] for row in rows) != config['updates']:
        return False, 'Parent has not completed the configured CFM update budget'
    return True, status


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('worker', type=int, choices=range(4))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    jobs = [job for job in manifest['jobs'] if job['gpu'] == args.worker]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.worker),
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    child = None
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    state_path = ROOT / manifest['status_root'] / f'worker{args.worker}.json'
    states = {}
    if not args.dry_run:
        state_path.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(updated_unix=time.time(), jobs=states), indent=2) + '\n')
        temporary.replace(state_path)

    pending = list(jobs)
    while pending and not stopping:
        for job in list(pending):
            name = job['parent_program']
            ready, status = parent_ready(job)
            command = [sys.executable, '-m', 'benchmarks.gmm40.post_evaluate_cfm',
                       '--cfm-run', job['cfm_run'], '--output', job['output'],
                       '--tolerances', '.001', '.0001', '--ess-batch16-repeats', '8']
            if args.dry_run:
                print(json.dumps(dict(parent=name, ready=ready, status=status, command=command)), flush=True)
                continue
            if not ready:
                states[name] = dict(state='waiting', parent_status=status)
                continue
            output = ROOT / job['output']
            checkpoint = ROOT / job['cfm_run'] / 'best_cfm.pt'
            checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            if (output / 'summary.json').exists():
                summary = json.loads((output / 'summary.json').read_text())
                if summary['checkpoint_sha256'] != checkpoint_sha:
                    raise RuntimeError('Existing post-evaluation belongs to a different checkpoint')
                states[name] = dict(state='completed', output=job['output'], wandb_url=summary['wandb_url'])
                pending.remove(job)
                continue
            if output.exists():
                states[name] = dict(state='failed', reason='Existing incomplete output retained', output=job['output'])
                pending.remove(job)
                continue
            states[name] = dict(state='running', checkpoint_sha256=checkpoint_sha, command=command)
            save()
            print(json.dumps(states[name]), flush=True)
            child = subprocess.Popen(command, cwd=ROOT, env=env)
            code = child.wait()
            child = None
            if code == 0 and (output / 'summary.json').exists():
                summary = json.loads((output / 'summary.json').read_text())
                if summary['checkpoint_sha256'] != checkpoint_sha:
                    raise RuntimeError('CFM checkpoint changed during fixed-checkpoint evaluation')
                states[name] = dict(state='completed', output=job['output'], wandb_url=summary['wandb_url'])
            else:
                states[name] = dict(state='failed', exit_code=code, output=job['output'])
            pending.remove(job)
            save()
            if stopping:
                break
        if args.dry_run:
            return
        save()
        if pending and not stopping:
            time.sleep(30)
    sys.exit(1 if any(row['state'] == 'failed' for row in states.values()) else 0)


if __name__ == '__main__':
    main()
