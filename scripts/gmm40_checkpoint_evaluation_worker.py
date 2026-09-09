#!/usr/bin/env python3
"""Evaluate fixed serialized checkpoints as they become available; no training edits."""
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
TRAIN_PYTHON = '/workspace/.venv-optiq-no-anchor/bin/python'
EVAL_PYTHON = '/venv/main/bin/python'


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint_ready(job):
    """Timing records are appended only AFTER full checkpoint serialization."""
    checkpoint = ROOT / job['checkpoint']
    timing = checkpoint.parent / 'training_timing.jsonl'
    records = []
    if timing.exists():
        # A concurrent writer can leave its final line incomplete for one read.
        lines = timing.read_text().splitlines(keepends=True)
        records = [json.loads(line) for line in lines if line.endswith('\n')]
    matching = [row for row in records if row['checkpoint'] == checkpoint.name
                and row['update'] == job['update']]
    ready = checkpoint.is_file() and bool(matching)
    if job.get('require_completed', False):
        completion = checkpoint.parent / 'completed.json'
        ready = ready and completion.is_file()
        if ready:
            record = read(completion)
            ready = record.get('finished') is True and record['updates'] == job['update']
    if ready:
        return True, 'Serialized checkpoint and matching timing record are available'
    status = subprocess.run(['supervisorctl', 'status', job['parent_program']],
                            text=True, capture_output=True, check=False).stdout.strip()
    fields = status.split()
    if len(fields) >= 2 and fields[1] in ('EXITED', 'STOPPED', 'FATAL'):
        raise RuntimeError('Parent is terminal without the requested checkpoint: ' + status)
    return False, status


def verify_export(job, directory):
    info = read(directory / 'summary.json')
    checkpoint = (ROOT / job['checkpoint']).resolve()
    if (Path(info['checkpoint']).resolve() != checkpoint
            or info['checkpoint_sha256'] != digest(checkpoint)
            or info['actor_update'] != job['update']
            or info['count'] != 100000
            or info['seed'] != 20260930
            or info['samples_sha256'] != digest(directory / 'samples.npy')):
        raise RuntimeError('Fixed-checkpoint export identity mismatch: ' + str(directory))
    for name, value in job.get('expected_config', {}).items():
        if info['parent_config'].get(name) != value:
            raise RuntimeError('Export does not match frozen configuration: ' + name)
    return info['samples_sha256']


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('worker', type=int, choices=range(4))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    manifest = read(args.manifest)
    jobs = [job for job in manifest['jobs'] if job['gpu'] == args.worker]
    status_path = ROOT / manifest['status_root'] / f'worker{args.worker}.json'
    states, child, stopping = {}, None, False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    def save():
        status_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(updated_unix=time.time(), jobs=states), indent=2) + '\n')
        temporary.replace(status_path)

    def stage(job, key, python, module, arguments, cpu=True):
        nonlocal child
        if stopping:
            raise InterruptedError('Checkpoint evaluation stopped')
        output = ROOT / job[key]
        if (output / 'summary.json').is_file():
            return
        if output.exists():
            raise RuntimeError('Incomplete output retained without overwrite: ' + str(output))
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                   PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES='' if cpu else str(args.worker))
        if cpu:
            env['JAX_PLATFORMS'] = 'cpu'
        else:
            env.pop('JAX_PLATFORMS', None)
        command = [python, '-m', module, *map(str, arguments), '--output', str(output)]
        states[job['label']] = dict(state=key, command=command)
        save()
        print(json.dumps(states[job['label']]), flush=True)
        child = subprocess.Popen(command, cwd=ROOT, env=env)
        code = child.wait()
        child = None
        if stopping:
            raise InterruptedError('Checkpoint evaluation stopped')
        if code or not (output / 'summary.json').is_file():
            raise RuntimeError(f'{key} failed with exit {code}; existing outputs retained')

    for job in jobs:
        label = job['label']
        try:
            ready, reason = checkpoint_ready(job)
            if args.dry_run:
                print(json.dumps(dict(label=label, ready=ready, reason=reason, job=job)), flush=True)
                continue
            while not ready and not stopping:
                states[label] = dict(state='waiting', reason=reason)
                save()
                time.sleep(15)
                ready, reason = checkpoint_ready(job)
            if stopping:
                raise InterruptedError('Checkpoint evaluation stopped')
            export = ROOT / job['export']
            stage(job, 'export', TRAIN_PYTHON, 'benchmarks.gmm40.export_samples',
                  ['--checkpoint', ROOT / job['checkpoint'], '--count', 100000, '--seed', 20260930])
            sample_hash = verify_export(job, export)
            samples = export / 'samples.npy'
            stage(job, 'paper', EVAL_PYTHON, 'benchmarks.gmm40.paper_sample_evaluate',
                  ['--samples', samples, '--label', label, '--no-baselines',
                   *(['--confirmation'] if job.get('confirmation', False) else [])])
            stage(job, 'kde', EVAL_PYTHON, 'benchmarks.gmm40.kde_validation', ['--samples', samples])
            for key in ['paper', 'kde']:
                if read(ROOT / job[key] / 'summary.json')['config']['samples_sha256'] != sample_hash:
                    raise RuntimeError('Sample evaluation belongs to a different export')
            if job.get('confirmation', False) and not read(ROOT / job['paper'] / 'summary.json')['config'].get('confirmation'):
                raise RuntimeError('Expected a declared independent-seed confirmation evaluation')
            stage(job, 'timing', TRAIN_PYTHON, 'benchmarks.gmm40.timing_audit', ['--exports', export])
            timing = read(ROOT / job['timing'] / 'summary.json')['candidates']
            if len(timing) != 1 or timing[0]['stages'][-1]['checkpoint_sha256'] != digest(ROOT / job['checkpoint']):
                raise RuntimeError('Timing evaluation belongs to a different checkpoint')
            if job.get('cfm'):
                stage(job, 'cfm', EVAL_PYTHON, 'benchmarks.gmm40.idem_evaluate',
                      ['--samples', samples, '--seed', 0, '--updates', 100000,
                       '--sampling-batch-size', 256], cpu=False)
                cfg = read(ROOT / job['cfm'] / 'config.json')
                history = (ROOT / job['cfm'] / 'history.jsonl').read_text().splitlines()
                if (cfg['samples_sha256'] != sample_hash or cfg['updates'] != 100000
                        or max(json.loads(line)['cfm_update'] for line in history) != 100000):
                    raise RuntimeError('Incomplete or mismatched auxiliary CFM fit')
                stage(job, 'post', EVAL_PYTHON, 'benchmarks.gmm40.post_evaluate_cfm',
                      ['--cfm-run', ROOT / job['cfm'], '--tolerances', .001, .0001,
                       '--ess-batch16-repeats', 8], cpu=False)
                post = read(ROOT / job['post'] / 'summary.json')
                if post['checkpoint_sha256'] != digest(ROOT / job['cfm'] / 'best_cfm.pt'):
                    raise RuntimeError('Post evaluation belongs to a different CFM checkpoint')
                for reference in job.get('reference_posts', []):
                    key = 'reference_post_' + str(reference['seed'])
                    reference_job = dict(job, **{key: reference['output']})
                    stage(reference_job, key, EVAL_PYTHON, 'benchmarks.gmm40.post_evaluate_cfm',
                          ['--cfm-run', ROOT / job['cfm'], '--reference-seed', reference['seed'],
                           '--tolerances', .001, '--ess-batch16-repeats', 8], cpu=False)
                    result = read(ROOT / reference['output'] / 'summary.json')
                    if (result['checkpoint_sha256'] != digest(ROOT / job['cfm'] / 'best_cfm.pt')
                            or result['reference_seed'] != reference['seed']):
                        raise RuntimeError('Fresh-reference evaluation identity mismatch')
            states[label] = dict(state='completed', samples_sha256=sample_hash,
                                 checkpoint_sha256=digest(ROOT / job['checkpoint']), job=job)
            save()
        except BaseException as exc:
            if not args.dry_run:
                states[label] = dict(state='stopped' if stopping else 'failed', error=repr(exc))
                save()
            raise


if __name__ == '__main__':
    main()
