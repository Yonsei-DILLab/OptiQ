"""Run five unchanged GMM40 MEOW workers and collect comparable seed metrics."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def reports(output):
    found = {}
    for path in sorted((Path(output) / 'evaluations').glob('step_*/summary.json')):
        record = json.loads(path.read_text())
        found[record['actor_updates']] = record
    return found


def summarize(records):
    keys = set.intersection(*(set(r['full_policy']) for r in records))
    result = {}
    for key in sorted(keys):
        values = [r['full_policy'][key] for r in records]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            if not all(math.isfinite(v) for v in values):
                raise ValueError(f'Nonfinite {key}: {values}')
            result[key] = dict(values=values, mean=statistics.mean(values),
                               sample_std=statistics.stdev(values) if len(values) > 1 else None)
    return result


def collect(root, plan):
    histories = {j['seed']: reports(j['output']) for j in plan['jobs']}
    common = sorted(set.intersection(*(set(h) for h in histories.values())))
    curves = []
    for step in common:
        values = [histories[seed][step] for seed in plan['seeds']]
        curves.append(dict(actor_updates=step, seeds=plan['seeds'], metrics=summarize(values)))
    result = dict(campaign=plan['campaign'], updated_unix=time.time(),
                  expected_seeds=plan['seeds'], aggregation='per-seed mean and sample standard deviation (ddof=1)',
                  evaluation_policy='stochastic_flow', reference_seed=20260917,
                  latest_common_actor_updates=common[-1] if common else None,
                  complete=(len(common) > 0 and common[-1] == 100000 and
                            all((Path(j['output']) / 'completed.json').exists() for j in plan['jobs'])),
                  learning_curve=curves,
                  per_seed=[dict(seed=s, latest=histories[s][max(histories[s])] if histories[s] else None)
                            for s in plan['seeds']])
    save(root / 'metrics-latest.json', result)
    if result['complete']:
        import numpy as np
        for job in plan['jobs']:
            output = Path(job['output'])
            cfg = json.loads((output / 'config.json').read_text())
            done = json.loads((output / 'completed.json').read_text())
            assert done == dict(actor_updates=100000, completed=True)
            assert cfg['baseline'] == 'meow' and cfg['seed'] == job['seed']
            assert cfg['steps'] == 100000 and cfg['batch'] == 256
            assert cfg['eval_samples'] == 10000 and cfg['warmup_transitions'] == 1000
            assert cfg['entropy_coefficient'] == 1. and cfg['learning_rate'] == 3e-4
            assert cfg['runtime']['revision'] == plan['revision']
            assert cfg['evaluation_policy'] == 'stochastic_flow'
            final = histories[job['seed']][100000]
            assert final['Q_evaluations'] == 101000
            samples = np.load(output / 'evaluations/step_0100000/samples_full_policy.npy')
            assert samples.shape == (10000, 2) and np.isfinite(samples).all()
            assert np.max(np.abs(samples)) <= 40.00001
            assert len(list((output / 'checkpoints').glob('step_*.pt'))) == 16
            assert (output / final['checkpoint']).stat().st_size > 0
        result['final'] = curves[-1]
        result['artifact_validation'] = 'All five final checkpoints, 16 checkpoints/seed, samples, budgets and oracle counts verified'
        save(root / 'metrics-final.json', result)
    return result


def verify_source(plan):
    code = Path(plan['code_root'])
    revision = subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != plan['revision']:
        raise RuntimeError('Pinned GMM revision changed')
    for relative, expected in plan['source_sha256'].items():
        actual = hashlib.sha256((code / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f'Pinned source changed: {relative}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--collect-only', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    plan = json.loads((root / 'plan.json').read_text())
    if args.collect_only:
        result = collect(root, plan)
        print(json.dumps(dict(complete=result['complete'], latest_common_actor_updates=result['latest_common_actor_updates'])))
        return
    import fcntl
    handle = (root / 'controller.lock').open('a')
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if any(Path(j['output']).exists() or Path(j['preflight']).exists() for j in plan['jobs']):
        raise RuntimeError('Preserving existing outputs; this campaign never overwrites or restarts workers')
    verify_source(plan)
    plan.update(status='preflight', controller_pid=os.getpid(), started_unix=time.time())
    save(root / 'plan.json', plan)
    lock = threading.Lock()

    def run_job(job, phase):
        output = job['preflight'] if phase == 'preflight' else job['output']
        command = [plan['python'], '-u', '-m', 'gmm40.baselines.run', '--baseline', 'meow',
                   '--seed', str(job['seed']), '--steps', '2' if phase == 'preflight' else '100000',
                   '--batch', '256', '--learning-rate', '0.0003', '--warmup', '1000',
                   '--buffer-size', '1000000', '--eval-samples', '10000', '--threads', '1',
                   '--device', 'cuda', '--output', output]
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(job['gpu']), JAX_PLATFORMS='cpu',
                           XLA_PYTHON_CLIENT_PREALLOCATE='false', OMP_NUM_THREADS='1',
                           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                           WANDB_MODE='offline', MUJOCO_GL='egl', PYTHONUNBUFFERED='1')
        verify_source(plan)
        log_path = root / 'logs' / f"seed-{job['seed']}-{phase}.log"
        launch = ['taskset', '-c', ','.join(map(str, job['cpu_affinity']))] + command
        with log_path.open('x') as log:
            process = subprocess.Popen(launch, cwd=plan['code_root'], env=environment,
                                       stdout=log, stderr=subprocess.STDOUT)
            with lock:
                job.update(status=phase, phase=phase, pid=process.pid,
                           phase_started_unix=time.time(), command=command, log=str(log_path))
                save(root / 'plan.json', plan)
            exit_code = process.wait()
        with lock:
            job['returncode'] = exit_code
            if exit_code == 0:
                done = json.loads((Path(output) / 'completed.json').read_text())
                expected = 2 if phase == 'preflight' else 100000
                if done != dict(actor_updates=expected, completed=True):
                    raise RuntimeError(f"Completion marker mismatch for seed {job['seed']}")
                job['status'] = 'preflight_passed' if phase == 'preflight' else 'completed'
                job[f'{phase}_finished_unix'] = time.time()
            else:
                job.update(status='failed', failed_phase=phase)
            save(root / 'plan.json', plan)
        print(json.dumps(dict(seed=job['seed'], phase=phase, returncode=exit_code)), flush=True)
        return exit_code

    with ThreadPoolExecutor(max_workers=5) as pool:
        for phase in ['preflight', 'training']:
            with lock:
                plan['status'] = phase
                save(root / 'plan.json', plan)
            futures = [pool.submit(run_job, j, phase) for j in plan['jobs']]
            while not all(f.done() for f in futures):
                if phase == 'training':
                    collect(root, plan)
                time.sleep(10)
            failed = []
            for index, future in enumerate(futures):
                try:
                    if future.result() != 0:
                        failed.append(index)
                except Exception as exc:
                    plan['jobs'][index].update(status='failed', error=repr(exc))
                    failed.append(index)
            if failed:
                plan.update(status='failed', finished_unix=time.time())
                save(root / 'plan.json', plan)
                collect(root, plan)
                raise RuntimeError(f'{phase} failed for seed indices {failed}; logs and artifacts preserved')
    result = collect(root, plan)
    assert result['complete']
    plan.update(status='completed', finished_unix=time.time())
    save(root / 'plan.json', plan)
    print(json.dumps(result['final']), flush=True)


if __name__ == '__main__':
    main()
