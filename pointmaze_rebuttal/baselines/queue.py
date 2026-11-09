"""Finite host-local method barriers, after the existing five iBOLT seeds."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import subprocess
import threading
import time

from ..guards import assert_work_branch, assert_output_boundary, code_hashes
from .profiles import METHODS, configuration


def atomic_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def run_logged_command(command, source, env, log_path, on_start):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('x') as log:
        child = subprocess.Popen(command, cwd=source, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT)
        on_start(child.pid)
        return child.wait()


def make_plan(maze, output, profile='matched'):
    if maze not in ('medium', 'hard'):
        raise ValueError(maze)
    jobs = []
    for method in METHODS:
        for seed in range(5):
            config = configuration(method, maze, seed, profile=profile)
            name = f'{maze}-{method}-s{seed}'
            jobs.append(dict(name=name, method=method, maze=maze, seed=seed,
                gpu=seed % 4, status='queued', output=str(output/'runs'/name),
                preflight=str(output/'preflights'/name), settings=config))
    return dict(maze=maze, method_order=list(METHODS), seeds=list(range(5)),
                profile=profile, status='waiting_for_ibolt', jobs=jobs)


def job_commands(job, python, profile):
    c = job['settings']
    common = [python, '-u', '-m', 'pointmaze_rebuttal.baselines.run',
        '--method', job['method'], '--maze', job['maze'], '--seed', str(job['seed']),
        '--profile', profile]
    for key in ('num_envs', 'batch_size', 'updates_per_collect', 'warmup', 'replay_capacity'):
        common += ['--'+key.replace('_', '-'), str(c[key])]
    short = c['warmup'] + c['num_envs']
    preflight = common + ['--steps', str(short), '--eval-every', str(short),
        '--eval-episodes', '5', '--final-episodes', '5', '--output', job['preflight']]
    main = common + ['--steps', str(c['steps']), '--eval-every', str(c['eval_every']),
        '--eval-episodes', str(c['eval_episodes']), '--final-episodes', str(c['final_episodes']),
        '--output', job['output']]
    return preflight, main


def execute_stages(plan, execute, persist, lock=None):
    """Run four GPU lanes per method; a failure blocks all pending methods."""
    lock = lock or threading.RLock()
    aborted = threading.Event()
    for method in METHODS:
        plan.update(status='running', current_method=method)
        persist()
        def lane(gpu):
            for job in (j for j in plan['jobs'] if j['method'] == method and j['gpu'] == gpu):
                if aborted.is_set():
                    return
                with lock:
                    job.update(status='waiting_for_gpu', started_unix=time.time())
                    persist()
                try:
                    execute(job)
                    with lock:
                        job.update(status='completed', finished_unix=time.time())
                        persist()
                except BaseException as error:
                    aborted.set()
                    with lock:
                        plan['abort_requested'] = True
                        job.update(status='failed', error=repr(error), finished_unix=time.time())
                        persist()
                    return
        with ThreadPoolExecutor(max_workers=4) as workers:
            list(workers.map(lane, range(4)))
        if aborted.is_set():
            plan.update(status='blocked_by_failure')
            persist()
            raise RuntimeError('A baseline failed; pending work is blocked, without automatic retry')
    plan.update(status='completed', finished_unix=time.time())
    persist()


def dependency_status(path):
    data = json.loads(path.read_text())
    jobs = data['jobs']
    if sorted(j['seed'] for j in jobs) != list(range(5)):
        raise RuntimeError('iBOLT dependency must contain exactly seeds 0..4')
    for job in jobs:
        if job['status'] not in ('queued', 'running', 'completed'):
            raise RuntimeError('iBOLT dependency failed or stopped: ' + str(job['seed']))
    if not all(j['status'] == 'completed' for j in jobs):
        return False
    for job in jobs:
        result = json.loads((Path(job['output'])/'completed.json').read_text())
        if not result['completed'] or result['updates'] != 992000:
            raise RuntimeError('iBOLT dependency result failed accounting validation')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    assert_work_branch()
    plan_path = args.plan.resolve()
    assert_output_boundary(plan_path)
    plan = json.loads(plan_path.read_text())
    output = plan_path.parent
    for name in ('logs', 'preflights', 'runs'):
        (output/name).mkdir(parents=True, exist_ok=True)
    lock_file = (output/'controller.lock').open('a')
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_lock = threading.RLock()
    def persist():
        with state_lock:
            atomic_json(plan_path, plan)
    try:
        while not dependency_status(Path(plan['dependency'])):
            plan.update(status='waiting_for_ibolt', checked_unix=time.time())
            persist()
            time.sleep(15)
        ready = json.loads((output/'ready.json').read_text())
        if not ready['all_cpu_smokes_passed']:
            raise RuntimeError('Baseline implementations are not validated')
        if ready['source_hashes'] != code_hashes():
            raise RuntimeError('Frozen baseline source changed after validation')
        source = Path(plan['source'])
        env_base = dict(os.environ, PYTHONPATH=str(source),
            XLA_PYTHON_CLIENT_PREALLOCATE='false', MUJOCO_GL='egl', WANDB_MODE='offline',
            OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')

        def execute(job):
            assert_work_branch()
            if ready['source_hashes'] != code_hashes():
                raise RuntimeError('Frozen baseline source changed')
            gpu = job['gpu']
            gpu_lock = (Path(plan['workspace'])/'outputs'/f'.pointmaze-gpu-{gpu}.lock').open('a')
            with gpu_lock:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX)
                while True:
                    if plan.get('abort_requested'):
                        raise RuntimeError('Pending job blocked by another failure')
                    row = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
                        '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
                    memory, utilization = map(int, row.strip().split(','))
                    if memory < 200 and utilization == 0:
                        break
                    time.sleep(10)
                env = dict(env_base, CUDA_VISIBLE_DEVICES=str(gpu))
                config = job['settings']
                preflight, main_command = job_commands(job, plan['python'], plan['profile'])
                with state_lock:
                    job.update(preflight_command=preflight, command=main_command)
                    persist()
                for phase, command, destination in (
                    ('preflight', preflight, job['preflight']), ('training', main_command, job['output'])):
                    if plan.get('abort_requested'):
                        raise RuntimeError('Next phase blocked by another failure')
                    with state_lock:
                        job.update(phase=phase, status=phase)
                        persist()
                    log_path = output/'logs'/f'{job["name"]}-{phase}.log'
                    def on_start(pid):
                        with state_lock:
                            job['pid'] = pid
                            persist()
                    code = run_logged_command(command, source, env, log_path, on_start)
                    if code != 0:
                        raise RuntimeError(f'{phase} exited {code}: {log_path}')
                    completed = json.loads((Path(destination)/'completed.json').read_text())
                    expected = config['updates_per_collect'] if phase == 'preflight' else config['expected_updates']
                    if not completed['completed'] or completed['backend'] != 'gpu' or completed['updates'] != expected:
                        raise RuntimeError(f'{phase} did not pass result accounting')
        execute_stages(plan, execute, persist, state_lock)
    except BaseException as error:
        plan.update(status='blocked_by_failure', error=repr(error))
        persist()
        raise


if __name__ == '__main__':
    main()
