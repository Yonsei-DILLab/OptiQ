"""Finite five-seed iBOLT UTD=1 campaign, four GPU lanes, no retries."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from .guards import ROOT, assert_output_boundary, assert_work_branch, code_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maze', choices=('medium', 'hard'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    branch = assert_work_branch()
    assert_output_boundary(args.output)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    jobs = []
    for seed in range(5):
        destination = output / f'{args.maze}-ibolt-T5-utd1-b256-s{seed}'
        command = [sys.executable, '-u', '-m', 'pointmaze_rebuttal.run',
                   '--method', 'ibolt', '--maze', args.maze, '--seed', str(seed),
                   '--temperature', '5', '--num-envs', '256', '--batch-size', '256',
                   '--updates-per-collect', '256', '--output', str(destination)]
        jobs.append(dict(seed=seed, gpu=seed % 4, status='queued', command=command,
                         output=str(destination)))
    manifest = dict(branch=branch, maze=args.maze, utd=1, batch_size=256,
                    num_envs=256, updates_per_collect=256, seeds=list(range(5)),
                    source_hashes=code_hashes(), jobs=jobs,
                    started_unix=time.time(), pid=os.getpid())
    lock = threading.Lock()

    def save():
        temporary = output / 'campaign.json.tmp'
        temporary.write_text(json.dumps(manifest, indent=2) + '\n')
        temporary.replace(output / 'campaign.json')

    def lane(gpu):
        for job in (j for j in jobs if j['gpu'] == gpu):
            assert_work_branch()
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                       XLA_PYTHON_CLIENT_PREALLOCATE='false', MUJOCO_GL='egl',
                       WANDB_MODE='offline', OMP_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            with (output / f'seed-{job["seed"]}.log').open('x') as log:
                process = subprocess.Popen(job['command'], cwd=ROOT, env=env,
                                           stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT)
                with lock:
                    job.update(status='running', pid=process.pid, started_unix=time.time())
                    save()
                code = process.wait()
            with lock:
                complete = Path(job['output'], 'completed.json').exists()
                job.update(status='completed' if code == 0 and complete else 'failed',
                           returncode=code, finished_unix=time.time())
                save()
            print(json.dumps(job), flush=True)

    save()
    with ThreadPoolExecutor(max_workers=4) as workers:
        list(workers.map(lane, range(4)))
    manifest['finished_unix'] = time.time()
    save()
    if any(job['status'] != 'completed' for job in jobs):
        raise SystemExit('Campaign contains failed runs; see campaign.json and seed logs')


if __name__ == '__main__':
    main()
