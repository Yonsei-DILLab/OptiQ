"""Actual GPU operations, independent SAC numerical checks and native-task smoke runs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .config import configuration
from .run import ROOT, atomic, source_provenance


def gpu_worker(task, output):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import torch
    from .env import make_env
    from .run import create_model, evaluate
    from . import numerical
    assert jax.default_backend() == 'gpu' and torch.cuda.is_available()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    x = torch.ones((512, 512), device='cuda')
    assert (x @ x)[0, 0].item() == 512
    y = jnp.ones((512, 512))
    assert float((y @ y).block_until_ready()[0, 0]) == 512
    output.mkdir(parents=True, exist_ok=False)
    env = make_env(task)
    paired = make_env(task)
    try:
        a, _ = env.reset(seed=901)
        b, _ = paired.reset(seed=901)
        assert np.array_equal(a, b)
        rng = np.random.default_rng(710)
        for t in range(200):
            action = rng.uniform(-1, 1, env.action_space.shape).astype(np.float32)
            a, b = env.step(action), paired.step(action)
            assert np.array_equal(a[0], b[0]) and a[1:4] == b[1:4]
            assert a[1] == (10. if a[4]['success'] else -.1)
            assert not a[2] and a[3] == (t == 199)
        shape = dict(observation=env.observation_space.shape, action=env.action_space.shape)
    finally:
        env.close(); paired.close()
    # Reuse the independent golden-equation checker with native observation/action
    # shapes and the ordinary MlpPolicy (LayerNorm off). Its synthetic reward set
    # tests equations; actual +/-10/-0.1 rewards are covered by the native smoke.
    report = numerical.run(task, output)
    report.update(task=task, actual_shapes=shape, layer_norm=False)
    # Verify the evaluator leaves all learner RNG streams and empty replay intact.
    c = configuration('sac', task, 4, 'e256-u025', smoke=True)
    env = make_env(task)
    try:
        model = create_model(c, env)
        key = np.asarray(model.jax_state.key).copy()
        np_state = np.random.get_state()
        torch_state = torch.get_rng_state().clone()
        cuda_state = torch.cuda.get_rng_state().clone()
        evaluate(model, c, 20)
        assert np.array_equal(key, np.asarray(model.jax_state.key))
        after = np.random.get_state()
        assert np_state[0] == after[0] and np.array_equal(np_state[1], after[1]) and np_state[2:] == after[2:]
        assert torch.equal(torch_state, torch.get_rng_state())
        assert torch.equal(cuda_state, torch.cuda.get_rng_state())
        assert model.replay_buffer.size() == 0
    finally:
        env.close()
    report.update(evaluation_rng_preserved=True, replay_empty_after_evaluation=True)
    atomic(output / 'numeric-and-env.json', report)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--worker-task', choices=('door', 'hammer', 'pen', 'relocate'))
    a = p.parse_args()
    proof = source_provenance()
    if a.worker_task:
        gpu_worker(a.worker_task, a.root)
        return
    a.root.mkdir(parents=True, exist_ok=True)
    attempt = a.root / time.strftime('attempt-%Y%m%dT%H%M%S', time.gmtime())
    attempt.mkdir(exist_ok=False)
    cpus = sorted(os.sched_getaffinity(0))
    processes = []
    for gpu, task in enumerate(('door', 'hammer', 'pen', 'relocate')):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), JAX_PLATFORMS='cuda',
                   XLA_PYTHON_CLIENT_PREALLOCATE='false', OMP_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                   MUJOCO_GL='egl', PYTHONUNBUFFERED='1')
        log = (attempt / f'gpu{gpu}.log').open('w')
        cmd = ['taskset', '-c', ','.join(map(str, cpus[gpu * 8:(gpu + 1) * 8])),
               sys.executable, '-m', 'adroit_online.preflight', '--root', str(attempt / task),
               '--worker-task', task]
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        processes.append((gpu, task, env, proc, log))
    for gpu, task, env, proc, log in processes:
        code = proc.wait(); log.close()
        assert code == 0, (gpu, task, 'numeric validation failed', attempt)
    # Each GPU checks one task and both actual training paths, concurrently across GPUs.
    for method in ('sac', 'td3'):
        children = []
        for gpu, task, env, _, _ in processes:
            log = (attempt / f'{task}-{method}-smoke.log').open('w')
            cmd = ['taskset', '-c', ','.join(map(str, cpus[gpu * 8:(gpu + 1) * 8])),
                   sys.executable, '-m', 'adroit_online.run', '--method', method,
                   '--task', task, '--seed', '4', '--profile', 'e256-u025', '--smoke',
                   '--output', str(attempt / f'{task}-{method}-smoke')]
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            children.append((task, proc, log))
        for task, proc, log in children:
            code = proc.wait(); log.close()
            assert code == 0, (task, method, 'smoke failed', attempt)
    # Exercise both additional collection/update shapes on a real task before release.
    children = []
    for gpu, (method, profile) in enumerate((('sac', 'e256-u1'), ('td3', 'e256-u1'),
                                           ('sac', 'e64-u1'), ('td3', 'e64-u1'))):
        env = processes[gpu][2]
        name = f'pen-{method}-{profile}-smoke'
        log = (attempt / (name + '.log')).open('w')
        cmd = ['taskset', '-c', ','.join(map(str, cpus[gpu * 8:(gpu + 1) * 8])),
               sys.executable, '-m', 'adroit_online.run', '--method', method, '--task', 'pen',
               '--seed', '5', '--profile', profile, '--smoke', '--output', str(attempt / name)]
        children.append((name, subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT), log))
    for name, proc, log in children:
        code = proc.wait(); log.close()
        assert code == 0, (name, 'profile smoke failed', attempt)
    atomic(a.root / 'passed.json', dict(passed=True, source_commit=proof['commit'],
           gpus=[0, 1, 2, 3], attempt=str(attempt), numerical_tasks=4, smoke_runs=12,
           finished_unix=time.time()))


if __name__ == '__main__':
    main()
