"""Native Adroit GPU and integration checks for the unchanged iBOLT learner."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .config import IBOLT_TEMPERATURES, TASKS, configuration
from .ibolt_allocation import wait_for_admission
from .run import ROOT, atomic, source_provenance


def worker(task, output, temperature):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import torch
    from stable_baselines3.common.logger import configure
    from .env import make_env
    from .ibolt import rng_state
    from .run import create_model, evaluate, train

    assert jax.default_backend() == 'gpu'
    x = jnp.ones((512, 512))
    assert float((x @ x).block_until_ready()[0, 0]) == 512
    c = configuration('ibolt', task, 4, 'e256-u025', smoke=True, temperature=temperature)
    output.mkdir(parents=True, exist_ok=False)
    env, paired = make_env(task), make_env(task)
    try:
        a, _ = env.reset(seed=901); b, _ = paired.reset(seed=901)
        assert np.array_equal(a, b)
        rng = np.random.default_rng(710)
        for t in range(200):
            action = rng.uniform(-1, 1, env.action_space.shape).astype(np.float32)
            a, b = env.step(action), paired.step(action)
            assert np.array_equal(a[0], b[0]) and a[1:4] == b[1:4]
            assert a[1] == (10. if a[4]['success'] else -.1)
            assert not a[2] and a[3] == (t == 199)
        model = create_model(c, env, output)
        model.set_logger(configure(None, []))
        before = rng_state(model)
        numpy_before = np.random.get_state(); torch_before = torch.get_rng_state().clone()
        report = evaluate(model, c, 20)
        after = rng_state(model)
        for key in before:
            if isinstance(before[key], np.ndarray): assert np.array_equal(before[key], after[key]), key
            else: assert before[key] == after[key], key
        numpy_after = np.random.get_state()
        assert numpy_before[0] == numpy_after[0] and np.array_equal(numpy_before[1], numpy_after[1])
        assert numpy_before[2:] == numpy_after[2:]
        assert torch.equal(torch_before, torch.get_rng_state())
        assert model.replay_buffer.size() == 0
        assert not getattr(model.policy, 'evaluation_mu_only', False)
        atomic(output / 'environment-and-evaluation.json', dict(
            passed=True, task=task, temperature=temperature, jax_devices=[str(d) for d in jax.devices()],
            primary=report['primary'], evaluation_rng_preserved=True,
            empty_standard_replay=True, shapes=dict(observation=env.observation_space.shape,
                                                   action=env.action_space.shape)))
    finally:
        env.close(); paired.close()
    del model
    # Real vector collector, DACER, TD/actor updates, timeout masks, paired
    # evaluation and exact checkpoint serialization on every native task.
    train(c, output / 'smoke')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--task', choices=tuple(TASKS))
    p.add_argument('--temperature', type=float, choices=IBOLT_TEMPERATURES, default=.25)
    p.add_argument('--temperatures', type=float, nargs='+', choices=IBOLT_TEMPERATURES, default=[.25])
    p.add_argument('--peer-root', type=Path)
    a = p.parse_args()
    proof = source_provenance()
    if a.task:
        worker(a.task, a.root, a.temperature)
        return
    a.root.mkdir(parents=True, exist_ok=True)
    attempt = a.root / time.strftime('attempt-%Y%m%dT%H%M%S', time.gmtime())
    attempt.mkdir(exist_ok=False)
    cpus = sorted(os.sched_getaffinity(0)); assert len(cpus) >= 32
    for temperature in a.temperatures:
        wait_for_admission(a.peer_root, attempt / 'gpu-admission.json')
        children = []
        for i, task in enumerate(TASKS):
            name = f't{round(temperature * 100):03d}-{task}'
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % 2), JAX_PLATFORMS='cuda',
                       XLA_PYTHON_CLIENT_PREALLOCATE='false', OMP_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                       MUJOCO_GL='egl', PYTHONUNBUFFERED='1')
            log = (attempt / (name + '.log')).open('w')
            cmd = ['taskset', '-c', ','.join(map(str, cpus[i * 8:(i + 1) * 8])),
                   sys.executable, '-m', 'adroit_online.ibolt_preflight', '--root', str(attempt / name),
                   '--task', task, '--temperature', str(temperature)]
            children.append((name, subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log,
                                                    stderr=subprocess.STDOUT), log))
            time.sleep(5)
        failures = []
        for name, proc, log in children:
            code = proc.wait(); log.close()
            if code: failures.append((name, code))
        assert not failures, (failures, attempt)
    atomic(a.root / 'passed.json', dict(passed=True, source_commit=proof['commit'],
           gpus=[0, 1], attempt=str(attempt), tasks=list(TASKS), temperatures=a.temperatures,
           smoke_runs=4 * len(a.temperatures),
           finished_unix=time.time()))


if __name__ == '__main__':
    main()
