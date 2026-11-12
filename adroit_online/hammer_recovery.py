"""Explicit one-shot recovery of the authorized Hammer seed4 T0.10 attempt.

The failed attempt is immutable. Diagnostics use its checkpoint with newly
collected native transitions, never masquerading as an exact training resume.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

from .config import configuration, run_id
from .controller import gpu_state
from .priority_controller import alive, choose_slot, process_ticks, verify_adopted
from .run import ROOT, atomic, source_provenance

NAME = 'hammer-ibolt-s4-e256-u025-t010'
ATTEMPT = NAME + '-retry1'
CHECKPOINT = 1_003_520
XLA_FLAGS = '--xla_gpu_enable_triton_gemm=false'


def diagnostic(failed, output, blocks, probe=False):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import torch
    from stable_baselines3.common.logger import configure
    from .env import make_env
    from .ibolt import checkpoint
    from .run import create_model, evaluate

    proof = source_provenance()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert jax.default_backend() == 'gpu'
    c = configuration('ibolt', 'hammer', 4, 'e256-u025', temperature=.10)
    # Diagnostic storage only; the full retry uses the original one-million replay.
    c['replay_capacity'] = 4096
    env = make_env('hammer')
    model = create_model(c, env, output)
    model.set_logger(configure(None, []))
    try:
        model.load_model(failed / 'checkpoints', CHECKPOINT, CHECKPOINT)
        aux = pickle.loads((failed / 'checkpoints' /
                            f'auxiliary-state-{CHECKPOINT:09d}.pkl').read_bytes())
        for obj, attr, key in [(model, 'key', 'learner_key'),
                               (model.policy, 'key', 'policy_key'),
                               (model.policy, 'noise_key', 'policy_noise_key'),
                               (model, 'regulator_key', 'regulator_key')]:
            old = getattr(obj, attr)
            value = jnp.asarray(aux[key])
            if jax.dtypes.issubdtype(old.dtype, jax.dtypes.prng_key):
                value = jax.random.wrap_key_data(value)
            setattr(obj, attr, value)
        model.regulator_log_alpha = jnp.asarray(aux['regulator_log_alpha'])
        model.regulator_state = jax.tree_util.tree_map(jnp.asarray, aux['regulator_optimizer'])
        model.regulator_rng.bit_generator.state = aux['regulator_numpy']
        model.regulator_next_update = aux['regulator_next_update']
        model.regulator_entropy = aux['regulator_entropy']
        model.num_timesteps, model._n_updates = aux['steps'], aux['updates']
        generator = np.random.default_rng(13579)
        obs, _ = env.reset(seed=24680)
        for _ in range(2048):
            action = generator.uniform(-1, 1, env.action_space.shape).astype(np.float32)
            next_obs, reward, terminal, timeout, info = env.step(action)
            info = dict(info, **{'TimeLimit.truncated': timeout and not terminal})
            model.replay_buffer.add(obs[None], next_obs[None], action[None],
                                   np.array([reward]), np.array([terminal or timeout]), [info])
            obs = next_obs
            if terminal or timeout:
                obs, _ = env.reset(seed=int(generator.integers(2**30)))
        np.random.seed(24680)
        started = time.time()
        if probe:
            model.train(batch_size=256, gradient_steps=1)
            leaves = jax.tree_util.tree_leaves((model.policy.actor_state,
                                               model.policy.qf_state, model.key))
            def host_array(x):
                if hasattr(x, 'dtype') and jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key):
                    x = jax.random.key_data(x)
                return np.asarray(x)
            np.savez(output / 'one-update.npz', **{f'leaf{i}': host_array(x)
                                                  for i, x in enumerate(leaves)})
            atomic(output / 'passed.json', dict(passed=True, diagnostic_only=True,
                source_commit=proof['commit'], additional_updates=1,
                xla_flags=os.environ.get('XLA_FLAGS', ''), leaves=len(leaves)))
            return
        for block in range(blocks):
            model.num_timesteps += 256
            model.train(batch_size=256, gradient_steps=64)
            if (block + 1) % 32 == 0 or block + 1 == blocks:
                checkpoint(model, output / 'checkpoints', verify=True)
                print(json.dumps(dict(block=block + 1, updates=model._n_updates)), flush=True)
        report = evaluate(model, c, 20)
        assert model._n_updates == aux['updates'] + blocks * 64
        atomic(output / 'passed.json', dict(passed=True, diagnostic_only=True,
            source_commit=proof['commit'], failed_source_commit=json.loads(
                (failed / 'spec.json').read_text())['source_commit'],
            checkpoint=CHECKPOINT, additional_updates=blocks * 64,
            native_transitions=2048, checkpoint_restore_exact=True,
            finite_states=True, evaluation_episodes=report['episodes'],
            seconds=time.time() - started, finished_unix=time.time()))
    finally:
        env.close()


def recovery(root):
    proof = source_provenance()
    lock = (root / 'hammer-recovery.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_file = root / 'recovery-queue-state.json'
    previous_diagnostic = None
    if state_file.exists():
        previous_diagnostic = json.loads(state_file.read_text())
        previous = previous_diagnostic['jobs'][NAME]
        assert previous['status'] == 'failed' and previous['phase'] == 'memcheck'
        assert previous['source_commit'] == 'b15bb764cdaaadc4a605c5d35070cc73c2171fed'
        assert not alive(previous) and not (root / 'runs' / ATTEMPT).exists()
        history = root / 'recovery-initial-diagnostic.json'
        assert not history.exists()
        atomic(history, previous_diagnostic)
    q = json.loads((root / 'priority-queue-state.json').read_text())
    old = q['jobs'][NAME]
    assert old['status'] == 'failed' and old['returncode'] == -6 and not alive(old)
    failed = root / 'runs' / NAME
    assert not (failed / 'completed.json').exists()
    assert (failed / 'checkpoints' / f'actor_state_{CHECKPOINT}.msgpack').exists()
    output = root / 'runs' / ATTEMPT
    assert not output.exists()
    peers = []
    for name, state in q['jobs'].items():
        if state['status'] in ('running', 'preflight') and alive(state):
            if state['status'] == 'running':
                verify_adopted(root, name, state, root / ('source-' + state['source_commit']))
            peers.append(state)
    config = configuration('ibolt', 'hammer', 4, 'e256-u025', temperature=.10)
    spec = json.loads((failed / 'spec.json').read_text())
    assert all(spec[k] == v for k, v in config.items())
    slot = choose_slot(config, peers, gpu_state(), sorted(os.sched_getaffinity(0)), 1)
    assert slot is not None, 'GPU1 recovery slot is occupied'
    gpu, cpus = slot
    record = dict(logical_run_id=NAME, attempt_id=ATTEMPT, output=str(output),
                  original_output=str(failed), original_source_commit=spec['source_commit'],
                  source_commit=proof['commit'], gpu=gpu, cpus=cpus,
                  status='diagnostic', fresh_from_scratch=True, automatic_retry=False,
                  controller_pid=os.getpid(), started_unix=time.time(),
                  runtime_environment={'XLA_FLAGS': XLA_FLAGS},
                  previous_diagnostic=previous_diagnostic,
                  preserved_peers={s['pid']: s['start_ticks'] for s in peers})
    def save():
        atomic(state_file, dict(source_commit=proof['commit'], jobs={NAME: record},
                               updated_unix=time.time()))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), JAX_PLATFORMS='cuda',
               XLA_PYTHON_CLIENT_PREALLOCATE='false', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               MUJOCO_GL='egl', PYTHONUNBUFFERED='1', JAX_TRACEBACK_FILTERING='off',
               PYTHONFAULTHANDLER='1')
    # Isolate the reproduced GEMM autotuner crash without changing shared wheels,
    # drivers or the numerical algorithm. Each diagnostic gets a fresh context.
    prefix = ['taskset', '-c', ','.join(map(str, cpus))]
    cases = [('probe-original', 0), ('probe-cublas', 0),
             ('memcheck-cublas', 4), ('stress-cublas', 128)]
    save()
    for label, blocks in cases:
        dest = root / 'preflight' / 'hammer-recovery' / label
        cmd = [*prefix]
        if label == 'memcheck-cublas':
            cmd += ['/usr/local/cuda/bin/compute-sanitizer', '--tool', 'memcheck',
                    '--error-exitcode', '86', '--target-processes', 'application-only']
        cmd += [sys.executable, '-m', 'adroit_online.hammer_recovery', '--diagnostic',
                '--root', str(root), '--output', str(dest), '--blocks', str(blocks)]
        if label.startswith('probe-'):
            cmd += ['--probe']
        child_env = dict(env, XLA_FLAGS='' if label == 'probe-original' else XLA_FLAGS)
        with (root / 'process-logs' / ('hammer-recovery-' + label + '.log')).open('x') as log:
            p = subprocess.Popen(cmd, cwd=ROOT, env=child_env, stdout=log, stderr=subprocess.STDOUT)
            record.update(phase=label, pid=p.pid, start_ticks=process_ticks(p.pid)); save()
            code = p.wait()
        if code or not (dest / 'passed.json').exists():
            record.update(status='failed', returncode=code, finished_unix=time.time()); save()
            raise RuntimeError(('Recovery diagnostic failed', label, code))
        if label == 'probe-cublas':
            import numpy as np
            folder = root / 'preflight' / 'hammer-recovery'
            with np.load(folder / 'probe-original' / 'one-update.npz') as a, np.load(
                    folder / 'probe-cublas' / 'one-update.npz') as b:
                assert a.files == b.files
                errors = {}
                for key in a.files:
                    np.testing.assert_allclose(a[key], b[key], rtol=1e-5, atol=1e-6,
                                               err_msg=key)
                    errors[key] = float(np.max(np.abs(a[key].astype(float) - b[key].astype(float))))
            record['numerical_comparison'] = dict(passed=True, leaves=len(errors),
                max_absolute_difference=max(errors.values()), rtol=1e-5, atol=1e-6)
            atomic(folder / 'numerical-comparison.json', record['numerical_comparison'])
    record['diagnostics_passed'] = [label for label, _ in cases]
    cmd = [*prefix, sys.executable, '-m', 'adroit_online.run', '--method', 'ibolt',
           '--task', 'hammer', '--seed', '4', '--profile', 'e256-u025',
           '--temperature', '0.10', '--output', str(output)]
    with (root / 'process-logs' / (ATTEMPT + '.log')).open('x') as log:
        p = subprocess.Popen(cmd, cwd=ROOT, env=dict(env, XLA_FLAGS=XLA_FLAGS),
                             stdout=log, stderr=subprocess.STDOUT)
        record.update(status='running', phase='full', pid=p.pid,
                      start_ticks=process_ticks(p.pid), command=cmd, full_started_unix=time.time())
        save()
        code = p.wait()
    record.update(status='completed' if code == 0 and (output / 'completed.json').exists()
                  else 'failed', returncode=code, finished_unix=time.time())
    save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--diagnostic', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--blocks', type=int, default=128)
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    if args.diagnostic:
        diagnostic(args.root / 'runs' / NAME, args.output, args.blocks, args.probe)
    else:
        recovery(args.root)
