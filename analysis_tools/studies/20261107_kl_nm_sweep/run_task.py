"""Instrument the archived experiment without modifying its numerical kernels."""
import argparse
import contextlib
import json
import os
import signal
import time
from pathlib import Path
from common import append, file_sha, now, tasks, verify_source, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--phase', choices=['pilot', 'reuse', 'train'], required=True)
    parser.add_argument('--index', type=int, required=True)
    args = parser.parse_args()
    task = tasks(args.phase)[args.index]
    root = args.root.resolve()
    out = root / 'runtime' / args.phase / task['id']
    out.mkdir(parents=True, exist_ok=True)
    commit = verify_source(root)
    timing_path = out / ('timing_' + os.environ['OPTIQ_ATTEMPT_ID'] + '.json')
    elapsed = {k: 0.0 for k in ['initialization', 'compile', 'training', 'evaluation', 'checkpoint_io']}
    start = time.monotonic()

    @contextlib.contextmanager
    def measured(name):
        t = time.monotonic()
        try:
            yield
        finally:
            elapsed[name] += time.monotonic() - t
            write(timing_path, dict(seconds=elapsed, elapsed=time.monotonic()-start, utc=now()))

    with measured('initialization'):
        import flax.serialization
        import hashlib
        import jax
        import numpy as np
        from experiments.kl_six_highL_1d.core import implementation
        from experiments.kl_six_highL_1d.evaluate import evaluate
        assert jax.default_backend() == 'gpu'
        exp = implementation(task['config'])(task['config'], task['method'], task['L'], task['seed'])
        initial = hashlib.sha256(flax.serialization.to_bytes(exp.state.params)).hexdigest()
        audit = json.loads((root / 'REUSE_AUDIT.json').read_text())
        origin = next(x for x in audit['entries'] if x['target'] == task['target']
                      and x['method'] == task['method'] and x['seed'] == task['seed'])
        assert initial == origin['initial_parameter_sha256'], (initial, origin)
        meta = dict(task, source_commit=commit, initial_parameter_sha256=initial,
                    attempt_id=os.environ['OPTIQ_ATTEMPT_ID'], utc=now(),
                    starting_step=0, original_training_commit=origin['source_commit'] if args.phase == 'reuse' else commit)
        if (out / 'RUN.json').exists():
            old = json.loads((out / 'RUN.json').read_text())
            assert old['source_commit'] == commit and old['config_sha256'] == task['config_sha256']
            if args.phase != 'pilot' and (out / 'checkpoint.msgpack').exists():
                exp.restore(out / 'checkpoint.msgpack')
            meta['starting_step'] = int(exp.state.step)
            append(out / 'RESUME.jsonl', meta)
        else:
            write(out / 'RUN.json', meta)

    def save():
        with measured('checkpoint_io'):
            tmp = out / 'checkpoint.tmp'
            exp.save(tmp)
            tmp.replace(out / 'checkpoint.msgpack')

    def evaluation(step):
        before = np.asarray(exp.key).copy()
        with measured('evaluation'):
            metric = evaluate(exp, out, step)
            write(out / f'metrics_{step:06d}.json', metric)
        assert np.array_equal(before, np.asarray(exp.key)), 'Evaluation changed training RNG'
        return metric

    if args.phase == 'reuse':
        ck = root / 'reused' / task['target'] / f'{task["method"]}_s{task["seed"]}' / 'checkpoint.msgpack'
        assert file_sha(ck) == origin['checkpoint_sha256']
        exp.restore(ck)
        assert int(exp.state.step) == 100000
        metric = evaluation(100000)
        write(out / 'COMPLETE.json', dict(step=100000, reuse_validated=True, origin=origin,
              evaluation_commit=commit, sample_count=metric['sample_count'], utc=now()))
        return

    stopped = []
    for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGUSR1]:
        signal.signal(sig, lambda s, frame: stopped.append(s))
    block = task['config']['train_block']
    with measured('compile'):
        exp.advance_fn.lower(exp.state, exp.key, block).compile()
    final = (4 * block if args.phase == 'pilot' else 100000)
    try:
        while True:
            step = int(exp.state.step)
            write(out / 'STATUS.json', dict(step=step, phase='training', utc=now(), source_commit=commit))
            if args.phase == 'train' and step in task['config']['eval_steps'] and not (out / f'metrics_{step:06d}.json').exists():
                evaluation(step)
            if step % task['config']['checkpoint_interval'] == 0 or step == final or stopped:
                save()
            if stopped:
                write(out / 'STATUS.json', dict(step=step, phase='paused', utc=now()))
                raise SystemExit(75)
            if step >= final:
                break
            with measured('training'):
                t = time.monotonic()
                info = exp.advance(block)  # archived implementation synchronizes here
                seconds = time.monotonic() - t
            assert all(np.isfinite(v) for v in info.values()), info
            append(out / 'training.jsonl', dict(step=int(exp.state.step), seconds=seconds, **info))
        if args.phase == 'pilot':
            # Checkpoint contents and RNG are exact. Repeated library-GEMM RKL
            # steps on this GPU show float32 variation even without restoration.
            before = jax.tree_util.tree_map(lambda x: np.array(x, copy=True), (exp.state, exp.key))
            with measured('training'):
                exp.advance(block)
                expected = jax.tree_util.tree_map(lambda x: np.array(x, copy=True), (exp.state, exp.key))
                exp.restore(out / 'checkpoint.msgpack')
                for x, y in zip(jax.tree_util.tree_leaves(before), jax.tree_util.tree_leaves((exp.state, exp.key))):
                    assert np.array_equal(x, np.asarray(y)), 'Checkpoint changed saved contents'
                exp.advance(block)
                differences = []
                pairs = zip(jax.tree_util.tree_flatten_with_path(expected)[0],
                            jax.tree_util.tree_flatten_with_path((exp.state, exp.key))[0])
                for (path, x), (_, y) in pairs:
                    y = np.asarray(y)
                    exact = np.array_equal(x, y)
                    close = exact if np.issubdtype(x.dtype, np.integer) else np.allclose(x, y, rtol=1e-6, atol=1e-7)
                    differences.append(dict(path=str(path), exact=bool(exact), passed=bool(close),
                        max_abs=float(np.max(np.abs(x.astype(float)-y.astype(float))))))
                write(out/'RESTORE_VALIDATION.json', dict(checkpoint_contents_exact=True,
                      rtol=1e-6, atol=1e-7, leaves=differences))
                assert all(x['passed'] for x in differences), 'Resume differs beyond float32 tolerance'
            exp.restore(out / 'checkpoint.msgpack')
            before = np.asarray(exp.key).copy()
            with measured('evaluation'):
                samples = exp.samples(32768)
                assert all(np.isfinite(x).all() for x in samples)
            assert np.array_equal(before, np.asarray(exp.key))
        write(out / 'COMPLETE.json', dict(step=final, source_commit=commit, phase=args.phase,
              initial_parameter_sha256=initial, checkpoint_restore_passed=args.phase == 'pilot', utc=now()))
        write(out / 'STATUS.json', dict(step=final, phase='complete', utc=now()))
    except BaseException as error:
        save()
        write(out / 'STATUS.json', dict(step=int(exp.state.step), phase='paused' if stopped else 'failed', error=repr(error), utc=now()))
        raise


if __name__ == '__main__':
    main()
