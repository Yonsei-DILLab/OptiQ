"""One exclusive GPU worker; dynamic file-locked queue and attempt accounting."""
import argparse
import fcntl
import importlib.metadata
import json
import os
import platform
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from common import SOURCE, append, now, tasks, verify_source, write


def gpu_info(gpu):
    fields = 'uuid,name,memory.total,utilization.gpu,memory.used,power.draw'
    p = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-gpu=' + fields,
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15)
    if p.returncode:
        return dict(error=p.stderr[-500:])
    row = [x.strip() for x in p.stdout.strip().split(',')]
    def number(x):
        try:
            return float(x)
        except ValueError:
            return None
    return dict(uuid=row[0], model=row[1], memory_total_mib=number(row[2]),
                utilization_percent=number(row[3]), memory_used_mib=number(row[4]), power_w=number(row[5]))


def attempt(root, task, index, gpu, commit):
    attempt_id = f'{int(time.time())}_{uuid.uuid4().hex[:10]}'
    out = root / 'runtime' / task['phase'] / task['id']
    out.mkdir(parents=True, exist_ok=True)
    attempt_dir = out / 'attempts' / attempt_id
    attempt_dir.mkdir(parents=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OPTIQ_ATTEMPT_ID=attempt_id,
               PYTHONUNBUFFERED='1', JAX_THREEFRY_PARTITIONABLE='false',
               XLA_PYTHON_CLIENT_PREALLOCATE='false', OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', JAX_PLATFORMS='cuda',
               PYTHONNOUSERSITE='1')
    for name in ['LD_LIBRARY_PATH', 'PYTHONPATH', 'JAX_DEFAULT_MATMUL_PRECISION']:
        env.pop(name, None)
    # Same library GEMM / rolled density scan settings as the archived Vast run.
    env['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false --xla_gpu_enable_while_loop_unrolling=WHILE_LOOP_UNROLLING_NO_UNROLL'
    metadata = {k: v for k, v in task.items() if k != 'config'}
    metadata.update(attempt_id=attempt_id, job_id=f'kl_nm_gpu{gpu}:{attempt_id}',
                    source_commit=commit, hostname=platform.node(), gpu_index=gpu,
                    gpu_count=1, allocation_share=1, B=task['config']['batch'],
                    gpu=gpu_info(gpu), start_utc=now(), start_epoch=time.time(),
                    packages={k: importlib.metadata.version(k) for k in ['jax','jaxlib','flax','optax','numpy','scipy']},
                    python=sys.version, environment={k:env[k] for k in ['JAX_THREEFRY_PARTITIONABLE','XLA_FLAGS','XLA_PYTHON_CLIENT_PREALLOCATE']},
                    timing_status='running', queue_seconds=None)
    start = time.monotonic()
    write(attempt_dir / 'BEGIN.json', metadata)
    append(root / f'compute_attempts_gpu{gpu}.jsonl', dict(event='begin', **metadata))
    power_points = []
    peak = 0.0
    util_sum = 0.0
    util_n = 0
    samples = 0
    stopping = []
    with (attempt_dir / 'process.log').open('a') as log:
        process = subprocess.Popen([sys.executable, str(SOURCE/'run_task.py'), '--root', str(root),
                                   '--phase', task['phase'], '--index', str(index)], env=env,
                                   cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        def stop(sig, frame):
            stopping.append(sig)
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        old_handlers = {s: signal.signal(s, stop) for s in [signal.SIGTERM, signal.SIGINT]}
        while True:
            point = gpu_info(gpu)
            delta = time.monotonic() - start
            append(attempt_dir / 'telemetry.jsonl', dict(elapsed_seconds=delta, utc=now(), **point))
            samples += 1
            if point.get('memory_used_mib') is not None:
                peak = max(peak, point['memory_used_mib'])
            if point.get('power_w') is not None:
                power_points.append((delta, point['power_w']))
            if point.get('utilization_percent') is not None:
                util_sum += point['utilization_percent']; util_n += 1
            progress = dict(metadata, last_heartbeat_utc=now(), elapsed_seconds=delta,
                            allocated_gpu_hours=delta/3600, peak_memory_mib=peak, process_pid=process.pid)
            write(attempt_dir / 'CURRENT.json', progress)
            if process.poll() is not None:
                break
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        for s, handler in old_handlers.items():
            signal.signal(s, handler)
    total = time.monotonic() - start
    timing = out / f'timing_{attempt_id}.json'
    measured = json.loads(timing.read_text())['seconds'] if timing.exists() else {}
    energy = sum((b[0]-a[0])*(a[1]+b[1])/2 for a,b in zip(power_points,power_points[1:])) / 3_600_000
    result = dict(metadata, end_utc=now(), end_epoch=time.time(), exit_code=process.returncode,
                  elapsed_seconds=total, allocated_gpu_hours=total/3600,
                  synchronized_training_gpu_hours=measured.get('training',0)/3600,
                  timing_seconds=measured, timing_status='finished',
                  energy_kwh_estimate=energy, energy_method='trapezoid sampled whole-device power',
                  energy_covered_seconds=power_points[-1][0]-power_points[0][0] if len(power_points)>1 else 0,
                  telemetry_interval_seconds=5, telemetry_samples=samples, valid_power_samples=len(power_points),
                  peak_memory_mib=peak, mean_utilization_percent=util_sum/util_n if util_n else None)
    status_path = out/'STATUS.json'
    result['final_status'] = json.loads(status_path.read_text()) if status_path.exists() else None
    write(attempt_dir/'END.json', result)
    append(root/f'compute_attempts_gpu{gpu}.jsonl', dict(event='end', **result))
    summaries = [json.loads(p.read_text()) for p in (out/'attempts').glob('*/END.json')]
    write(out/'compute_summary.json', dict(attempts=len(summaries),
          allocated_gpu_hours=sum(x['allocated_gpu_hours'] for x in summaries),
          synchronized_training_gpu_hours=sum(x['synchronized_training_gpu_hours'] for x in summaries),
          complete=(out/'COMPLETE.json').exists()))
    print(json.dumps(dict(task=task['id'],phase=task['phase'],gpu=gpu,exit_code=process.returncode,
                          gpu_hours=result['allocated_gpu_hours'])), flush=True)
    if stopping:
        raise SystemExit(75)


def work(root, phase, gpu, commit):
    for index, task in enumerate(tasks(phase)):
        out = root/'runtime'/phase/task['id']
        out.mkdir(parents=True, exist_ok=True)
        with (out/'worker.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            if (out/'COMPLETE.json').exists():
                continue
            failures = [p for p in (out/'attempts').glob('*/END.json')
                        if json.loads(p.read_text())['exit_code'] not in (0,75)]
            if failures:
                continue
            attempt(root, task, index, gpu, commit)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--phase', choices=['preflight','train'], required=True)
    a = p.parse_args(); root = a.root.resolve()
    gpu_lock = (root/f'gpu{a.gpu}.lock').open('a')
    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cpus = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, cpus[a.gpu*4:a.gpu*4+4] or cpus[:4])
    commit = verify_source(root)
    if a.phase == 'train':
        gate = json.loads((root/'VALIDATION.json').read_text())
        assert gate['status'] == 'passed' and gate['source_commit'] == commit
    wid = f'{a.phase}_gpu{a.gpu}_{int(time.time())}'
    allocation = dict(id=wid, gpu_index=a.gpu, gpu_count=1, start_epoch=time.time(), start_utc=now(),
                      source_commit=commit, phase=a.phase, hostname=platform.node())
    write(root/'allocations'/f'{wid}.json', allocation)
    started = time.monotonic()
    try:
        for phase in (['pilot','reuse'] if a.phase == 'preflight' else ['train']):
            work(root,phase,a.gpu,commit)
    finally:
        allocation.update(end_epoch=time.time(), end_utc=now(), allocated_gpu_hours=(time.monotonic()-started)/3600)
        write(root/'allocations'/f'{wid}.json',allocation)


if __name__ == '__main__':
    main()
