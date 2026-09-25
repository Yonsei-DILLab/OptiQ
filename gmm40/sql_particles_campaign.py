"""Frozen fixed-Q SQL particle ablation on two four-GPU 5090 shards."""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from .campaign import identity, read, verified, write

SOURCE = Path(__file__).resolve().parents[1]
OPS = Path('/home/heechan/OptiQ-ops')
CTL = ['supervisorctl', '-c', str(OPS/'supervisor/supervisord.conf')]


def gpus():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name',
                                   '--format=csv,noheader'], text=True)
    rows = list(csv.reader(raw.splitlines(), skipinitialspace=True))
    assert len(rows) == 4 and all('RTX 5090' in row[2] for row in rows), rows
    mapping = {row[1]: int(row[0]) for row in rows}
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                   '--format=csv,noheader'], text=True)
    busy = {mapping[r[0]] for r in csv.reader(raw.splitlines(), skipinitialspace=True)
            if r and r[0] in mapping}
    return rows, busy


def prepare(root, shard):
    plan = read(SOURCE/'gmm40/sql_particles_plan.json')
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip()
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=SOURCE, text=True).strip()
    assert shard in (0, 1)
    hashes = {}
    for rel in ('gmm40/sql.py', 'gmm40/sql_jax.py', 'gmm40/target.py'):
        old = subprocess.check_output(['git', 'show', f"{plan['reference_commit']}:{rel}"], cwd=SOURCE)
        assert old == (SOURCE/rel).read_bytes(), 'Baseline implementation changed: '+rel
        hashes[rel] = hashlib.sha256(old).hexdigest()
    rows, _ = gpus()
    jobs = []
    for k in plan['particles']:
        for seed in plan['seeds_by_shard'][str(shard)]:
            name = f'sql_k{k}_s{seed}_100k'
            args = ['--method', 'sql', '--name', name, '--seed', str(seed),
                    '--steps', str(plan['steps']), '--batch', str(plan['batch']),
                    '--width', str(plan['width']), '--depth', str(plan['depth']),
                    '--temperature', str(plan['temperature']),
                    '--eval-samples', str(plan['eval_samples']),
                    '--sql-kernel-particles', str(k), '--sql-kernel-update-ratio', str(plan['kernel_update_ratio']),
                    '--sql-value-particles', str(plan['value_particles']),
                    '--sql-target-update-interval', str(plan['target_update_interval'])]
            jobs.append(dict(name=name, method='sql', seed=seed, particles=k,
                             steps=plan['steps'], args=args))
    manifest = dict(plan=plan, shard=shard, jobs=jobs, source=str(SOURCE), source_commit=sha,
                    unchanged_baseline_sha256=hashes, gpu_inventory=rows, created=time.time())
    if (root/'manifest.json').exists():
        saved = read(root/'manifest.json')
        assert saved['source_commit'] == sha and saved['plan'] == plan and saved['shard'] == shard
        return saved
    for name in ('jobs', 'logs', 'wandb', 'proofs', 'results', 'summary'):
        (root/name).mkdir(parents=True, exist_ok=True)
    for job in jobs:
        write(root/'jobs'/(job['name']+'.json'), dict(name=job['name'], status='pending'))
    write(root/'manifest.json', manifest)
    return manifest


def env(root, m, gpu):
    out = dict(os.environ)
    out.update(OPTIQ_SOURCE_DIR=m['source'], GMM40_REPO_ROOT=m['source'],
               GMM40_RESULTS_ROOT=str(root/'results'), GMM40_SOURCE_COMMIT=m['source_commit'],
               GMM40_CAMPAIGN=m['plan']['name'], GMM40_WANDB_DIR=str(root/'wandb'),
               CUDA_VISIBLE_DEVICES=str(gpu), JAX_PLATFORMS='cuda,cpu',
               XLA_PYTHON_CLIENT_PREALLOCATE='false', JAX_COMPILATION_CACHE_DIR=str(OPS/'cache/jax'),
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
               PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', MPLBACKEND='Agg')
    credential = Path('/workspace/optiq-clean-seed3to7-20260909/wandb_api_key')
    if credential.exists():
        out['WANDB_API_KEY'] = credential.read_text().strip()
    return out


def preflight(root, k):
    import jax
    import numpy as np
    from .sql import SQL
    from .sql_jax import SQLConfig
    from .target import initialize_target
    m = read(root/'manifest.json'); p = m['plan']
    assert jax.default_backend() == 'gpu'
    target = initialize_target()
    values = {key: target.metadata[key] for key in ('means', 'std', 'weights')}
    digest = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
    assert digest == p['reference_target_values_sha256']
    config = SQLConfig(hidden_dims=(p['width'],)*p['depth'], temperature=p['temperature'],
                       kernel_particles=k, kernel_update_ratio=p['kernel_update_ratio'],
                       value_particles=p['value_particles'],
                       target_update_interval=p['target_update_interval'])
    agent = SQL(target, seed=0, batch=p['batch'], config=config)
    initial = agent.advance(2)
    assert agent.updates == 2 and all(np.isfinite(v) for v in initial.values())
    sample, _, _ = agent.evaluate_samples(256, 937)
    assert sample.shape == (256, 2) and np.isfinite(sample).all() and abs(sample).max() <= 40.0001
    checkpoint = root/'proofs'/f'k{k}-preflight.bin'
    agent.save(checkpoint)
    restored = SQL(target, seed=0, batch=p['batch'], config=config)
    restored.restore(checkpoint)
    assert restored.updates == 2
    agent.advance(3)
    start = time.monotonic()
    info = agent.advance(10)
    elapsed = time.monotonic()-start
    assert agent.updates == 15 and all(np.isfinite(v) for v in info.values())
    proof = dict(status='passed', particles=k, batch=p['batch'], source_commit=m['source_commit'],
                 target_values_sha256=digest, seconds_per_update=elapsed/10,
                 fixed_particles=k//2, updated_particles=k//2,
                 metrics=info, device=str(jax.devices()[0]), gpu=os.environ['CUDA_VISIBLE_DEVICES'])
    write(root/'proofs'/f'k{k}.json', proof)
    print(json.dumps(proof), flush=True)


def job(root, name):
    m = read(root/'manifest.json')
    item = next(j for j in m['jobs'] if j['name'] == name)
    k = item['particles']; proof = read(root/'proofs'/f'k{k}.json', {})
    if proof.get('status') != 'passed' or proof.get('source_commit') != m['source_commit']:
        with (root/'proofs'/f'k{k}.lock').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            proof = read(root/'proofs'/f'k{k}.json', {})
            if proof.get('status') != 'passed' or proof.get('source_commit') != m['source_commit']:
                subprocess.run([sys.executable, '-B', '-u', '-m', 'gmm40.sql_particles_campaign',
                                'preflight', '--root', str(root), '--particles', str(k)], check=True)
    os.execv(sys.executable, [sys.executable, '-B', '-u', '-m', 'gmm40.campaign_job', *item['args']])


def dispatch(root):
    m = read(root/'manifest.json'); p = m['plan']
    procs = {}; leases = {}; logs = {}
    with (root/'controller.lock').open('a') as controller_lock:
        fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            states = {j['name']: read(root/'jobs'/(j['name']+'.json')) for j in m['jobs']}
            for item in m['jobs']:
                name = item['name']; state = states[name]
                if state['status'] not in ('starting', 'running'):
                    continue
                proc = procs.get(name)
                if proc is not None and proc.poll() is None:
                    continue
                if proc is None and state.get('pid') and identity(state['pid']) == state.get('proc_start') and identity(state['pid']):
                    continue
                code = procs.pop(name).wait() if proc is not None else None
                if name in logs:
                    logs.pop(name).close()
                if name in leases:
                    leases.pop(name).close()
                good = verified(root, item) and code in (0, None)
                state.update(status='completed' if good else 'failed', exit_code=code, finished=time.time())
                write(root/'jobs'/(name+'.json'), state)
                if not good:
                    write(root/'failure.json', dict(job=item, state=state,
                                                    log=str(root/'logs'/(name+'.log'))))
            running = [s for s in states.values() if s['status'] == 'running']
            pending = [j for j in m['jobs'] if states[j['name']]['status'] == 'pending']
            completed = [s['name'] for s in states.values() if s['status'] == 'completed']
            failed = [s['name'] for s in states.values() if s['status'] == 'failed']
            if len(completed) == len(m['jobs']):
                write(root/'status.json', dict(phase='reporting', completed=completed, running=[], queued=[]))
                report(root)
                write(root/'status.json', dict(phase='completed', completed=completed, running=[], queued=[], updated=time.time()))
                return
            if not failed:
                _, busy = gpus()
                busy |= {s['gpu'] for s in running}
                for gpu in p['gpus']:
                    if gpu in busy or not pending:
                        continue
                    handle = (OPS/'locks'/f'gpu-{gpu}.lock').open('a')
                    try:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        handle.close(); continue
                    item = pending.pop(0); name = item['name']
                    assert not (root/'results'/name).exists(), 'Unowned output: '+name
                    log = (root/'logs'/(name+'.log')).open('x')
                    cmd = [p['python'], '-B', '-u', '-m', 'gmm40.sql_particles_campaign',
                           'job', '--root', str(root), '--name', name]
                    state = dict(name=name, status='starting', seed=item['seed'],
                                 particles=item['particles'], gpu=gpu, started=time.time())
                    write(root/'jobs'/(name+'.json'), state)
                    proc = subprocess.Popen(cmd, cwd=m['source'], env=env(root, m, gpu),
                                            stdout=log, stderr=subprocess.STDOUT, pass_fds=(handle.fileno(),))
                    procs[name], leases[name], logs[name] = proc, handle, log
                    state.update(status='running', pid=proc.pid, proc_start=identity(proc.pid))
                    write(root/'jobs'/(name+'.json'), state)
                    running.append(state)
                    print(f'Started {name} on GPU{gpu} PID{proc.pid}', flush=True)
            write(root/'status.json', dict(phase='failed' if failed else 'running' if running else 'waiting',
                                           completed=completed, failed=failed, running=running,
                                           queued=[j['name'] for j in pending], updated=time.time(),
                                           source_commit=m['source_commit']))
            if failed and not running:
                return
            time.sleep(p['backfill_seconds'])


def report(root):
    import numpy as np
    m = read(root/'manifest.json'); rows = []
    assert all(verified(root, j) for j in m['jobs'])
    for item in m['jobs']:
        path = root/'results'/item['name']/'evaluations'/'step_0100000'/'metrics.json'
        result = read(path)
        rows.append(dict(particles=item['particles'], seed=item['seed'],
                         mmd2=result['mmd2'], coverage=result['mode_coverage'],
                         near=result['high_density_fraction'],
                         train_seconds=read(root/'results'/item['name']/'status.json')['train_seconds']))
    write(root/'result.json', dict(status='completed', source_commit=m['source_commit'],
                                   shard=m['shard'], rows=rows))
    with (root/'summary'/'per_seed.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def register(root, shard):
    m = prepare(root, shard)
    name = m['plan']['name']+f'-shard{shard}'
    config = OPS/'supervisor/jobs'/(name+'.conf')
    assert not config.exists(), 'Already registered'
    config.write_text('\n'.join([f'[program:{name}]',
        f"command={m['plan']['python']} -B -u -m gmm40.sql_particles_campaign run --root {root}",
        f'directory={SOURCE}', 'autostart=true', 'autorestart=false', 'startsecs=3',
        'startretries=0', 'stopasgroup=true', 'killasgroup=true', 'stopwaitsecs=30',
        'redirect_stderr=true', f'stdout_logfile={root}/controller.log',
        'stdout_logfile_maxbytes=10MB',
        'environment=PYTHONDONTWRITEBYTECODE="1",JAX_PLATFORMS="cpu",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2"', '']))
    write(root/'registration.json', dict(service=name, source_commit=m['source_commit'], shard=shard,
                                         automatic_restarts=False, jobs=len(m['jobs'])))
    subprocess.run(CTL+['reread'], check=True)
    subprocess.run(CTL+['update', name], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['register', 'run', 'job', 'preflight', 'report'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--shard', type=int)
    parser.add_argument('--name')
    parser.add_argument('--particles', type=int)
    args = parser.parse_args(); root = args.root.resolve(); root.mkdir(parents=True, exist_ok=True)
    os.environ.update(GMM40_REPO_ROOT=str(SOURCE), GMM40_RESULTS_ROOT=str(root/'results'))
    if args.action == 'register': register(root, args.shard)
    elif args.action == 'job': job(root, args.name)
    elif args.action == 'preflight': preflight(root, args.particles)
    elif args.action == 'report': report(root)
    else:
        try: dispatch(root)
        except Exception as exc:
            write(root/'failure.json', dict(stage='controller', error=repr(exc), time=time.time()))
            raise


if __name__ == '__main__':
    main()
