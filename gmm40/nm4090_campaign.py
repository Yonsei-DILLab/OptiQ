"""4090-only N/M ablation; unchanged learner, independent slots, no retries."""
import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

from .campaign import identity, read, verified, write

SOURCE = Path(__file__).resolve().parents[1]
OPS = Path('/home/heechan/OptiQ-ops')
LEGACY = [Path('/workspace/optiq-gmm-trg-20260921'), Path('/workspace/optiq-nm-20260922')]
CTL = ['supervisorctl', '-c', str(OPS/'supervisor/supervisord.conf')]


def inventory():
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name',
                                    '--format=csv,noheader'], text=True)
    rows = list(csv.reader(text.splitlines(), skipinitialspace=True))
    assert len(rows) == 4 and all('RTX 4090' in row[2] for row in rows), rows
    mapping = {row[1]: int(row[0]) for row in rows}
    text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader'], text=True)
    busy = {mapping[row[0]] for row in csv.reader(text.splitlines(), skipinitialspace=True)
            if row and row[0] in mapping}
    return rows, busy


def invariant_hashes(plan):
    prefix = 'analysis_tools/experiments/20260920_truncated_mll/optiq_dime/'
    paths = ['gmm40/'+p+'.py' for p in ('optiq_trg', 'optiq', 'target', 'evaluation')]
    paths += [prefix+p+'.py' for p in ('policy', 'semi_implicit', 'distillation', 'box_gaussian')]
    hashes = {}
    for path in paths:
        old = subprocess.check_output(['git', 'show', plan['reference_commit']+':'+path], cwd=SOURCE)
        assert old == (SOURCE/path).read_bytes(), 'Reference learner/evaluation changed: '+path
        hashes[path] = hashlib.sha256(old).hexdigest()
    return hashes


def make_jobs(plan):
    jobs = []
    keys = ('steps', 'batch', 'width', 'depth', 'temperature', 'eval_samples',
            'mean_output_init_scale', 'trg_log_std_max', 'trg_initial_log_std', 'trg_teacher_std_floor')
    for seed in plan['seeds']:
        for nm in plan['nm']:
            name = f'ibolt_nm{nm}_s{seed}_100k'
            args = ['--method', plan['method'], '--name', name, '--seed', str(seed),
                    '--n', str(nm), '--m', str(nm)]
            for key in keys:
                args += ['--'+key.replace('_', '-'), str(plan[key])]
            jobs.append(dict(name=name, method=plan['method'], seed=seed, nm=nm,
                             steps=plan['steps'], args=args))
    return jobs


def prepare(root):
    plan = read(SOURCE/'gmm40/nm4090_plan.json')
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip()
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=SOURCE, text=True).strip()
    hashes = invariant_hashes(plan)
    rows, _ = inventory()
    if (root/'manifest.json').exists():
        old = read(root/'manifest.json')
        assert old['source_commit'] == sha and old['plan'] == plan
        return old
    jobs = make_jobs(plan)
    manifest = dict(source=str(SOURCE), source_commit=sha, plan=plan, jobs=jobs,
                    gpu_inventory=rows, unchanged_source_sha256=hashes, created=time.time())
    for job in jobs:
        write(root/'jobs'/(job['name']+'.json'), dict(name=job['name'], status='pending'))
    for name in ('logs', 'wandb', 'proofs'):
        (root/name).mkdir(parents=True, exist_ok=True)
    write(root/'manifest.json', manifest)
    write(root/'results/queue.json', dict(jobs=jobs))
    return manifest


def lease(gpu):
    handles = []
    paths = [OPS/'locks'/f'gpu-{gpu}.lock'] + [p/f'gpu{gpu}.lock' for p in LEGACY if p.exists()]
    try:
        for path in paths:
            handle = path.open('a')
            handles.append(handle)
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return handles
    except BlockingIOError:
        for handle in handles:
            handle.close()
        return None


def environment(root, manifest, gpu):
    env = dict(os.environ)
    env.update(OPTIQ_SOURCE_DIR=manifest['source'], GMM40_REPO_ROOT=manifest['source'],
               GMM40_RESULTS_ROOT=str(root/'results'), GMM40_SOURCE_COMMIT=manifest['source_commit'],
               GMM40_CAMPAIGN=manifest['plan']['name'], GMM40_WANDB_DIR=str(root/'wandb'),
               CUDA_VISIBLE_DEVICES=str(gpu), JAX_PLATFORMS='cuda,cpu',
               XLA_PYTHON_CLIENT_PREALLOCATE='false', JAX_COMPILATION_CACHE_DIR=str(OPS/'cache/jax'),
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
               PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', MPLBACKEND='Agg')
    credential = Path('/workspace/optiq-clean-seed3to7-20260909/wandb_api_key')
    if credential.exists():
        env['WANDB_API_KEY'] = credential.read_text().strip()
    return env


def preflight(root, manifest, nm):
    import jax
    import numpy as np
    from .optiq_trg import OptiQTRG
    from .target import initialize_target
    assert jax.default_backend() == 'gpu'
    p = manifest['plan']
    target = initialize_target()
    target_values = {k: target.metadata[k] for k in ('means', 'std', 'weights')}
    target_hash = hashlib.sha256(json.dumps(target_values, sort_keys=True).encode()).hexdigest()
    assert target_hash == p['reference_target_values_sha256'], 'GMM40 target differs from paper control'
    kwargs = dict(n=nm, m=nm, batch=p['batch'], hidden_dims=(p['width'],)*p['depth'],
                  temperature=p['temperature'], log_std_max=p['trg_log_std_max'],
                  initial_log_std=p['trg_initial_log_std'], teacher_std_floor=p['trg_teacher_std_floor'],
                  mean_output_init_scale=p['mean_output_init_scale'])
    agent = OptiQTRG(target, seed=0, **kwargs)
    params = sum(x.size for x in jax.tree_util.tree_leaves(agent.state.params))
    assert params == 133636
    info = agent.advance(2)
    assert agent.updates == int(agent.state.step) == 2
    assert all(np.isfinite(v) for v in info.values())
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(agent.state))
    samples, _, extra = agent.evaluate_samples(256, 937)
    for x in (samples, extra['mu_only']):
        assert x.shape == (256, 2) and np.isfinite(x).all() and abs(x).max() <= 40
    checkpoint = root/'proofs'/f'nm{nm}-preflight.bin'
    agent.save(checkpoint)
    restored = OptiQTRG(target, seed=0, **kwargs)
    restored.restore(checkpoint)
    assert restored.updates == 2
    for x, y in zip(jax.tree_util.tree_leaves(agent.state), jax.tree_util.tree_leaves(restored.state)):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(agent.key, restored.key)
    agent.advance(10)  # Compile this timing block separately.
    start = time.monotonic()
    info = agent.advance(10)
    elapsed = time.monotonic()-start
    assert agent.updates == int(agent.state.step) == 22
    assert all(np.isfinite(v) for v in info.values())
    proof = dict(status='passed', nm=nm, batch=p['batch'], updates=22, parameters=params,
                 reference_target_values_sha256=target_hash,
                 source_commit=manifest['source_commit'], metrics=info,
                 seconds_per_update=elapsed/10, devices=[str(d) for d in jax.devices()],
                 memory_stats=jax.devices()[0].memory_stats(),
                 packages={k: importlib.metadata.version(k) for k in
                           ('jax', 'jaxlib', 'flax', 'optax', 'torch', 'numpy', 'wandb')},
                 gpu=os.environ['CUDA_VISIBLE_DEVICES'], time=time.time())
    write(root/'proofs'/f'nm{nm}.json', proof)
    print(json.dumps(proof), flush=True)


def run_job(root, name):
    m = read(root/'manifest.json')
    job = next(j for j in m['jobs'] if j['name'] == name)
    # Probe all shapes once on the first acquired GPU, before any main training.
    for nm in m['plan']['nm']:
        proof = read(root/'proofs'/f'nm{nm}.json', {})
        if proof.get('status') == 'passed' and proof.get('source_commit') == m['source_commit']:
            continue
        with (root/'proofs'/f'nm{nm}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            proof = read(root/'proofs'/f'nm{nm}.json', {})
            if proof.get('status') == 'passed' and proof.get('source_commit') == m['source_commit']:
                continue
            subprocess.run([sys.executable, '-B', '-u', '-m', 'gmm40.nm4090_campaign',
                            'preflight', '--root', str(root), '--nm', str(nm)], check=True)
    os.execv(sys.executable, [sys.executable, '-B', '-u', '-m', 'gmm40.campaign_job', *job['args']])


def dispatch(root):
    m = read(root/'manifest.json')
    handles, processes, logs = {}, {}, {}
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            states = {j['name']: read(root/'jobs'/(j['name']+'.json')) for j in m['jobs']}
            for job in m['jobs']:
                name = job['name']; s = states[name]
                if s['status'] not in ('running', 'starting'):
                    continue
                proc = processes.get(name)
                if proc is not None and proc.poll() is None:
                    continue
                if proc is None and s.get('pid') and identity(s['pid']) == s.get('proc_start') and identity(s['pid']):
                    continue  # Adopt live work; never duplicate it after a controller interruption.
                code = processes.pop(name).wait() if proc is not None else None
                if name in logs:
                    logs.pop(name).close()
                for handle in handles.pop(name, []):
                    handle.close()
                good = verified(root, job) and code in (0, None)
                s.update(status='completed' if good else 'failed', exit_code=code, finished=time.time())
                write(root/'jobs'/(name+'.json'), s)
                if not good:
                    write(root/'failure.json', dict(job=job, state=s, log=str(root/'logs'/(name+'.log'))))
            failed = [s for s in states.values() if s['status'] == 'failed']
            running = [s for s in states.values() if s['status'] == 'running']
            pending = [j for j in m['jobs'] if states[j['name']]['status'] == 'pending']
            completed = [s['name'] for s in states.values() if s['status'] == 'completed']
            if len(completed) == len(m['jobs']):
                write(root/'status.json', dict(phase='reporting', completed=completed, queued=[], running=[]))
                report(root)
                write(root/'status.json', dict(phase='completed', completed=completed, queued=[], running=[], updated=time.time()))
                archive = root/'final-results.tar.gz'
                with tarfile.open(archive, 'w:gz') as tar:
                    for item in ('manifest.json', 'status.json', 'result.json', 'registration.json', 'jobs', 'proofs', 'results', 'summary'):
                        tar.add(root/item, arcname=item)
                h = hashlib.file_digest(archive.open('rb'), 'sha256').hexdigest()
                write(root/'archive.json', dict(file=archive.name, sha256=h, bytes=archive.stat().st_size))
                return
            if not failed:
                _, busy = inventory()
                busy |= {s['gpu'] for s in running}
                for gpu in m['plan']['gpus']:
                    if gpu in busy or not pending:
                        continue
                    leases = lease(gpu)
                    if leases is None:
                        continue
                    job = pending.pop(0); name = job['name']
                    assert not (root/'results'/name).exists(), 'Existing unowned output: '+name
                    log = (root/'logs'/(name+'.log')).open('x')
                    cmd = [sys.executable, '-B', '-u', '-m', 'gmm40.nm4090_campaign',
                           'job', '--root', str(root), '--name', name]
                    s = dict(name=name, status='starting', gpu=gpu, nm=job['nm'], seed=job['seed'], started=time.time())
                    write(root/'jobs'/(name+'.json'), s)
                    proc = subprocess.Popen(cmd, env=environment(root, m, gpu), cwd=m['source'],
                                            stdout=log, stderr=subprocess.STDOUT,
                                            pass_fds=tuple(h.fileno() for h in leases))
                    processes[name], handles[name], logs[name] = proc, leases, log
                    s.update(status='running', pid=proc.pid, proc_start=identity(proc.pid))
                    write(root/'jobs'/(name+'.json'), s)
                    running.append(s)
                    print(f'Started {name} on RTX4090 GPU{gpu}, PID{proc.pid}', flush=True)
            write(root/'status.json', dict(phase='failed' if failed else 'running' if running else 'waiting',
                                          running=running, queued=[j['name'] for j in pending], completed=completed,
                                          failed=failed, updated=time.time(), source_commit=m['source_commit']))
            if failed and not running:
                return
            time.sleep(m['plan']['backfill_seconds'])


def report(root):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .evaluation import background
    from .target import Target
    m = read(root/'manifest.json')
    assert all(verified(root, j) for j in m['jobs'])
    target = Target(); rows = []; summary = root/'summary'; summary.mkdir(exist_ok=True)
    for mode, suffix in [('full_policy', ''), ('mu_only', '_mu_only')]:
        fig, axes = plt.subplots(3, 4, figsize=(15, 12), constrained_layout=True)
        for job in m['jobs']:
            folder = root/'results'/job['name']/'evaluations'/'step_0100000'
            metrics = read(folder/('metrics'+suffix+'.json'))
            samples = np.load(folder/('samples'+suffix+'.npy'))
            assert samples.shape == (10000, 2) and np.isfinite(samples).all()
            row = dict(nm=job['nm'], seed=job['seed'], mode=mode, mmd=math.sqrt(max(metrics['mmd2'], 0)),
                       mmd2=metrics['mmd2'], coverage=metrics['mode_coverage'], near=metrics['high_density_fraction'])
            rows.append(row)
            ax = axes[m['plan']['nm'].index(job['nm']), job['seed']]
            background(ax, target)
            ax.scatter(*samples.T, s=2.1, alpha=.4, c='#0000ff', linewidths=0, rasterized=True)
            ax.set_title(f"iBOLT N=M={job['nm']} | seed{job['seed']}\n{row['coverage']}/40 | MMD {row['mmd']:.4f}")
        fig.suptitle(f'GMM40 | 100k actor updates | {mode}')
        for ext in ('png', 'pdf'):
            fig.savefig(summary/f'{mode}.{ext}', dpi=180)
        plt.close(fig)
    aggregates = []
    for mode in ('full_policy', 'mu_only'):
        for nm in m['plan']['nm']:
            selected = [r for r in rows if r['mode'] == mode and r['nm'] == nm]
            assert len(selected) == 4
            aggregates.append(dict(mode=mode, nm=nm, seeds=4, **{
                key: dict(mean=float(np.mean([r[key] for r in selected])),
                          sample_sd=float(np.std([r[key] for r in selected], ddof=1)))
                for key in ('mmd', 'mmd2', 'coverage', 'near')}))
    write(root/'result.json', dict(status='completed', source_commit=m['source_commit'], rows=rows, aggregates=aggregates))
    with (summary/'per_seed.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def register(root):
    m = prepare(root); name = m['plan']['name']; conf = OPS/'supervisor/jobs'/(name+'.conf')
    assert not conf.exists(), 'Already registered; do not duplicate'
    conf.write_text('\n'.join([f'[program:{name}]',
        f"command={m['plan']['python']} -B -u -m gmm40.nm4090_campaign run --root {root}",
        f'directory={SOURCE}', 'autostart=true', 'autorestart=false', 'startsecs=3', 'startretries=0',
        'stopasgroup=true', 'killasgroup=true', 'stopwaitsecs=30', 'redirect_stderr=true',
        f'stdout_logfile={root}/controller.log', 'stdout_logfile_maxbytes=10MB',
        'environment=PYTHONDONTWRITEBYTECODE="1",JAX_PLATFORMS="cpu",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2"', '']))
    write(root/'registration.json', dict(service=name, source_commit=m['source_commit'], source=str(SOURCE),
                                        created=time.time(), automatic_restarts=False, jobs=len(m['jobs'])))
    subprocess.run(CTL+['reread'], check=True)
    subprocess.run(CTL+['update', name], check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['register', 'run', 'job', 'preflight', 'report'])
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--name'); p.add_argument('--nm', type=int)
    a = p.parse_args(); root = a.root.resolve(); root.mkdir(parents=True, exist_ok=True)
    os.environ.update(GMM40_REPO_ROOT=str(SOURCE), GMM40_RESULTS_ROOT=str(root/'results'))
    if a.action == 'register': register(root)
    elif a.action == 'job': run_job(root, a.name)
    elif a.action == 'preflight': preflight(root, read(root/'manifest.json'), a.nm)
    elif a.action == 'report': report(root)
    else:
        try: dispatch(root)
        except Exception as exc:
            write(root/'failure.json', dict(stage='controller', error=repr(exc), time=time.time()))
            raise


if __name__ == '__main__':
    main()
