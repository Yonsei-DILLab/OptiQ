"""Seed-priority Vast GMM40 sweep with bounded, non-retrying GPU workers."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .campaign import read, write, verified
from . import nm4090_campaign as base

MODULE = 'gmm40.extended_campaign'
NAME = 'gmm40-extended-nm-20260926'
HOST_NM = {'vast1': [256], 'vast2': [1024, 4096], 'vast3': [16, 64], 'vast4': [1, 4]}


def inventory():
    rows = list(csv.reader(subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name',
        '--format=csv,noheader'], text=True).splitlines(), skipinitialspace=True))
    assert len(rows) == 4 and all(any(x in r[2] for x in ('3090', '4090', '5090')) for r in rows)
    mapping = {r[1]: int(r[0]) for r in rows}
    apps = csv.reader(subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader'], text=True).splitlines(), skipinitialspace=True)
    return rows, {mapping[r[0]] for r in apps if r and r[0] in mapping}


def enable_cloud():
    from .cloud_trg import CloudTRG
    from . import optiq_trg
    optiq_trg.OptiQTRG = CloudTRG


def equivalence(root):
    import jax
    import numpy as np
    from .optiq_trg import OptiQTRG
    from .cloud_trg import CloudTRG
    from .target import initialize_target
    target = initialize_target()
    errors = []
    for n in (1, 4, 16):
        kw = dict(seed=0, n=n, m=n, batch=8, hidden_dims=(256,)*3,
            log_std_max=-3.5, initial_log_std=-4., teacher_std_floor=.05,
            mean_output_init_scale=16.)
        a, b = OptiQTRG(target, **kw), CloudTRG(target, **kw)
        ia, ib = a.advance(1), b.advance(1)
        delta = []
        for x, y in zip(jax.tree_util.tree_leaves(a.state.params), jax.tree_util.tree_leaves(b.state.params)):
            np.testing.assert_allclose(x, y, atol=2e-6, rtol=2e-5)
            delta.append(float(np.max(np.abs(np.asarray(x)-np.asarray(y)))))
        for k in ia:
            np.testing.assert_allclose(ia[k], ib[k], atol=2e-4, rtol=2e-4, err_msg=k)
        np.testing.assert_array_equal(a.key, b.key)
        errors.append(dict(n=n, max_parameter_error=max(delta), metrics=ia))
    write(root/'proofs/cloud-equivalence.json', dict(status='passed', cases=errors))


def register(root, host):
    plan = read(base.SOURCE/'gmm40/nm4090_plan.json')
    plan.update(name=NAME, host=host, nm=HOST_NM[host],
                gpus=[1,3] if HOST_NM[host]==[512] else list(range(4)),
                gpu_model='Vast 3090/4090/5090', python=sys.executable,
                reference_commit='c429fbb', seed_order='0 then 1 then 2 then 3')
    root.mkdir(parents=True, exist_ok=True)
    planpath = root/'plan.json'
    write(planpath, plan)
    base.inventory = inventory
    manifest = base.prepare(root, planpath)
    # Initialize the shared target once, before concurrent evaluation workers.
    os.environ.update(base.environment(root, manifest, 0))
    from .target import initialize_target
    initialize_target()
    names = []
    for gpu in plan['gpus']:
        name = f'{root.name}-gpu{gpu}'
        conf = Path('/etc/supervisor/conf.d')/(name+'.conf')
        assert not conf.exists(), 'Already registered: '+str(conf)
        conf.write_text('\n'.join([f'[program:{name}]',
            f'command={sys.executable} -B -u -m {MODULE} worker --root {root} --gpu {gpu}',
            f'directory={base.SOURCE}', 'autostart=true', 'autorestart=false',
            'startsecs=0', 'startretries=0', 'stopasgroup=true', 'killasgroup=true',
            'stopwaitsecs=30', 'redirect_stderr=true', f'stdout_logfile={root}/logs/worker{gpu}.log',
            'stdout_logfile_maxbytes=10MB', 'environment=PYTHONDONTWRITEBYTECODE="1"', '']))
        names.append(name)
    write(root/'registration.json', dict(services=names, source_commit=manifest['source_commit']))
    subprocess.run(['supervisorctl', 'reread'], check=True)
    for name in names:
        subprocess.run(['supervisorctl', 'update', name], check=True)


def worker(root, gpu):
    manifest = read(root/'manifest.json')
    # Respect both current and historical GPU leases.
    (base.OPS/'locks').mkdir(parents=True, exist_ok=True)
    leases = base.lease(gpu)
    if leases is None or gpu in inventory()[1]:
        raise RuntimeError(f'GPU {gpu} occupied or leased; no training started')
    env = base.environment(root, manifest, gpu)
    for path in ('/workspace/optiq-mujoco-exp-20260908/wandb_api_key',
                 '/workspace/optiq-mujoco-warmup-20260908/wandb_api_key'):
        if 'WANDB_API_KEY' not in env and Path(path).exists():
            env['WANDB_API_KEY'] = Path(path).read_text().strip()
    while True:
        with (root/'queue.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            jobs = [j for j in manifest['jobs'] if read(root/'jobs'/(j['name']+'.json'))['status']=='pending']
            if not jobs:
                return
            job = jobs[0]
            state = dict(name=job['name'], status='preflight', gpu=gpu, nm=job['nm'], seed=job['seed'], started=time.time())
            write(root/'jobs'/(job['name']+'.json'), state)
        code = 1
        with (root/'logs'/(job['name']+'.log')).open('x') as log:
            nm = job['nm']
            with (root/'proofs'/f'nm{nm}.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                proof = read(root/'proofs'/f'nm{nm}.json', {})
                if proof.get('status') != 'passed':
                    code = subprocess.call([sys.executable, '-B', '-u', '-m', MODULE, 'probe',
                        '--root', str(root), '--nm', str(nm)], env=env, stdout=log, stderr=subprocess.STDOUT)
                else:
                    code = 0
            if code == 0:
                state.update(status='running', training_started=time.time())
                write(root/'jobs'/(job['name']+'.json'), state)
                code = subprocess.call([sys.executable, '-B', '-u', '-m', MODULE, 'train',
                    '--root', str(root), '--nm', str(nm), *job['args']], env=env,
                    stdout=log, stderr=subprocess.STDOUT, pass_fds=tuple(h.fileno() for h in leases))
        state.update(status='completed' if code==0 and verified(root, job) else 'failed',
                     exit_code=code, finished=time.time())
        write(root/'jobs'/(job['name']+'.json'), state)
        if state['status']=='failed':
            # No automatic retries and no further same-shape jobs after a failure.
            with (root/'queue.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                for j in manifest['jobs']:
                    path = root/'jobs'/(j['name']+'.json')
                    s = read(path)
                    if j['nm']==nm and s['status']=='pending':
                        s.update(status='held_after_failure')
                        write(path, s)


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('mode', choices=['register','worker','probe','train'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--host', choices=list(HOST_NM))
    p.add_argument('--gpu', type=int)
    p.add_argument('--nm', type=int)
    a, rest = p.parse_known_args()
    if a.mode=='register':
        if a.nm is not None:
            assert a.host=='vast2' and a.nm==512, 'Only the approved 4096-to-512 replacement'
            HOST_NM[a.host]=[512]
        register(a.root, a.host)
    elif a.mode=='worker': worker(a.root, a.gpu)
    elif a.mode=='probe':
        if a.nm>=1024:
            equivalence(a.root)
            enable_cloud()
        base.preflight(a.root, read(a.root/'manifest.json'), a.nm)
    else:
        if a.nm>=1024: enable_cloud()
        sys.argv = [sys.argv[0], *rest]
        from .campaign_job import main as train
        train()


if __name__=='__main__': main()
