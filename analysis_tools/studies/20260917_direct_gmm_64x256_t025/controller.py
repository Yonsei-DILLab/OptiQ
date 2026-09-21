"""Append T0.25 after the queued N64/M256/T0.5 campaign fully finishes."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
import traceback


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def tasks_for(plan):
    return [dict(name=f'{env}_s{seed}', env=env, seed=seed, gpu=gpu,
                 state='pending')
            for env in plan['order'] for gpu, seed in enumerate(plan['seeds'])]


def command(plan, task, commit):
    return [plan['python'], str(Path(plan['repo']) / 'run_optiq_dime.py'),
            '--config-name=' + plan['config_name'],
            'benchmark=' + task['env'], f'seed={task["seed"]}',
            f'alg.actor.num_policy_samples={plan["n"]}',
            f'alg.actor.proposals_per_policy_sample={plan["m"] // plan["n"]}',
            f'alg.actor.temperature={plan["temperature"]}',
            'alg.actor.distillation_loss=direct_gmm_nll',
            f'total_steps={plan["total_steps"]}',
            'output_root=' + str(Path(plan['root']) / 'outputs' / task['name']),
            f'run_name={task["env"]}-DirectGMM-T{plan["temperature"]}-N{plan["n"]}-M{plan["m"]}-s{task["seed"]}',
            'wandb.activate=true', 'wandb.mode=online',
            'wandb.entity=' + plan['wandb_entity'],
            'wandb.project=' + plan['wandb_project'],
            'wandb.group=' + plan['wandb_group'],
            '+experiment_commit=' + commit]


def predecessor_state(root):
    root = Path(root)
    if not (root / 'queue.json').exists():
        return 'waiting_for_predecessor'
    q = read(root / 'queue.json')
    if any(t['state'] == 'failed' for t in q['tasks']) or (root / 'CONTROLLER_ERROR.json').exists():
        return 'blocked_predecessor_failure'
    if (q.get('state') == 'complete' and len(q['tasks']) == 20
            and all(t['state'] == 'complete' for t in q['tasks'])
            and (root / 'ALL_COMPLETE.json').exists()):
        return 'ready'
    return 'waiting_for_predecessor'


def verify_source(plan):
    repo = Path(plan['repo'])
    source = read(repo / 'DIRECT_GMM_SOURCE.json')
    assert source['code_id'] == plan['source_code_id']
    for rel, expected in source['files'].items():
        assert sha(repo / rel) == expected, f'Frozen source differs: {rel}'


def main(root):
    root = Path(root)
    plan = read(root / 'plan.json')
    assert str(root) == plan['root']
    assert plan['m'] % plan['n'] == 0
    deployment = read(root / 'DEPLOYMENT.json')
    commit = deployment['commit']
    assert len(commit) == 40
    for rel, expected in deployment['bundle_files'].items():
        assert sha(root / rel) == expected, rel
    own_lock = (root / 'controller.lock').open('a')
    fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify_source(plan)
    assert read(root / 'PREFLIGHT.json')['passed']
    assert read(root / 'PREFLIGHT.json')['commit'] == commit
    queue_path = root / 'queue.json'
    if queue_path.exists():
        q = read(queue_path)
        assert q['commit'] == commit
        if q.get('state') == 'complete':
            return
        # Checkpoints omit replay/RNG. Never silently restart interrupted runs.
        assert all(t['state'] == 'pending' for t in q['tasks']), 'Started queue needs explicit recovery'
    else:
        tasks = tasks_for(plan)
        for t in tasks:
            t['command'] = command(plan, t, commit)
        q = dict(campaign=plan['campaign'], order=plan['order'], tasks=tasks,
                 n=plan['n'], m=plan['m'], temperature=plan['temperature'],
                 total_steps=plan['total_steps'], commit=commit,
                 source_code_id=plan['source_code_id'], predecessor=plan['predecessor'],
                 state='waiting_for_predecessor', registered_unix=time.time())
        write(queue_path, q)
    while True:
        state = predecessor_state(plan['predecessor'])
        if state == 'ready':
            break
        if q.get('state') != state:
            q['state'] = state
            write(queue_path, q)
        time.sleep(30)
    # The predecessor holds this lock for all five environments. Acquiring it
    # after its success marker also waits for its controller to fully exit.
    gpu_lock = (Path(plan['predecessor']) / 'controller.lock').open('a')
    fcntl.flock(gpu_lock, fcntl.LOCK_EX)
    assert predecessor_state(plan['predecessor']) == 'ready'
    verify_source(plan)
    subprocess.run([plan['python'], str(root / 'preflight.py'), '--root', str(root)], check=True)
    env = os.environ.copy()
    for key in ['WANDB_RUN_ID', 'WANDB_RESUME', 'WANDB_PROJECT', 'WANDB_NAME',
                'WANDB_RUN_GROUP', 'JAX_PLATFORMS', 'JAX_PLATFORM_NAME']:
        env.pop(key, None)
    env.update(WANDB_API_KEY=Path('/workspace/optiq-secrets/wandb_api_key').read_text().strip(),
               WANDB_ENTITY=plan['wandb_entity'], WANDB_MODE='online',
               XLA_PYTHON_CLIENT_PREALLOCATE='false', MUJOCO_GL='egl',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONUNBUFFERED='1')
    q.update(state='running', started_unix=time.time())
    write(queue_path, q)
    (root / 'logs').mkdir(exist_ok=True)
    for name in plan['order']:
        group = [t for t in q['tasks'] if t['env'] == name]
        active = []
        for t in group:
            out = root / 'outputs' / t['name']
            out.mkdir(parents=True, exist_ok=False)
            write(out / 'RUN_PROVENANCE.json', dict(commit=commit, command=t['command'],
                  source_code_id=plan['source_code_id'], campaign=plan['campaign']))
            log = (root / 'logs' / (t['name'] + '.log')).open('x')
            p = subprocess.Popen(t['command'], cwd=plan['repo'],
                                 env=dict(env, CUDA_VISIBLE_DEVICES=str(t['gpu'])),
                                 stdout=log, stderr=subprocess.STDOUT)
            t.update(state='running', pid=p.pid, output_root=str(out), started_unix=time.time())
            active.append((t, p, log))
            write(queue_path, q)
        while active:
            for t, p, log in active[:]:
                if p.poll() is None:
                    continue
                log.close()
                markers = list(Path(t['output_root']).glob('*/completed.json'))
                ok = (p.returncode == 0 and len(markers) == 1
                      and read(markers[0])['timesteps'] == plan['total_steps'])
                t.update(state='complete' if ok else 'failed', exit_code=p.returncode,
                         finished_unix=time.time())
                if ok:
                    run = markers[0].parent
                    t['wandb_url'] = read(markers[0])['wandb_url']
                    files = {str(f.relative_to(run)): sha(f) for f in run.rglob('*')
                             if f.is_file() and not f.is_symlink()
                             and 'wandb' not in f.relative_to(run).parts}
                    write(run / 'ARTIFACTS_SHA256.json', files)
                active.remove((t, p, log))
                write(queue_path, q)
            if active:
                time.sleep(15)
        if any(t['state'] == 'failed' for t in group):
            q['state'] = 'blocked_after_failed_group'
            write(queue_path, q)
            return
    q.update(state='complete', finished_unix=time.time())
    write(queue_path, q)
    write(root / 'ALL_COMPLETE.json', dict(runs=len(q['tasks']), commit=commit,
                                          finished_unix=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=str(Path(__file__).resolve().parent))
    args = parser.parse_args()
    try:
        main(args.root)
    except Exception:
        write(Path(args.root) / 'CONTROLLER_ERROR.json',
              dict(error=traceback.format_exc(), time=time.time()))
        raise
