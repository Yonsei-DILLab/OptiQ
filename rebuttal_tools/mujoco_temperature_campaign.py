"""Two-seed temperature selection followed by three held-out training seeds.

This controller can run outside the immutable learner worktree. The plan pins the
learner revision and contains every adopted process identity and output path.
"""
import argparse
from collections import Counter
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

GRID = (.1, .25, .5)
ACTIVE = ('initializing', 'running')


def score_temperature(jobs):
    """Average the last five native stochastic-z evaluations within each seed."""
    import numpy as np
    rows = []
    for temperature in GRID:
        seeds = []
        for seed in (0, 1):
            matches = [j for j in jobs if j['temperature'] == temperature and j['seed'] == seed]
            assert len(matches) == 1 and matches[0]['status'] == 'completed'
            j = matches[0]
            result = json.loads((Path(j['output']) / 'completed.json').read_text())
            files = list((Path(result['actual_run']) / 'eval').rglob('evaluations_stochastic_z.npz'))
            assert len(files) == 1
            with np.load(files[0]) as data:
                steps = data['timesteps'][-5:]
                rewards = data['results'][-5:]
                assert steps.tolist() == [980000, 985000, 990000, 995000, 1000000]
                assert rewards.shape == (5, 10) and np.isfinite(rewards).all()
                means = rewards.mean(axis=1)
                seeds.append(dict(seed=seed, score=float(means.mean()),
                    evaluation_steps=steps.tolist(), evaluation_means=means.tolist(),
                    source=str(files[0])))
        scores = [s['score'] for s in seeds]
        rows.append(dict(temperature=temperature, mean=float(np.mean(scores)),
            seed_std=float(np.std(scores, ddof=1)), seeds=seeds))
    winner = max(rows, key=lambda r: (r['mean'], -r['temperature']))
    return dict(temperature=winner['temperature'], candidates=rows,
        rule='Within-seed last 5 evaluations (980k..1M, 10 episodes each), then mean over seeds 0 and 1; ties choose lower temperature.',
        selected_unix=time.time(), evaluation_mode='stochastic_z: fresh z, mu only')


def make_job(plan, task, utd, temperature, seed, phase='search'):
    ident = f'{task}-ibolt-crossqAC-ls5to3.5-utd{utd}-T{temperature:g}-s{seed}'
    folder = Path(plan['folder'])
    return dict(id=ident, task=task, method='ibolt', utd=utd,
        temperature=temperature, seed=seed, phase=phase, status='queued',
        source_commit=plan['source_commit'], output=str(folder/'runs'/ident),
        spec=str(folder/'specs'/(ident+'.json')), gpu=None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    path = args.plan.resolve()
    folder = path.parent
    lock = (folder/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads(path.read_text())
    sys.path.insert(0, plan['source'])
    from rebuttal_tools.mujoco_logstd_worker import atomic_json
    from rebuttal_tools.mujoco_logstd_campaign import (
        live, process_info, inventory, available_host_mib, command, environment,
        inspect_environments,
    )
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=plan['source'], text=True).strip() == plan['source_commit']
    for name in ('logs', 'runs', 'preflights', 'specs', 'selections'):
        (folder/name).mkdir(exist_ok=True)
    plan.update(controller_pid=os.getpid(), status='validating_gpu')
    atomic_json(path, plan)
    try:
        if not plan.get('preflight', {}).get('passed'):
            inspect_environments(plan, folder)
            spec = dict(id='humanoid-preflight-utd2', task='humanoid', method='ibolt',
                seed=100, temperature=.5, utd=2, source_commit=plan['source_commit'],
                output=str(folder/'preflights'/'humanoid-preflight-utd2'))
            specpath = folder/'specs'/'humanoid-preflight-utd2.json'
            atomic_json(specpath, spec)
            gpus, _ = inventory()
            gpu = min(gpus, key=lambda g: gpus[g]['used'])
            peak = 0
            with (folder/'logs'/'preflight.log').open('x') as stream:
                child = subprocess.Popen(command(plan, specpath, plan['cpu_lanes'][0], True),
                    cwd=plan['source'], env=environment(plan, gpu), stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT)
                while child.poll() is None:
                    _, apps = inventory()
                    peak = max(peak, apps.get(child.pid, 0))
                    time.sleep(2)
                if child.wait() != 0:
                    raise RuntimeError('GPU preflight failed: '+str(folder/'logs'/'preflight.log'))
            result = json.loads((Path(spec['output'])/'completed.json').read_text())
            assert result['backend'] == 'gpu' and result['updates'] == 12 and result['actor_updates'] == 4
            plan['preflight'] = dict(passed=True, result=result, peak_gpu_mib=peak)
            plan['gpu_reserve_mib'] = max(2560, math.ceil((peak*1.25+512)/512)*512)
            atomic_json(folder/'preflight-verification.json', plan['preflight'])
        if args.preflight_only:
            plan['status'] = 'prepared'
            atomic_json(path, plan)
            return
        children = {}
        stopping = False

        def shutdown(_signum, _frame):
            nonlocal stopping
            stopping = True

        signal.signal(signal.SIGTERM, shutdown)
        signal.signal(signal.SIGINT, shutdown)
        plan['status'] = 'running'
        while True:
            if stopping:
                for j in plan['jobs']:
                    if j['status'] in ACTIVE and live(j):
                        os.kill(j['pid'], signal.SIGTERM)
                plan['status'] = 'stopped'
                atomic_json(path, plan)
                return
            for j in plan['jobs']:
                if j['status'] not in ACTIVE:
                    continue
                output = Path(j['output'])
                if (output/'completed.json').exists():
                    result = json.loads((output/'completed.json').read_text())
                    assert result['updates'] == 995000*j['utd'] and result['steps'] == 1000000
                    j.update(status='completed', finished_unix=result['finished_unix'], progress=result)
                elif (output/'failure.json').exists() or not live(j):
                    error = json.loads((output/'failure.json').read_text()) if (output/'failure.json').exists() else dict(error='Exited without completion')
                    j.update(status='failed', error=error['error'], finished_unix=time.time())
                elif (output/'progress.json').exists():
                    j['progress'] = json.loads((output/'progress.json').read_text())
                    if j['progress'].get('updates', 0) > 0:
                        j['status'] = 'running'
            for pid, child in list(children.items()):
                if child.poll() is not None:
                    child.wait()
                    del children[pid]
            for task in plan['tasks']:
                for utd in (1, 2):
                    key = f'{task}-utd{utd}'
                    if key in plan['selections']:
                        continue
                    group = [j for j in plan['jobs'] if j['task'] == task and j['utd'] == utd and j['phase'] == 'search']
                    assert len(group) == 6
                    if any(j['status'] == 'failed' for j in group):
                        plan.setdefault('blocked_selections', {})[key] = 'Search seed failed; no silent seed replacement or selection from incomplete grid.'
                    elif all(j['status'] == 'completed' for j in group):
                        selection = score_temperature(group)
                        atomic_json(folder/'selections'/(key+'.json'), selection)
                        plan['selections'][key] = selection
                        plan['jobs'] += [make_job(plan, task, utd, selection['temperature'], seed, 'confirmation') for seed in (2, 3, 4)]
                        print('Selected '+key+' T='+str(selection['temperature']), flush=True)
            gpus, apps = inventory()
            active = [j for j in plan['jobs'] if j['status'] in ACTIVE]
            for j in plan['jobs']:
                if j['status'] != 'queued':
                    continue
                host_deficit = sum(max(0, plan['host_reserve_mib'][x['task']]-(process_info(x.get('pid', 0)) or {}).get('rss_mib', 0)) for x in active)
                if available_host_mib()-host_deficit-plan['host_reserve_mib'][j['task']] < plan['host_headroom_mib']:
                    j['waiting_reason'] = 'host memory reservation'
                    continue
                choices = []
                for gpu, info in gpus.items():
                    own = [x for x in active if x['gpu'] == gpu]
                    deficit = sum(max(0, plan['gpu_reserve_mib']-apps.get(x.get('pid'), 0)) for x in own)
                    projected = info['used']+deficit+plan['gpu_reserve_mib']
                    if projected <= info['total']-plan['gpu_headroom_mib'] and sum(x['status'] == 'initializing' for x in own) < plan['max_initializing_per_gpu']:
                        choices.append((projected/info['total'], gpu))
                if not choices:
                    j['waiting_reason'] = 'GPU memory / initialization reservation'
                    continue
                gpu = min(choices)[1]
                cpus = min(plan['cpu_lanes'], key=lambda c: sum(x.get('cpus') == c for x in active))
                spec = {k: j[k] for k in ('id', 'task', 'method', 'seed', 'utd', 'temperature', 'output', 'source_commit')}
                if not Path(j['spec']).exists():
                    atomic_json(j['spec'], spec)
                with (folder/'logs'/(j['id']+'.log')).open('x') as stream:
                    child = subprocess.Popen(command(plan, j['spec'], cpus), cwd=plan['source'],
                        env=environment(plan, gpu), stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                info = process_info(child.pid)
                j.update(status='initializing', gpu=gpu, cpus=cpus, pid=child.pid,
                    start_ticks=info['start_ticks'], started_unix=time.time())
                j.pop('waiting_reason', None)
                active.append(j)
                children[child.pid] = child
                print('Started '+j['id']+' GPU'+str(gpu), flush=True)
            plan.update(counts=dict(Counter(j['status'] for j in plan['jobs'])), checked_unix=time.time(), gpu_memory=gpus)
            atomic_json(path, plan)
            if not any(plan['counts'].get(s, 0) for s in (*ACTIVE, 'queued')):
                plan['status'] = 'completed_with_failures' if plan['counts'].get('failed') else 'completed'
                atomic_json(path, plan)
                break
            time.sleep(10)
    except Exception as error:
        plan.update(status='controller_error', error=repr(error))
        atomic_json(path, plan)
        raise


if __name__ == '__main__':
    main()
