"""Select stable Simple variants before launching authorized Medium/Hard jobs."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

TRAINING_SOURCE = 'caf133d3ed4bdc8e4f48c029e75ed8f6450144ac'
SIMPLE_CAMPAIGN = 'pointmaze-simple-entropy-grid-1m-20260926'


def score(records):
    """Require two consecutive late checkpoints, not one lucky trajectory."""
    by_step = {r['step']: r for r in records}
    if not {800000, 1000192} <= set(by_step):
        return None
    summaries = [by_step[s]['policy'] for s in (800000, 1000192)]
    fractions = []
    for s in summaries:
        if len(s['goals']) != 4 or sum(s['goals']) + s['failure'] != s['episodes']:
            raise ValueError('invalid Simple evaluation accounting')
        p = [c / s['episodes'] for c in s['goals']]
        if sum(p) < .9 or min(p) < .05:
            return None
        fractions.append(p)
    return (min(min(p) for p in fractions),
            min(-sum(x * math.log(x / sum(p)) for x in p) / sum(p) for p in fractions),
            min(sum(p) for p in fractions))


def main():
    from .deadline_queue import prepare, verify, OPS
    from .run_nway_job import atomic_json, verify_source
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--controller-commit', required=True)
    args = parser.parse_args()
    verify_source(args.source, TRAINING_SOURCE)
    args.output.mkdir(parents=True, exist_ok=False)
    state = dict(status='watching', controller_source=args.controller_commit,
                 training_source=TRAINING_SOURCE, methods={})
    atomic_json(args.output / 'status.json', state)
    try:
        while len(state['methods']) < 2:
            queue = json.loads((args.parent / 'queue.json').read_text())
            if queue['source_commit'] != TRAINING_SOURCE:
                raise ValueError('parent source differs')
            if queue['state'] not in ('running', 'complete'):
                raise RuntimeError('parent paused or failed; no new jobs authorized to start')
            for method, gpus in (('mfpo', [0, 2]), ('meow', [4, 5])):
                if method in state['methods']:
                    continue
                jobs = [j for j in queue['jobs'] if j['method'] == method]
                if len(jobs) != 3:
                    raise ValueError('unexpected sensitivity grid')
                if any(j['state'] != 'complete' for j in jobs):
                    continue
                checked = []
                for job in jobs:
                    run = args.parent / 'runs' / job['name']
                    proof = json.loads((args.parent / 'proofs' / f"{job['name']}-runs.json").read_text())
                    if verify(run, job, TRAINING_SOURCE, False) != proof:
                        raise ValueError('source/config/raw evaluation/SHA proof mismatch')
                    records = [json.loads(p.read_text()) for p in sorted((run / 'evaluations').glob('*_summary.json'))]
                    checked.append(dict(job=job, score=score(records),
                                        evaluations=[r for r in records if r['step'] in (800000, 1000192)]))
                passing = [c for c in checked if c['score'] is not None]
                selection = dict(method=method, candidates=checked,
                    criterion='both 800k and1M: success>=90%, every goal>=5% of all episodes',
                    chosen=max(passing, key=lambda c: c['score'])['job']['name'] if passing else None)
                atomic_json(args.output / f'selection-{method}.json', selection)
                if not passing:
                    state['methods'][method] = dict(status='no_qualifying_configuration')
                    atomic_json(args.output / 'status.json', state)
                    continue
                winner = max(passing, key=lambda c: c['score'])['job']
                followups = []
                for maze in ('medium', 'hard'):
                    job = {k: winner[k] for k in ('name', 'task', 'method', 'temperature', 'host', 'entropy_diagnostics')}
                    job['name'] = winner['name'].replace('pm_simple-', f'pm_{maze}-')
                    job['task'] = f'pm_{maze}'
                    key = 'meow_alpha' if method == 'meow' else 'mfpo_target_entropy_per_dim'
                    job[key] = winner[key]
                    followups.append(job)
                campaign = f'pointmaze-entropy-selected-{method}-1m-20260926'
                root = args.parent.parent / campaign
                plan_path = args.output / f'plan-{method}.json'
                atomic_json(plan_path, dict(campaign=campaign, jobs=followups,
                    selection=selection['chosen'], controller_source=args.controller_commit))
                prepare(root, TRAINING_SOURCE, 'vast-heechan-46', args.source,
                    plan_path=plan_path, campaign=campaign, gpus=gpus)
                # Persist preparation before external launch. No retries after an uncertain failure.
                state['methods'][method] = dict(status='prepared', root=str(root), selected=winner['name'])
                atomic_json(args.output / 'status.json', state)
                subprocess.run([sys.executable, '-m', 'maze_benchmarks.deadline_queue',
                    '--phase', 'start', '--root', str(root), '--source-commit', TRAINING_SOURCE,
                    '--training-source', str(args.source), '--host', 'vast-heechan-46'], check=True)
                state['methods'][method]['status'] = 'launched'
                atomic_json(args.output / 'status.json', state)
            if len(state['methods']) < 2:
                time.sleep(30)
        state['status'] = 'selection_complete'
        atomic_json(args.output / 'status.json', state)
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        atomic_json(args.output / 'failure.json', state)
        raise


if __name__ == '__main__':
    main()
