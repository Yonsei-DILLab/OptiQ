"""Read-only, standard-library collector for frozen Adroit curve snapshots.

Run on an experiment host and redirect stdout to an analysis snapshot. No learner
modules are imported and no files on the experiment host are changed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time


def collect(root):
    result = dict(observed_start_unix=time.time(), root=str(root), runs={}, status={})
    queue_path = root / 'queue-state.json'
    if (root / 'priority-queue-state.json').exists():
        queue_path = root / 'priority-queue-state.json'
    result['queue'] = json.loads(queue_path.read_text())
    recovery = root / 'recovery-queue-state.json'
    replacements = {}
    if recovery.exists():
        result['recovery'] = json.loads(recovery.read_text())
        for name, state in result['recovery']['jobs'].items():
            result['queue']['jobs'][name] = state.copy()
            if (Path(state['output']) / 'spec.json').exists():
                replacements[name] = state['attempt_id']
    if (root / 'queue-delegation.json').exists():
        result['delegation'] = json.loads((root / 'queue-delegation.json').read_text())
        for name, destination in result['delegation']['jobs'].items():
            state = result['queue']['jobs'][name]
            assert state['status'] in ('pending', 'migrated')
            state.update(status='migrated', destination=destination)
    for run in sorted((root / 'runs').glob('*')):
        if run.name in replacements:
            continue  # Failed attempts remain on disk; never splice their curves.
        if not (run / 'spec.json').exists():
            continue
        raw_spec = (run / 'spec.json').read_bytes()
        spec = json.loads(raw_spec)
        logical_name = next((n for n, attempt in replacements.items() if attempt == run.name), run.name)
        row = dict(spec=spec, spec_sha256=hashlib.sha256(raw_spec).hexdigest())
        if logical_name != run.name:
            row.update(attempt_id=run.name, logical_run_id=logical_name)
        for filename in ('progress.json', 'completed.json', 'error.json'):
            if (run / filename).exists():
                row[filename] = json.loads((run / filename).read_text())
        pid = spec.get('pid')
        row['process_present'] = bool(pid and (Path('/proc') / str(pid)).exists())
        result['status'][logical_name] = row.copy()
        if spec['profile'] != 'e256-u025':
            continue
        row['evaluations'] = []
        for path in sorted((run / 'evaluations').glob('*.json')):
            if path.name.endswith('-uniform-random.json'):
                continue
            raw = path.read_bytes()
            evaluation = json.loads(raw)
            assert evaluation['kind'] in ('initial', 'regular', 'final')
            count = evaluation['episodes']
            record = {k: evaluation[k] for k in ('step', 'updates', 'episodes', 'primary', 'kind')}
            record.update(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), modes={},
                          reset_seeds_sha256=hashlib.sha256(json.dumps(evaluation['reset_seeds']).encode()).hexdigest())
            for mode, values in evaluation.items():
                if not isinstance(values, dict) or 'episode_records' not in values:
                    continue
                episodes = values['episode_records']
                assert len(episodes) == count
                for ep in episodes:
                    assert bool(ep['success_any']) == (ep['success_steps'] > 0)
                    assert bool(ep['success_25_steps']) == (ep['success_steps'] >= 25)
                    assert 0 <= ep['success_steps'] <= spec['horizon']
                metrics = {}
                for metric in ('success_any', 'success_25_steps', 'success_final'):
                    successes = sum(bool(ep[metric]) for ep in episodes)
                    assert abs(values[metric] - successes / count) < 1e-12
                    metrics[metric] = successes / count
                    metrics[metric + '_count'] = successes
                metrics['mean_success_steps'] = sum(ep['success_steps'] for ep in episodes) / count
                record['modes'][mode] = metrics
            assert record['primary'] in record['modes']
            row['evaluations'].append(record)
        result['runs'][logical_name] = row
    result['observed_unix'] = time.time()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    print(json.dumps(collect(args.root)))
