"""Export verified small histogram records without transferring action archives."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    records = {}
    if args.out.exists():
        records = json.loads(args.out.read_text())['records']
    missing = []
    for task in json.loads((args.root / 'data/TASKS.json').read_text()):
        folder = args.root / 'runtime/early' / task['case'] / f'{task["method"]}_s{task["seed"]}'
        for step in [0, 10, 100, 1000]:
            key = f'{task["case"]}/{task["method"]}/{task["seed"]}/{step}'
            metric_file = folder / f'metrics_{step:06d}.json'
            file = folder / f'samples_{step:06d}.npz'
            if not metric_file.exists():
                missing.append(key)
                continue
            # Evaluations are immutable once their metrics file has been published.
            if key in records:
                continue
            metric = json.loads(metric_file.read_text())
            run = json.loads((folder / 'RUN.json').read_text())
            assert run['initial_parameter_sha256'] == task['initial_parameter_sha256']
            assert run['L'] == task['L']
            with np.load(file) as z:
                actions = z['actions'].ravel()
                edges = z['edges'].copy()
                mass = z['histogram_mass'].copy()
                count = len(actions)
                assert count == metric['sample_count'] == 2**20
                assert len(mass) == 512 and np.isclose(mass.sum(), 1)
                assert np.allclose(np.histogram(actions, edges)[0] / count, mass)
                assert np.isclose(.5 * np.abs(mass - z['target_mass']).sum(), metric['histogram_TV'])
                probe = {k: z[k].ravel()[:128].tolist() for k in ['actions', 'mu', 'sigma']}
            records[key] = dict(
                case=task['case'], method=task['method'], seed=task['seed'], step=step,
                file='login4:' + str(file), sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                count=count, edges=edges.tolist(), histogram_mass=mass.tolist(), probe=probe,
                metric=metric, source_commit=run['source_commit'], L=run['L'],
                initial_parameter_sha256=run['initial_parameter_sha256'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(time=time.time(), records=records, missing=missing)))
    tmp.replace(args.out)
    print(json.dumps(dict(records=len(records), missing=len(missing), out=str(args.out))))


if __name__ == '__main__':
    main()
