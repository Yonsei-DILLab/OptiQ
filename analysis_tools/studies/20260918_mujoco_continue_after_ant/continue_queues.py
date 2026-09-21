"""Clear only the two explicitly authorized Ant dependencies; preserve source."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

CAMPAIGNS = [
    ('/lustre/hobbit9882/OptiQ-DirectGMM-128x256-20260917',
     'ec9bb32b424a37147ea6f3dacf08e9315f28a622',
     [2277773, 2277774, 2277775, 2277776, 2277777]),
    ('/lustre/hobbit9882/OptiQ-DirectGMM-128x256-T025-20260917',
     '23f226e0da65cf493afe27b9087a364d534a81cd',
     [2277872, 2277873, 2277874, 2277875, 2277876]),
]


def show(job):
    return subprocess.check_output(['scontrol', 'show', 'job', '-o', str(job)], text=True).strip()


def write(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def main(commit, apply):
    assert len(commit) == 40 and all(c in '0123456789abcdef' for c in commit)
    # Read/validate both campaigns before either scheduler mutation.
    ready = []
    for location, source_commit, jobs in CAMPAIGNS:
        root = Path(location)
        meta = json.loads((root / 'DEPLOYMENT.json').read_text())
        assert meta['commit'] == source_commit
        gate = json.loads((root / 'VALIDATION.json').read_text())
        assert gate['passed'] and gate['commit'] == source_commit
        subprocess.run(['/scratch2/hobbit9882/.venvs/optiq-mujoco-scalar/bin/python',
                        '-c', 'from common import verify; verify()'], cwd=root, check=True)
        before = {str(job): show(job) for job in jobs[1:]}
        for previous, current in zip(jobs, jobs[1:]):
            info = before[str(current)]
            assert 'JobState=PENDING ' in info, info
            assert f'Dependency=afterok:{previous}' in info, info
        assert not any((root / 'outputs').glob('humanoid_s*'))
        event = root / 'CONTINUATION_20260918.json'
        assert not event.exists(), 'Continuation already attempted; inspect its record before retrying.'
        record = dict(source_commit=source_commit, operation_commit=commit,
                      operation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      reason='User requested leaving interrupted Ant seeds untouched and continuing later environments.',
                      ant_array_unchanged=jobs[0], before=before,
                      command=['scontrol', 'update', f'JobId={jobs[1]}', 'Dependency='],
                      state='validated', validated_unix=time.time())
        ready.append((event, record, jobs))
    if not apply:
        print('Validated both source snapshots and eight queued arrays; no scheduler changes.')
        return
    for event, record, jobs in ready:
        write(event, record)
        result = subprocess.run(record['command'], capture_output=True, text=True)
        record.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                      state='released' if result.returncode == 0 else 'failed', changed_unix=time.time())
        write(event, record)
        result.check_returncode()
        record['after'] = {str(job): show(job) for job in jobs[1:]}
        write(event, record)
        assert f'Dependency=afterok:{jobs[0]}' not in record['after'][str(jobs[1])]
        for previous, current in zip(jobs[1:], jobs[2:]):
            assert f'Dependency=afterok:{previous}' in record['after'][str(current)]
        print(f'Released Humanoid {jobs[1]}; downstream dependencies preserved.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--operation-commit', required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    main(args.operation_commit, args.apply)
