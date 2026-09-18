"""Release all pending non-Ant arrays; express preference with Nice only."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

CAMPAIGNS = [
    ('/lustre/hobbit9882/OptiQ-DirectGMM-128x256-20260917',
     'ec9bb32b424a37147ea6f3dacf08e9315f28a622',
     [('hopper', 2277775, 10), ('walker2d', 2277776, 20), ('halfcheetah', 2277777, 30)]),
    ('/lustre/hobbit9882/OptiQ-DirectGMM-128x256-T025-20260917',
     '23f226e0da65cf493afe27b9087a364d534a81cd',
     [('hopper', 2277874, 10), ('walker2d', 2277875, 20), ('halfcheetah', 2277876, 30)]),
]


def show(job):
    return subprocess.check_output(['scontrol', 'show', 'job', '-o', str(job)], text=True).strip()


def write(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)


def main(commit, apply):
    assert re.fullmatch('[0-9a-f]{40}', commit)
    prepared = []
    for location, source_commit, targets in CAMPAIGNS:
        root = Path(location)
        assert json.loads((root / 'DEPLOYMENT.json').read_text())['commit'] == source_commit
        subprocess.run(['/scratch2/hobbit9882/.venvs/optiq-mujoco-scalar/bin/python',
                        '-c', 'from common import verify; verify()'], cwd=root, check=True)
        path = root / 'PRIORITY_ONLY_20260918.json'
        assert not path.exists(), 'Operation already attempted; inspect record before retry.'
        changes = []
        for env, job, nice in targets:
            before = show(job)
            assert 'JobState=PENDING ' in before
            assert f'-{env} ' in before
            assert not list((root / 'outputs').glob(f'{env}_s*'))
            changes.append(dict(env=env, array=job, before=before, state='validated',
                command=['scontrol', 'update', f'JobId={job}', 'Dependency=', f'Nice={nice}']))
        record = dict(source_commit=source_commit, operation_commit=commit,
                      operation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      ant_unchanged=True, humanoid_unchanged=True, changes=changes,
                      validated_unix=time.time())
        prepared.append((path, record))
    if not apply:
        print('Validated six arrays / 24 pending runs; no changes applied.')
        return
    # Interleave temperatures within each environment, preserving preference.
    for path, record in prepared:
        write(path, record)
    for index in range(3):
        for path, record in prepared:
            change = record['changes'][index]
            result = subprocess.run(change['command'], capture_output=True, text=True)
            change.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                          changed_unix=time.time(), state='released' if result.returncode == 0 else 'failed')
            write(path, record)
            result.check_returncode()
            change['after'] = show(change['array'])
            write(path, record)
            dependencies = re.findall(r'\bDependency=(\S+)', change['after'])
            assert dependencies and all(d in ['(null)', ''] for d in dependencies)
            assert all(int(n) == int(change['command'][-1].split('=')[1])
                       for n in re.findall(r'\bNice=(\d+)', change['after']))
            print(f"Released {change['array']} {change['env']}: dependency removed, {change['command'][-1]}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--operation-commit', required=True)
    p.add_argument('--apply', action='store_true')
    args = p.parse_args()
    main(args.operation_commit, args.apply)
