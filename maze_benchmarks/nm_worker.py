"""Independent, locked GPU backfill for the bounded PointMaze N/M campaign."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .deadline_queue import command, verify, transaction, claim
from .run_nway_job import atomic_json, verify_source


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--commit', required=True)
    p.add_argument('--gpu', type=int, required=True)
    a = p.parse_args()
    source = Path(__file__).resolve().parents[1]
    verify_source(source, a.commit)
    # Existing common locks used by previous operators; no active worker eviction.
    lock_paths = [Path(f'/tmp/pointmaze-nm-gpu{a.gpu}.lock'),
                  Path(f'/tmp/ibolt-drac-pointmaze-gpu{a.gpu}.lock'),
                  Path(f'/home/heechan/OptiQ-ops/locks/gpu-{a.gpu}.lock')]
    locks = []
    for path in lock_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = path.open('a+')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
    memory = int(subprocess.check_output(['nvidia-smi', '-i', str(a.gpu),
                 '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
    if memory > 200:
        raise RuntimeError(f'GPU{a.gpu} is occupied ({memory}MiB); refusing to launch')
    while (job := claim(a.root, f'gpu{a.gpu}')) is not None:
        try:
            for preflight in (True, False):
                stage = 'preflights' if preflight else 'runs'
                out = a.root / stage / job['name']
                if out.exists():
                    raise FileExistsError(out)
                cmd = command(job, out, a.commit, preflight)
                cmd += ['--components', str(job['N']), '--candidates', str(job['M'])]
                if not preflight:
                    cmd += ['--wandb-project', 'pointmaze-NM-ablation']
                atomic_json(a.root / 'logs' / f"{job['name']}-{stage}-command.json", cmd)
                with (a.root / 'logs' / f"{job['name']}-{stage}.log").open('w') as log:
                    subprocess.run(cmd, cwd=source, check=True, stdout=log, stderr=subprocess.STDOUT)
                proof = verify(out, job, a.commit, preflight)
                cfg = json.loads((out/'config.json').read_text())
                actor = cfg['agent']['alg']['actor']
                assert actor['num_policy_samples'] == job['N']
                assert actor['num_reference_samples'] == job['M']
                assert cfg['agent']['experiment']['components'] == job['N']
                records = [json.loads(s) for s in (out/'training-metrics.jsonl').read_text().splitlines()]
                for row in records:
                    assert row['train/actual_component_count'] == job['N'], row
                    assert row['train/actual_candidate_count'] == job['M'], row
                atomic_json(a.root/'proofs'/f"{job['name']}-{stage}.json", proof)
            job.update(state='complete', finished=time.time())
        except Exception as exc:
            job.update(state='failed', error=repr(exc), finished=time.time())
        def update(status):
            for j in status['jobs']:
                if j['name'] == job['name']:
                    j.update(job)
        transaction(a.root, update)
        print(json.dumps(job), flush=True)


if __name__ == '__main__':
    main()
