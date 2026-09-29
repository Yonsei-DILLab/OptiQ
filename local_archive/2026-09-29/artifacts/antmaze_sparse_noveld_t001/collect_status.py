"""Collect this campaign without changing or restarting server jobs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone

import numpy as np

HOST = 'vast-heechan-180'
CAMPAIGN = 'antmaze-sparse-noveld-t001-optiq-s0-20260923'
SOURCE = '8bb0c50a1356dc082106393f161ff3fc3d2de0af'
ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-first-evaluation', action='store_true')
    args = parser.parse_args()
    destination = ROOT / HOST
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        'rsync', '-az', '--exclude=wandb/', '--exclude=learner/',
        '--exclude=checkpoint*.pt', '--exclude=*.tmp',
        f'{HOST}:/home/heechan/optiq-experiments/{CAMPAIGN}/', str(destination) + '/',
    ], check=True)
    manifest = read(destination / 'manifest.json')
    assert manifest['source_commit'] == SOURCE
    status = read(destination / 'status.json')
    results = []
    for job in manifest['jobs']:
        run = destination / 'runs' / job['id']
        config = read(run / 'config.json')
        actor = config['native']['alg']['actor']
        assert config['source_commit'] == SOURCE
        assert config['reward_profile'] == 'sparse' and config['noveld_enabled']
        assert config['noveld_coefficient'] == actor['temperature'] == 0.01
        assert [actor[k] for k in ('log_std_min', 'log_std_max', 'initial_log_std')] == [-5, -1, -1]
        assert config['interim_eval_episodes'] == 40 and config['save_intermediate_policy']
        progress = read(run / 'progress.json')
        assert progress['updates'] == progress['rnd_updates'] > 0
        proofs = []
        for path in sorted(run.glob('policy-checkpoints/*/verification.json')):
            proof = read(path)
            data = (run / proof['path']).read_bytes()
            assert hashlib.sha256(data).hexdigest() == proof['sha256']
            assert len(data) == proof['bytes'] and proof['source_commit'] == SOURCE
            assert proof['readback_verified'] and proof['restored_policy_state_exact']
            proofs.append(proof['step'])
        first = run / 'evaluations' / '0000250112'
        evaluations = []
        for mode in ('native', 'policy'):
            folder = first / f'{mode}-natural'
            if not (folder / 'summary.json').exists():
                assert not args.require_first_evaluation, (job['id'], mode, 'not yet complete')
                continue
            summary = read(folder / 'summary.json')
            assert summary['episodes'] == 40 and not summary['fixed']
            with np.load(folder / 'rollouts.npz') as raw:
                assert len(raw['xy']) == len(raw['returns']) == len(raw['initial_full_state']) == 40
                assert np.isfinite(raw['returns']).all() and np.isfinite(raw['initial_full_state']).all()
                unique_starts = len(np.unique(raw['initial_full_state'], axis=0))
                assert unique_starts == 40
                for xy, length in zip(raw['xy'], raw['lengths']):
                    assert np.isfinite(xy[:int(length) + 1]).all()
                expected = np.where(raw['goals'] > 0, 10.0, 0.0)
                if job['task'] == 'v2':
                    expected = np.where(raw['goals'] == 1, 20.0, expected)
                assert np.allclose(raw['returns'], expected)
            evaluations.append(dict(mode=mode, episodes=40, unique_starts=unique_starts,
                                    success_rate=summary['success_rate'], raw_verified=True))
        if args.require_first_evaluation:
            assert 250112 in proofs and len(evaluations) == 2
        results.append(dict(id=job['id'], progress=progress, verified_policy_steps=proofs,
                            first_evaluations=evaluations, wandb=read(run / 'wandb.json')))
    report = dict(collected_at=datetime.now(timezone.utc).isoformat(), source_commit=SOURCE,
                  host=HOST, failed=status['failed'], pending=status['pending'], runs=results,
                  local_data='metadata, logs, raw evaluations and verified intermediate policy checkpoints; full replay checkpoints remain remote')
    (ROOT / 'launch-verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
