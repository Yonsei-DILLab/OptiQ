"""Read-only collection and launch verification for the approved four jobs."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
CAMPAIGN = 'antmaze-optiq-ddiffpg-lr-256x3-s0-20260923-r3'
SOURCE = 'cac1365fd9878d46874bbc09a840816ba1461499'
HOST = 'vast-heechan-180'


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--require-training', action='store_true')
    args = parser.parse_args()
    target = ROOT/HOST
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(['rsync', '-az', '--exclude=wandb/', '--exclude=learner/',
        '--exclude=checkpoint*.pt', '--exclude=*.tmp',
        f'{HOST}:/home/heechan/optiq-experiments/{CAMPAIGN}/', str(target)+'/' ], check=True)
    manifest, status = read(target/'manifest.json'), read(target/'status.json')
    assert manifest['source_commit'] == status['source_commit'] == SOURCE
    assert manifest['wandb_project'] == 'antmaze'
    assert len(manifest['jobs']) == 4 and not status['failed']
    rows = []
    for job in manifest['jobs']:
        row = {'id':job['id'], 'job':read(target/'jobs'/f'{job["id"]}.json')}
        for phase in ('preflight','runs'):
            run = target/phase/job['id']
            if not (run/'config.json').exists():
                assert not args.require_training, (job['id'],phase,'not initialized')
                continue
            cfg = read(run/'config.json'); native = cfg['native']['alg']
            assert cfg['source_commit'] == SOURCE
            assert native['actor']['hidden_dims'] == native['critic']['hs'] == [256,256,256]
            assert native['optimizer']['lr_actor'] == 3e-4 and native['optimizer']['lr_critic'] == 5e-4
            assert native['tau'] == .005 and cfg['temperature'] == .01
            assert cfg['num_envs'] == 256 and cfg['batch_size'] == 4096 and cfg['updates_per_vector_step'] == 8
            assert cfg['reward_profile'] == 'sparse' and cfg['noveld_coefficient'] == .01
            assert cfg['interim_eval_episodes'] == 40 and cfg['final_eval_episodes'] == 100
            assert cfg['save_intermediate_policy'] and cfg['eval_starts'] == 'random'
            assert cfg['wandb_project'] == 'antmaze'
            profile = read(run/'optiq-profile-verification.json')
            assert profile['verified'] and profile['rnd_lrs'] == [1e-4]
            row[phase] = {'profile':profile}
            if (run/'progress.json').exists():
                progress = read(run/'progress.json')
                assert progress['updates'] == progress['rnd_updates']
                row[phase]['progress'] = progress
            if phase == 'preflight' and (run/'result.json').exists():
                result = read(run/'result.json')
                assert result['completed'] and result['updates'] == result['rnd_updates'] == 8
                assert result['checkpoint']['environment_reward_verified']
                assert read(run/'parameter-audit.json')['passed']
                row[phase]['completed_verified'] = True
                for proof in (run/'policy-checkpoints').glob('*/verification.json'):
                    verification = read(proof)
                    assert verification['source_commit'] == SOURCE
                    assert verification['restored_policy_state_exact']
                    assert hashlib.sha256((run/verification['path']).read_bytes()).hexdigest() == verification['sha256']
            if phase == 'runs' and (run/'wandb.json').exists(): row['wandb'] = read(run/'wandb.json')
        if args.require_training:
            assert row['preflight']['completed_verified']
            assert row['runs']['progress']['updates'] > 0
            assert row['job']['phase'] == 'runs'
        rows.append(row)
    out = dict(collected_at=datetime.now(timezone.utc).isoformat(),source_commit=SOURCE,
               campaign=CAMPAIGN,status=status,runs=rows)
    (ROOT/'launch-verification.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'campaign':CAMPAIGN,'failed':status['failed'],
        'runs':[{'id':r['id'],'phase':r['job'].get('phase'),
                 'progress':r.get('runs',{}).get('progress'), 'wandb':r.get('wandb')} for r in rows]},indent=2))


if __name__ == '__main__':
    main()
