"""One SLURM task per seed. No automatic partial-checkpoint restart."""
import argparse
import csv
import math
import os
import socket
import subprocess
import time
import traceback
from common import ROOT, read, write, sha, verify, command


def launch(env, seed, validation=False):
    p, d = verify()
    if not validation:
        gate = read(ROOT / 'VALIDATION.json')
        assert gate['passed'] and gate['commit'] == d['commit']
    assert read(ROOT / 'PREFLIGHT.json')['passed']
    name = f'{env}_s{seed}'
    out = ROOT / ('validation' if validation else 'outputs') / name
    state = ROOT / 'status' / (('validation_' if validation else '') + name + '.json')
    out.mkdir(parents=True, exist_ok=False)
    cmd = command(p, env, seed, d['commit'], validation)
    record = dict(state='running', env=env, seed=seed, validation=validation,
                  commit=d['commit'], source_code_id=p['source_code_id'], command=cmd,
                  job_id=os.environ.get('SLURM_JOB_ID'), array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),
                  array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'), node=socket.gethostname(),
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), started_unix=time.time())
    write(state, record)
    write(out / 'RUN_PROVENANCE.json', record)
    process_env = os.environ.copy()
    # Each subprocess creates its own W&B run. Authentication remains in ~/.netrc.
    for key in ['WANDB_RUN_ID', 'WANDB_RESUME', 'WANDB_NAME', 'WANDB_PROJECT', 'WANDB_RUN_GROUP']:
        process_env.pop(key, None)
    process_env.update(WANDB_ENTITY=p['wandb_entity'], WANDB_MODE='online')
    try:
        subprocess.run(cmd, cwd=ROOT / 'repo', env=process_env, check=True)
        markers = list(out.glob('*/completed.json'))
        assert len(markers) == 1
        result = read(markers[0])
        assert result['timesteps'] == (128 if validation else p['total_steps'])
        actual = markers[0].parent
        if validation:
            assert result['updates'] == 96
            with (actual / 'logs/progress.csv').open() as f:
                rows = list(csv.DictReader(f))
            for field in ['train/actor_loss', 'train/critic_loss']:
                values = [float(r[field]) for r in rows if r.get(field)]
                assert values and all(math.isfinite(v) for v in values), field
            import wandb
            api = wandb.Api(timeout=30)
            run_id = result['wandb_url'].rstrip('/').split('/')[-1]
            remote = None
            for attempt in range(10):
                api.flush()
                remote = api.run(f'{p["wandb_entity"]}/{p["wandb_project"]}/{run_id}')
                if remote.summary.get('completed') and remote.summary.get('timesteps') == 128:
                    break
                time.sleep(3)
            assert remote.summary.get('completed') and remote.summary.get('timesteps') == 128
            assert remote.config['alg']['actor']['num_policy_samples'] == 128
            assert remote.config['alg']['actor']['proposals_per_policy_sample'] == 2
        manifest = {str(f.relative_to(actual)): sha(f) for f in actual.rglob('*')
                    if f.is_file() and not f.is_symlink() and 'wandb' not in f.relative_to(actual).parts}
        write(actual / 'ARTIFACTS_SHA256.json', manifest)
        record.update(state='complete', finished_unix=time.time(), wandb_url=result['wandb_url'],
                      timesteps=result['timesteps'], updates=result['updates'])
        write(state, record)
        return record
    except Exception:
        record.update(state='failed', finished_unix=time.time(), error=traceback.format_exc())
        write(state, record)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['validate', 'run'])
    parser.add_argument('--env')
    parser.add_argument('--seed', type=int)
    a = parser.parse_args()
    if a.mode == 'validate':
        records = [launch(env, 0, True) for env in ['ant', 'humanoid']]
        write(ROOT / 'VALIDATION.json', dict(passed=True, commit=read(ROOT / 'DEPLOYMENT.json')['commit'],
                                            records=records, checked_unix=time.time()))
        print('GPU validation passed: Ant and Humanoid, actual batch256/N128/M256, finite losses and online W&B.')
    else:
        launch(a.env, a.seed)
