"""Fresh replacements; reuse immutable, previously GPU-validated source."""
import argparse
import importlib.util
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback

BUNDLE = Path(__file__).resolve().parent
ROOT = BUNDLE.parents[1]  # campaign/operations/20260918_humanoid_fresh


def common():
    spec = importlib.util.spec_from_file_location('campaign_common', ROOT / 'common.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def context():
    c = common()
    p, original = c.verify(ROOT)
    deployment = c.read(BUNDLE / 'DEPLOYMENT.json')
    assert len(deployment['commit']) == 40
    for rel, digest in deployment['files'].items():
        assert c.sha(BUNDLE / rel) == digest, rel
    gate = c.read(ROOT / 'VALIDATION.json')
    assert gate['passed'] and gate['commit'] == original['commit']
    return c, p, original, deployment


def paths(seed):
    assert seed in range(4)
    name = f'humanoid_fresh_20260918_s{seed}'
    return ROOT / 'outputs' / name, ROOT / 'status' / (name + '.json')


def command(c, p, original, deployment, seed):
    cmd = c.command(p, 'humanoid', seed, deployment['commit'])
    out, _ = paths(seed)
    cmd = ['output_root=' + str(out) if x.startswith('output_root=') else
           x + '-fresh2' if x.startswith('run_name=') else x for x in cmd]
    old_id = c.read(BUNDLE / 'replacements.json')[str(p['temperature'])][seed]
    cmd += ['+restart_of_wandb_run=' + old_id,
            '+original_experiment_commit=' + original['commit'], '+restart_kind=fresh']
    assert not any('resume' in x or 'load_model' in x for x in cmd)
    return cmd, old_id


def preflight():
    from hydra import initialize_config_dir, compose
    from omegaconf import OmegaConf
    import wandb
    c, p, original, d = context()
    records = []
    api = wandb.Api(timeout=30)
    for seed in range(4):
        cmd, old_id = command(c, p, original, d, seed)
        with initialize_config_dir(version_base=None, config_dir=str(ROOT / 'repo/configs')):
            cfg = compose(config_name=p['config_name'], overrides=cmd[3:])
        resolved = OmegaConf.to_container(cfg, resolve=True)
        previous = c.read(ROOT / 'resolved_configs' / f'humanoid_s{seed}.json')
        ignored = {'output_root', 'run_name', 'experiment_commit', 'restart_of_wandb_run',
                   'original_experiment_commit', 'restart_kind'}
        assert {k:v for k,v in resolved.items() if k not in ignored} == {
            k:v for k,v in previous.items() if k not in ignored}, 'Learning settings changed'
        assert cfg.total_steps == 1000000 and cfg.alg.learning_starts == 5000
        assert cfg.wandb.mode == 'online' and cfg.wandb.activate
        old = api.run(f'{p["wandb_entity"]}/{p["wandb_project"]}/{old_id}')
        assert old.state == 'crashed', (old_id, old.state)
        assert old.config['seed'] == seed and old.config['env_name'] == cfg.env_name
        assert old.config['alg']['actor']['temperature'] == p['temperature']
        assert not paths(seed)[0].exists(), 'Fresh output already exists'
        dest = BUNDLE / 'resolved_configs' / f'seed{seed}.json'
        c.write(dest, resolved)
        records.append(dict(seed=seed, replaced_id=old_id, config_sha256=c.sha(dest)))
    c.write(BUNDLE / 'PREFLIGHT.json', dict(passed=True, commit=d['commit'],
            original_commit=original['commit'], records=records, time=time.time()))
    print('Preflight passed: four fresh Humanoid replacements; learning settings identical.', flush=True)


def launch(seed):
    c, p, original, d = context()
    gate = c.read(BUNDLE / 'PREFLIGHT.json')
    assert gate['passed'] and gate['commit'] == d['commit']
    cmd, old_id = command(c, p, original, d, seed)
    out, status = paths(seed)
    out.mkdir(parents=True, exist_ok=False)
    record = dict(state='running', restart_kind='fresh', env='humanoid', seed=seed,
                  temperature=p['temperature'], replaced_wandb_id=old_id,
                  commit=d['commit'], original_commit=original['commit'],
                  source_code_id=p['source_code_id'], command=cmd,
                  job_id=os.environ.get('SLURM_JOB_ID'),
                  array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),
                  array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'),
                  node=socket.gethostname(), started_unix=time.time())
    c.write(status, record)
    c.write(out / 'RUN_PROVENANCE.json', record)
    env = os.environ.copy()
    for key in ['WANDB_RUN_ID', 'WANDB_RESUME', 'WANDB_NAME', 'WANDB_PROJECT', 'WANDB_RUN_GROUP']:
        env.pop(key, None)
    env.update(WANDB_ENTITY=p['wandb_entity'], WANDB_MODE='online', WANDB_RESUME='never')
    try:
        subprocess.run(cmd, cwd=ROOT / 'repo', env=env, check=True)
        markers = list(out.glob('*/completed.json'))
        assert len(markers) == 1
        result = c.read(markers[0])
        assert result['timesteps'] == p['total_steps']
        assert result['wandb_url'].rstrip('/').split('/')[-1] != old_id
        actual = markers[0].parent
        manifest = {str(f.relative_to(actual)):c.sha(f) for f in actual.rglob('*')
                    if f.is_file() and not f.is_symlink() and 'wandb' not in f.relative_to(actual).parts}
        c.write(actual / 'ARTIFACTS_SHA256.json', manifest)
        record.update(state='complete', finished_unix=time.time(),
                      wandb_url=result['wandb_url'], timesteps=result['timesteps'], updates=result['updates'])
        c.write(status, record)
    except Exception:
        record.update(state='failed', finished_unix=time.time(), error=traceback.format_exc())
        c.write(status, record)
        raise


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['preflight', 'run'])
    ap.add_argument('--seed', type=int)
    args = ap.parse_args()
    preflight() if args.mode == 'preflight' else launch(args.seed)
