from pathlib import Path
import hashlib
import json


ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def verify(root=ROOT):
    p = read(root / 'plan.json')
    d = read(root / 'DEPLOYMENT.json')
    assert len(d['commit']) == 40
    for rel, h in d['files'].items():
        assert sha(root / rel) == h, rel
    assert d['numerical_base_commit'] == p['source_commit']
    return p, d


def command(p, env, seed, commit, validation=False):
    assert env in p['order'] and seed in p['seeds']
    assert p['m'] % p['n'] == 0
    root = Path(p['root'])
    group = p['wandb_group'] + ('_validation' if validation else '')
    prefix = 'validation-' if validation else ''
    args = [p['python'], str(root / 'repo/run_optiq_dime.py'),
            '--config-name=' + p['config_name'], 'benchmark=' + env, f'seed={seed}',
            f'alg.actor.num_policy_samples={p["n"]}',
            f'alg.actor.proposals_per_policy_sample={p["m"] // p["n"]}',
            f'alg.actor.temperature={p["temperature"]}',
            'alg.actor.distillation_loss=direct_gmm_nll',
            'output_root=' + str(root / ('validation' if validation else 'outputs') / f'{env}_s{seed}'),
            f'run_name={prefix}{env}-DirectGMM-SingleQ-MC64-T{p["temperature"]}-N{p["n"]}-M{p["m"]}-s{seed}',
            f'total_steps={128 if validation else p["total_steps"]}',
            'wandb.activate=true', 'wandb.mode=online',
            'wandb.entity=' + p['wandb_entity'], 'wandb.project=' + p['wandb_project'],
            'wandb.group=' + group,
            'wandb.job_type=' + ('configuration-validation' if validation else 'direct-gmm-mujoco'),
            '+experiment_commit=' + commit]
    if validation:
        args += ['alg.learning_starts=32', 'alg.actor.learning_starts=32',
                 'eval_interval=128', 'num_eval_episodes=1', 'checkpoint_interval=64']
    return args
