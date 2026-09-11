"""Read-only final-v2 release/config/provenance verification; no training or W&B."""
import hashlib
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def verify_configurations(root):
    reference = json.loads((root/'docs/v2/REFERENCE_CONFIG.json').read_text())
    before = json.loads((root/'tests/data/v2_pre_final_configs.json').read_text())
    archives = {'mujoco_v2':'original', 'mujoco_v2_conditional':'conditional',
                'mujoco_v2_guarded':'guarded', 'mujoco_v2_finite':'finite',
                'mujoco_v2_proximal':'proximal'}
    with initialize_config_dir(config_dir=str(root/'configs'),version_base=None):
        for name in ('mujoco_v2','mujoco_v2_checked','v2/final'):
            config = compose(config_name=name)
            assert OmegaConf.to_container(config.alg,resolve=True) == reference['alg'], name
        for old, new in archives.items():
            config = compose(config_name='archive/v2/'+new)
            assert OmegaConf.to_container(config,resolve=True) == before[old], old


def verify_launch_sources(root, expected):
    """Keep frozen run audits valid across this explicitly recorded config move.

    Never relax executable-source hashes. For changed config bytes, require the
    exact before hash, the entire recorded after config tree, and resolved value
    parity. Evaluation itself restores each run's saved config.json.
    """
    changed = {}
    for name, wanted in expected.items():
        actual = hashlib.sha256((root/name).read_bytes()).hexdigest()
        if actual != wanted:
            assert name.startswith('configs/'), f'Learning source changed: {name}'
            changed[name] = wanted
    if changed:
        migration = json.loads((root/'docs/v2/CONFIG_MIGRATION.json').read_text())
        for name, wanted in changed.items():
            assert migration['before_sha256'].get(name) == wanted, name
        for name, wanted in migration['after_sha256'].items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest() == wanted, name
        verify_configurations(root)
    return sorted(changed)


def main():
    manifest = json.loads((ROOT/'docs/v2/MANIFEST.json').read_text())
    for name, expected in manifest['common_source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
    verify_configurations(ROOT)
    print('PASS: common learning sources unchanged; final aliases and five archives verified.')
    print('Final algorithm:', manifest['algorithm_id'])
    print('Completed training source:', manifest['training_commit'])


if __name__ == '__main__':
    main()
