"""Canonical local launch paths must not select the archived Gaussian-W2 variant."""

import os
from pathlib import Path
import subprocess

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def test_default_output_and_wandb_paths():
    with initialize_config_dir(config_dir=str(ROOT / 'configs'), version_base=None):
        for name in ('mujoco_v2', 'mujoco_v2_checked', 'v2/final'):
            cfg = compose(config_name=name)
            assert (ROOT / cfg.output_root).resolve() == ROOT.parent / 'optiq-experiments/v2_checked64/outputs'
            assert cfg.wandb.entity == 'OptiQ'
            assert cfg.wandb.project == 'optiq_mujoco_v2_confirmation'
            assert cfg.alg.actor.distillation_loss == 'conditional_ot_nll'
            assert cfg.alg.actor.soft_guard.enabled
            assert cfg.alg.critic.use_layer_norm is False
            assert cfg.alg.optimizer.ac_grad_norm == 2


def test_launcher_rejects_implicit_historical_config():
    env = {**os.environ, 'OPTIQ_PYTHON': '/bin/true', 'OPTIQ_CONFIG': 'archive/v2/original'}
    result = subprocess.run(['bash', str(ROOT / 'scripts/run_v2.sh'), '0', '--check'],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert 'reserved for checked-K64' in result.stderr


def test_launcher_has_no_cross_checkout_credentials():
    source = (ROOT / 'scripts/run_v2.sh').read_text()
    assert '/workspace/OptiQ' not in source
    assert 'WANDB_RUN_ID WANDB_RESUME' in source
    assert 'scripts/verify_v2_final.py' in source


def test_managed_worker_uses_canonical_launcher_without_automatic_start():
    wrapper = (ROOT / 'scripts/supervisor_v2_checked64.sh').read_text()
    config = (ROOT / 'deploy/supervisor/optiq-v2-checked64.conf').read_text()
    assert 'cd /root/OptiQ' in wrapper
    assert 'scripts/run_v2.sh' in wrapper
    assert 'OPTIQ_CONFIG=mujoco_v2' in wrapper
    assert '. "${utils}/logging.sh" ""' in wrapper
    assert 'autostart=false' in config
    assert 'autorestart=false' in config
    assert 'numprocs=4' in config


def test_no_gaussian_w2_default_or_shadow_module():
    assert not (ROOT / 'optiq_dime/gaussian_teacher.py').exists()
    assert not (ROOT / 'configs/mujoco_v2_gaussian.yaml').exists()
    with initialize_config_dir(config_dir=str(ROOT / 'configs'), version_base=None):
        cfg = OmegaConf.to_container(compose(config_name='mujoco_v2'), resolve=True)
    assert 'gaussian_w2' not in str(cfg)
