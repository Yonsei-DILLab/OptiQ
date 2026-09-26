import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
import train

@pytest.mark.parametrize('task,t,c', [
    ('hopper',.05,.1),('walker2d',.1,.1),('halfcheetah',.25,.15),
    ('ant',.25,.1),('humanoid',.1,.15)])
def test_defaults(task,t,c):
    cfg = train.compose_config([f'benchmark={task}'])
    train.validate_config(cfg)
    assert cfg.alg.actor.temperature == t
    assert cfg.dacer.noise_scale == c and cfg.dacer.enabled
    assert list(cfg.alg.actor.hidden_dims) == list(cfg.alg.critic.hs) == [256,256]
    assert cfg.alg.actor.num_policy_samples == 64
    assert cfg.alg.actor.proposals_per_policy_sample == 1
    assert cfg.alg.actor.log_std_min == -5 and cfg.alg.actor.log_std_max == -1
    assert cfg.alg.actor.initial_log_std == -1 and cfg.fixed_log_std is None
    assert cfg.dual_mu_eval and cfg.mu_only_eval
    assert cfg.wandb.entity is None and cfg.wandb.mode == 'offline'

def test_overrides():
    cfg = train.compose_config(['benchmark=ant','temperature=0.5','dacer.enabled=false',
                               'fixed_log_std=-3','alg.actor.num_policy_samples=16'])
    train.validate_config(cfg)
    assert cfg.alg.actor.temperature == .5 and not cfg.dacer.enabled
    assert cfg.fixed_log_std == -3 and cfg.alg.actor.num_policy_samples == 16
