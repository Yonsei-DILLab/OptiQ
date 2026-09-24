"""Verify the actual DACER regulator receives the requested signed target."""
import json
import math
from pathlib import Path


def verify_target(learner, folder, expected):
    model=learner.model
    actual=float(model.regulator_cfg.target_entropy_per_dim)
    action_dim=int(model.action_space.shape[0])
    assert math.isfinite(expected) and actual==expected
    assert model.regulator_enabled and model.regulator_cfg.behavior_only
    assert learner.config['dacer']['target_entropy_per_dim']==expected
    assert action_dim==8
    preserved=dict(initial_alpha=.27,alpha_lr=.03,interval_updates=10000,
                   components=3,samples=200,noise_scale=.1,entropy_seed=42)
    for key,value in preserved.items():
        assert getattr(model.regulator_cfg,key)==value,(key,getattr(model.regulator_cfg,key))
    result=dict(verified=True,target_entropy_per_dim=actual,action_dim=action_dim,
                target_entropy=actual*action_dim,preserved=preserved,
                initial_noise_std=float(model.regulator_noise_std),
                behavior_only=True,source='live model.regulator_cfg; no state/RNG mutation')
    (Path(folder)/'dacer-target-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
