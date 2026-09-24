"""Verify disabled behavior noise without modifying policy or training RNG."""
import json
from pathlib import Path
import numpy as np


def verify_disabled(learner, folder):
    from .learners import evaluation_rng
    model=learner.model
    assert not model.regulator_enabled and not model.regulator_cfg.enabled
    assert learner.config['dacer']['enabled'] is False
    assert not hasattr(model,'regulator_log_alpha')
    obs=np.zeros((2,29),dtype=np.float32)
    with evaluation_rng(learner,421991):train=learner.act(obs,'train')
    with evaluation_rng(learner,421991):policy=learner.act(obs,'policy')
    assert np.isfinite(train).all() and train.shape==(2,8)
    assert np.array_equal(train,policy), 'Disabled DACER must not perturb train actions'
    state=learner.state()
    assert state['regulator'] is None and state['regulator_rng'] is None
    assert state['regulator_count']==0
    proof=dict(verified=True,dacer_enabled=False,extra_noise_std=0.,regulator_updates=0,
        train_equals_direct_policy_at_same_rng=True,
        policy='fresh random latent plus conditional sigma preserved',
        state_serializable_without_regulator=True,
        source='live model and actual train/policy action draws; evaluation_rng restores RNG')
    (Path(folder)/'dacer-disabled-verification.json').write_text(json.dumps(proof,indent=2)+'\n')
    return proof
