"""Explicit source-weight ablation keeps old adapter/checkpoint contracts safe."""
import json

import numpy as np
import pytest

from gmm40.v7 import GMM40V7
from gmm40.validate_v7 import make_agent, tree_error


def small_agent(correction=True, clip=None):
    return GMM40V7(seed=3, batch=2, num_students=16, proposal_components=8,
                   teacher_resample_count=4, actor_samples=4, iterations=5,
                   hidden_dims=(8, 8), dual_hidden_dims=(8, 8),
                   source_importance_correction=correction,
                   actor_max_grad_norm=clip)


@pytest.mark.parametrize('clip', [None, 2., 10.])
def test_default_preserves_signature_and_correction_false_is_distinct(clip):
    old = small_agent(clip=clip)
    changed = small_agent(False, clip)
    old_signature = json.loads(old.settings_signature)
    changed_signature = json.loads(changed.settings_signature)
    assert 'source_importance_correction' not in old_signature
    assert 'source_importance_correction' not in old.settings
    assert changed_signature.pop('source_importance_correction') is False
    assert changed_signature == old_signature
    assert changed.settings['source_importance_correction'] is False
    assert changed.actor_max_grad_norm == clip
    assert tree_error(old.state.params, changed.state.params) == 0
    assert tree_error(old.state.opt_state, changed.state.opt_state) == 0
    assert tree_error(old.dual_state, changed.dual_state) == 0
    assert tree_error(old.key, changed.key) == 0


@pytest.mark.parametrize('invalid', ['false', 'true', None, 0, 1])
def test_non_boolean_switch_is_rejected(invalid):
    with pytest.raises(ValueError, match='source_importance_correction'):
        small_agent(invalid)


def test_make_agent_forwards_false_but_old_config_keeps_true():
    reference = small_agent()
    config = dict(reference.settings, seed=3, batch=2, hidden_dims=(8, 8),
                  potential_solver='persistent_dual', dual_hidden_dims=(8, 8))
    assert make_agent(config).settings_signature == reference.settings_signature
    config['source_importance_correction'] = False
    disabled = make_agent(config)
    assert disabled.source_importance_correction is False
    assert disabled.actor_max_grad_norm is None
    data = disabled.prepare()
    np.testing.assert_array_equal(data['source_importance'],
                                  np.ones_like(data['source_importance']))
    metrics = disabled.advance(1)
    assert metrics['actor_source_importance_used'] == 0.
    assert metrics['actor_source_importance_mean'] == 1.
    assert metrics['actor_source_importance_max'] == 1.
    assert metrics['actor_source_importance_ess_fraction'] == 1.
    assert metrics['actor_source_log_importance_max'] == 0.
    assert metrics['actor_source_importance_block_max'] == 1.
    assert metrics['actor_gradient_block_max'] == metrics['actor_gradient_norm']
    assert metrics['actor_parameter_step_block_max'] > 0.
    assert 'actor_gradient_clip_limit' not in metrics


def test_no_source_correction_resume_and_cross_objective_rejection(tmp_path):
    off = small_agent(False)
    off.advance(1)
    checkpoint = tmp_path / 'no-source-correction.bin'
    off.save(checkpoint)
    resumed = small_agent(False)
    resumed.restore(checkpoint)
    assert tree_error(off.checkpoint(), resumed.checkpoint()) == 0
    off.advance(1)
    resumed.advance(1)
    assert tree_error(off.checkpoint(), resumed.checkpoint()) == 0
    for wrong in (small_agent(True), small_agent(False, 2.)):
        with pytest.raises(ValueError, match='settings/seed mismatch'):
            wrong.restore(checkpoint)
        assert wrong.updates == int(wrong.state.step) == int(wrong.dual_state.step) == 0
    original_checkpoint = tmp_path / 'source-corrected.bin'
    original = small_agent(True)
    original.save(original_checkpoint)
    old_compatible = small_agent(True)
    old_compatible.restore(original_checkpoint)
    assert tree_error(original.checkpoint(), old_compatible.checkpoint()) == 0
    with pytest.raises(ValueError, match='settings/seed mismatch'):
        small_agent(False).restore(original_checkpoint)
