"""Oracle-query accounting must not rewrite parent compute after count tuning."""
import json

import flax.serialization
import numpy as np
import pytest

from benchmarks.gmm40.accounting import density_evaluations


def _checkpoint(path, config, step):
    path.mkdir()
    (path / 'config.json').write_text(json.dumps(config))
    checkpoint = path / 'actor_state_final.msgpack'
    checkpoint.write_bytes(flax.serialization.msgpack_serialize({'step': np.array(step)}))
    return str(checkpoint)


def test_resumed_counts_use_actual_checkpoint_step_and_parent_counts(tmp_path):
    base = dict(batch_size=1, num_policy_samples=512, proposals_per_policy_sample=4,
                updates=50000)
    first = _checkpoint(tmp_path / 'first', base, 12345)
    second_config = dict(base, num_policy_samples=4096, proposals_per_policy_sample=1,
                         resume_checkpoint=first)
    second = _checkpoint(tmp_path / 'second', second_config, 15000)
    third = dict(base, num_policy_samples=2048, proposals_per_policy_sample=2,
                 batch_size=2, resume_checkpoint=second)
    expected = 12345 * 2048 + (15000 - 12345) * 4096 + 500 * 8192
    assert density_evaluations(third, 15500) == expected
    assert density_evaluations(base, 50000) == 50000 * 2048


def test_recorded_start_count_does_not_need_parent_files():
    cfg = dict(batch_size=1, num_policy_samples=4096, proposals_per_policy_sample=1,
               start_update=200000, target_density_evaluations_at_start=409600000,
               resume_checkpoint='/unavailable/old-checkpoint.msgpack')
    assert density_evaluations(cfg, 210000) == 450560000
    assert density_evaluations(cfg, 200000) == 409600000
    with pytest.raises(ValueError):
        density_evaluations(cfg, 199999)


def test_same_candidate_resume_matches_original_count(tmp_path):
    cfg = dict(batch_size=1, num_policy_samples=2048, proposals_per_policy_sample=1)
    cp = _checkpoint(tmp_path / 'parent', cfg, 50000)
    assert density_evaluations(dict(cfg, resume_checkpoint=cp), 75000) == 75000 * 2048
