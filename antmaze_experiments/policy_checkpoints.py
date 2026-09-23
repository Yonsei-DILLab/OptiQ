"""Small evaluation checkpoints, independent of the final full-replay checkpoint."""
import hashlib
from pathlib import Path

import numpy as np


def save_evaluation_checkpoint(learner, folder, step, config):
    import flax.serialization
    import jax
    import torch
    from .run import write

    if learner.method != 'optiq':
        raise ValueError('Intermediate evaluation checkpoints currently support OptiQ only')
    destination = Path(folder) / 'policy-checkpoints' / f'step_{step:010d}'
    destination.mkdir(parents=True, exist_ok=False)
    path = destination / 'policy.pt'
    payload = dict(kind='evaluation-only', config=config, step=step,
                   updates=learner.updates, learner=learner.state(),
                   includes_replay=False, includes_simulator_state=False)
    temporary = path.with_suffix('.pt.tmp')
    torch.save(payload, temporary)
    loaded = torch.load(temporary, map_location='cpu', weights_only=False)
    assert loaded['step'] == step and loaded['updates'] == learner.updates
    assert loaded['config'] == config
    policy = learner.model.policy
    template = dict(actor=policy.actor_state, critic=policy.qf_state,
                    target_actor=policy.target_actor_state)
    restored = flax.serialization.from_bytes(template, loaded['learner']['policy'])
    original_leaves = jax.tree_util.tree_leaves(template)
    restored_leaves = jax.tree_util.tree_leaves(restored)
    assert len(original_leaves) == len(restored_leaves)
    for before, after in zip(original_leaves, restored_leaves):
        assert np.array_equal(np.asarray(before), np.asarray(after))
    assert loaded['learner']['entropy'] == payload['learner']['entropy']
    assert loaded['learner']['regulator'] == payload['learner']['regulator']
    for name in ('key', 'policy_key', 'noise_key'):
        assert np.array_equal(loaded['learner'][name], payload['learner'][name])
    temporary.replace(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    proof = dict(kind='evaluation-only', path=str(path.relative_to(folder)),
                 step=step, updates=learner.updates, source_commit=config['source_commit'],
                 sha256=digest, bytes=path.stat().st_size, readback_verified=True,
                 restored_policy_state_exact=True, includes_replay=False,
                 includes_simulator_state=False)
    write(destination / 'verification.json', proof)
    return proof
