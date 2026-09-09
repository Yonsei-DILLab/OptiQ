"""Count oracle queries across resumed runs with different candidate counts."""
import json
from pathlib import Path


def density_evaluations(config, update):
    """Recover old run ancestry or use a new run's recorded starting count.

    Checkpoint steps, rather than filenames or requested final budgets, determine
    how much of each parent run was actually used. This is logging only.
    """
    update = int(update)
    if 'target_density_evaluations_at_start' in config:
        start = int(config['start_update'])
        previous = int(config['target_density_evaluations_at_start'])
    elif config.get('resume_checkpoint'):
        from flax.serialization import msgpack_restore

        checkpoint = Path(config['resume_checkpoint'])
        parent = json.loads((checkpoint.parent / 'config.json').read_text())
        start = int(msgpack_restore(checkpoint.read_bytes())['step'])
        previous = density_evaluations(parent, start)
    else:
        start, previous = 0, 0
    if start < 0 or previous < 0 or update < start:
        raise ValueError('Invalid update or parent oracle-query accounting')
    candidates = (config['batch_size'] * config['num_policy_samples']
                  * config['proposals_per_policy_sample'])
    if candidates <= 0:
        raise ValueError('Candidate count must be positive')
    return previous + (update - start) * candidates
