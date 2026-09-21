"""Check saved optimizer counters, independently of run status and log counters."""
import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import zipfile

import flax.serialization
import numpy as np
import torch

from .evaluation import atomic_json
from .target import RESULTS


def optax_counts(tree, prefix=''):
    counts = {}
    if isinstance(tree, dict):
        for key, value in tree.items():
            name = f'{prefix}/{key}'
            if key == 'count' and np.asarray(value).shape == ():
                counts[name] = int(value)
            else:
                counts.update(optax_counts(value, name))
    return counts


def torch_counts(optimizer):
    return {str(key): int(value['step']) for key, value in optimizer['state'].items()
            if 'step' in value}


def audit(folder):
    cfg = json.loads((folder/'config.json').read_text())
    status = json.loads((folder/'status.json').read_text())
    latest = json.loads((folder/'latest.json').read_text())
    navigation = cfg.get('Q') == 'learned, not oracle'
    original_budget = cfg['actor_updates'] if navigation else cfg['steps']
    stopped_early = status.get('status') == 'stopped_early'
    expected = status['saved_evaluated_updates'] if stopped_early else original_budget
    prefix = 'update' if navigation else 'step'
    extension = 'zip' if navigation and cfg['method'] == 'sac' else 'bin'
    path = folder/'checkpoints'/f'{prefix}_{expected:07d}.{extension}'
    groups = {}
    saved_updates = None

    if cfg['method']=='sql':
        state=flax.serialization.msgpack_restore(path.read_bytes())['state']
        saved_updates=int(state['updates'])
        for name in (['actor','critic'] if navigation else ['actor']):
            trainstate=state[name]
            groups[name]=dict(model_step=int(trainstate['step']),
                              optimizer_counts=optax_counts(trainstate['opt_state']))
    elif cfg['method'] in ('optiq', 'optiq_trg', 'mfpo'):
        state = flax.serialization.msgpack_restore(path.read_bytes())
        names = (['actor', 'critic'] if cfg['method'] == 'optiq'
                 else ['actor', 'logp_mvel', 'critic_1', 'critic_2', 'temp']) if navigation else (
                 ['state'] if cfg['method'] in ('optiq','optiq_trg') else ['state', 'divstate'])
        saved_updates = state.get('updates')
        for name in names:
            trainstate = state[name]
            groups[name] = dict(model_step=int(trainstate['step']),
                               optimizer_counts=optax_counts(trainstate['opt_state']))
    elif extension == 'zip':
        with zipfile.ZipFile(path) as archive:
            for name in ('actor.optimizer.pth', 'critic.optimizer.pth', 'ent_coef_optimizer.pth'):
                optimizer = torch.load(io.BytesIO(archive.read(name)), map_location='cpu', weights_only=False)
                groups[name] = dict(optimizer_counts=torch_counts(optimizer))
    else:
        state = torch.load(path, map_location='cpu', weights_only=False)
        saved_updates = int(state['updates'])
        names = ['actor_optimizer', 'critic_optimizer'] if navigation and cfg['method'] == 'dipo' else ['optimizer']
        for name in names:
            groups[name] = dict(optimizer_counts=torch_counts(state[name]))

    errors = []
    for name, group in groups.items():
        if group.get('model_step', expected) != expected:
            errors.append(f'{name}: model step {group["model_step"]} != {expected}')
        counts = group['optimizer_counts']
        if not counts or any(value != expected for value in counts.values()):
            errors.append(f'{name}: optimizer counts {sorted(set(counts.values()))} != {expected}')
        # Parameter IDs do not add value to the report; retain the checked group size.
        group['checked_optimizer_counter_count'] = len(counts)
        group['optimizer_step_values'] = sorted(set(counts.values()))
        del group['optimizer_counts']
    if saved_updates is not None and int(saved_updates) != expected:
        errors.append(f'saved update count {saved_updates} != {expected}')
    if status.get('status') not in ('completed', 'stopped_early'):
        errors.append('run status is not completed')
    if stopped_early and (not status.get('early_stop_authorized') or not status.get('process_exit_verified')):
        errors.append('early-stop authorization or process exit verification is missing')
    if latest.get('updates', latest.get('step')) != expected:
        errors.append('final evaluation does not match requested budget')
    if navigation and status.get('env_steps') != expected + cfg['warmup']:
        errors.append('environment step count does not include the requested warmup')
    result = dict(timestamp=datetime.now(timezone.utc).isoformat(),
                  status='passed' if not errors else 'failed', checkpoint=str(path),
                  actor_updates=expected, saved_updates=saved_updates, groups=groups,
                  original_budget=original_budget, run_status=status.get('status'),
                  full_budget_completed=not stopped_early,
                  last_observed_updates=status.get('last_observed_updates',expected),
                  final_evaluation_step=latest.get('updates', latest.get('step')), errors=errors)
    atomic_json(folder/'update_count_audit.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', action='append', help='Audit specified completed runs, including validation runs.')
    args = parser.parse_args()
    if args.name:
        names = args.name
    else:
        jobs = json.loads((RESULTS/'queue.json').read_text())['jobs']
        names = [job['name'] for job in jobs if (RESULTS/job['name']/'status.json').exists()
                 and json.loads((RESULTS/job['name']/'status.json').read_text()).get('status') in ('completed','stopped_early')]
    results = {name: audit(RESULTS/name) for name in names}
    print(json.dumps(results, indent=2))
    if any(result['errors'] for result in results.values()):
        raise SystemExit('Saved optimizer count audit failed.')


if __name__ == '__main__':
    main()
