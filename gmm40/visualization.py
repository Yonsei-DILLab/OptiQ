"""Explicit visualization views; raw full-policy evaluations are never relabeled."""
import json
from pathlib import Path


def select_result(folder, result, view='primary'):
    if view not in ('primary','full_policy'):
        raise ValueError(f'Unknown visualization view: {view}')
    folder=Path(folder)
    directory=folder/'evaluations'/f"step_{result['step']:07d}"
    metrics=directory/'metrics_mu_only.json';samples=directory/'samples_mu_only.npy'
    config=folder/'config.json'
    method=json.loads(config.read_text()).get('method') if config.exists() else None
    if view=='primary' and method in ('optiq','optiq_trg') and not metrics.exists():
        raise FileNotFoundError(f'Missing required OptiQ mu-only metrics: {metrics}')
    use_mu=view=='primary' and metrics.exists()
    if use_mu:
        selected=json.loads(metrics.read_text())
        # Keep step, training timing and provenance from the corresponding full record.
        return dict(result,**selected,visualization_mode='mu_only'),samples
    return dict(result,visualization_mode='full_policy' if view=='full_policy' else 'native_policy'),directory/'samples.npy'


def select_history(folder, records, view='primary'):
    return [row if view=='primary' and row.get('visualization_mode')=='mu_only'
            else select_result(folder,row,view)[0] for row in records]


def mode_label(result):
    return {'mu_only':'μ only (no conditional σ noise)',
            'native_policy':'native generator output',
            'full_policy':'full stochastic policy'}[result['visualization_mode']]
