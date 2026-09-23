"""Identical sample metrics; forward teacher probe is explicitly counterfactual."""
from ..kl_forward_wide_1d.evaluate import evaluate as base_evaluate


def evaluate(exp, folder, step):
    metrics = base_evaluate(exp, folder, step)
    for key in list(metrics):
        if key.startswith('teacher_probe_'):
            metrics['counterfactual_forward_' + key] = metrics.pop(key)
    probe = folder / f'teacher_{step:05d}.npz'
    probe.rename(folder / f'counterfactual_forward_teacher_{step:05d}.npz')
    metrics['method'] = exp.method
    metrics['L'] = exp.L
    return metrics

