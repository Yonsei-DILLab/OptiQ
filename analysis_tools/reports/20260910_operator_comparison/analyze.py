"""Read-only post-hoc comparison of completed experiments and operator means."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / '20260910_5seed_analysis'
spec = importlib.util.spec_from_file_location('frozen_problems', BASE / 'source/problems.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def numerical_maximum(problem):
    """Dense bounded grid followed by multistart continuous refinement."""
    p = problem.base() if problem.separable else problem
    x = np.linspace(-1, 1, 32769 if p.dim == 1 else 513)
    grid = x[:, None] if p.dim == 1 else np.stack(np.meshgrid(x, x, indexing='ij'), -1).reshape(-1, 2)
    q = p.q(grid)
    starts = np.concatenate([p.centers, grid[np.argsort(q)[-8:]]])
    results = [minimize(lambda a: -float(p.q(np.asarray(a))), a,
                        bounds=[(-1., 1.)] * p.dim, method='L-BFGS-B',
                        options={'ftol': 1e-14, 'gtol': 1e-10}) for a in starts]
    maximum = max(float(q.max()), *[-float(r.fun) for r in results])
    multiplier = problem.dim if problem.separable else 1
    return maximum * multiplier, (maximum - float(q.max())) * multiplier


frozen = []
for case in ['unimodal', 'asymmetric_1.3_0.08', 'modes2d_4', 'modes2d_8', 'separable_4', 'separable_8']:
    problem = module.make_problem(case)
    maximum, refinement = numerical_maximum(problem)
    rows = []
    for seed in range(5):
        with (BASE / 'runs' / f'frozen_{case}_default_seed{seed}' / 'estimates.csv').open() as f:
            row = next(r for r in csv.DictReader(f) if int(r['step']) == 20000 and int(r['k']) == 50 and r['method'] == 'optiq_raw')
        rows.append({k: float(row[k]) for k in ['truth', 'mean', 'bias', 'rmse']})
    truth = float(np.mean([r['truth'] for r in rows]))
    actor = float(np.mean([r['mean'] for r in rows]))
    frozen.append(dict(case=case, seeds=5, numerical_max=maximum, max_refinement=refinement,
                       boltzmann=truth, actor_mean=actor, actor_bias=actor-truth,
                       max_minus_boltzmann=maximum-truth,
                       fraction_of_max_boltzmann_gap_retained=(maximum-actor)/(maximum-truth)))

categorical = json.loads((HERE / 'categorical_summary.json').read_text())
assert categorical['completed'] == categorical['expected'] == 60
pairs = [r for r in categorical['paired_estimates'] if r['method'] == 'optiq_raw' and r['k'] == 50]
history = categorical['mode_history']
groups = []
for case, init in sorted({(r['case'], r['initialization']) for r in pairs}):
    rows = [r for r in pairs if (r['case'], r['initialization']) == (case, init)]
    assert {r['seed'] for r in rows} == set(range(5))
    g = dict(case=case, initialization=init, n=5,
             categorical_rmse_wins=sum(r['rmse_difference'] < 0 for r in rows))
    for v in ['argmax', 'categorical']:
        stats = {k: float(np.mean([r[v + '_' + k] for r in rows])) for k in ['mean', 'bias', 'rmse']}
        stats['distribution'] = {}
        for step in [0, 100, 20000]:
            h = [r for r in history if (r['case'], r['initialization'], r['variant'], r['step']) == (case, init, v, step)]
            assert len(h) == 5
            stats['distribution'][str(step)] = {k: float(np.mean([r[k] for r in h])) for k in ['bin_tv', 'effective_modes', 'coord_std']}
            stats['distribution'][str(step)]['one_effective_mode_seeds'] = sum(r['effective_modes'] <= 1.000001 for r in h)
        g[v] = stats
    groups.append(g)

movecar = []
for method in ['ddpg', 'sd2', 'td3', 'sd3', 'optiq']:
    rows = [json.loads((BASE / 'runs' / f'movecar_{method}_seed{s}' / 'learning.json').read_text())[-1] for s in range(5)]
    assert all(r['step'] == 1000000 for r in rows)
    movecar.append(dict(method=method, seeds=5, bias_by_seed=[r['bias'] for r in rows],
                       **{k: float(np.mean([r[k] for r in rows])) for k in ['q_mean', 'true_mean', 'bias', 'return_mean']}))

result = {'description': 'Frozen: default initialization, original argmax, 5 seeds, 20k updates, K=50. Max is numerical; the gap fraction is a scalar-value diagnostic, not density accuracy or true-return overestimation. MoveCar: final 1M-step online twin-mean Q versus the same first action followed by the method-specific frozen behavior policy, discounted continuing rollout horizon 2048. This differs from the target/noisy backup policy and is not a matched-operator causal comparison.',
          'frozen': frozen, 'categorical_completed': 60, 'categorical': groups, 'movecar': movecar}
(HERE / 'analysis.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
