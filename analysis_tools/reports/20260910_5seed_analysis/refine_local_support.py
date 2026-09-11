"""Exact Gaussian-mixture box masses and independently refined conditional Q.

No actor training. This corrects edge-alignment error in masking a global grid.
"""
from pathlib import Path
import json, csv, sys
import numpy as np
from scipy.special import ndtr, logsumexp
P = Path(__file__).resolve().parent
sys.path.insert(0, str(P))
from source.problems import make_problem, CASES

rows = []
for case in CASES:
    p = make_problem(case)
    if p.constant:
        continue
    b = p.base() if p.separable else p
    d = b.dim
    def boxmass(lo, hi):
        return float(np.sum(b.weights * np.prod(ndtr((hi-b.centers)/b.scales)-ndtr((lo-b.centers)/b.scales), axis=1)))
    full = boxmass(np.full(d, -1.), np.full(d, 1.))
    truth = p.truth()[2]
    for i, center in enumerate(b.centers):
        lo, hi = np.maximum(-1., center-.5), np.minimum(1., center+.5)
        bm = boxmass(lo, hi)
        history = []
        for n in [256, 512, 1024]:
            axes = [l+(np.arange(n)+.5)*(h-l)/n for l,h in zip(lo,hi)]
            grid = axes[0][:,None] if d == 1 else np.stack(np.meshgrid(*axes,indexing='ij'),-1).reshape(-1,d)
            q = b.q(grid)
            w = np.exp(q/.25-logsumexp(q/.25))
            value = float(w@q) * (p.dim if p.separable else 1)
            history.append({'n': n, 'conditional_q': value})
        assert abs(history[-1]['conditional_q']-history[-2]['conditional_q']) < 1e-4
        prob = (bm/full)**(p.dim if p.separable else 1)
        rows.append(dict(case=case, center=i, target_support_mass=prob,
                         conditional_truth=value, global_truth=truth, bias_limit=value-truth,
                         refinement_delta=abs(history[-1]['conditional_q']-history[-2]['conditional_q']), history=history))
(P/'local_support_refined.json').write_text(json.dumps(rows, indent=2))
keys = [k for k in rows[0] if k != 'history']
with (P/'local_support_refined.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)
print('boxes',len(rows),'maximum refinement delta',max(x['refinement_delta'] for x in rows))
for r in rows:
    if r['case'] in ['modes2d_8','separable_8']:
        print({k:v for k,v in r.items() if k!='history'})
