"""Post-hoc evaluation only; original samples and training metrics are untouched."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist, pdist

ROOT = Path(__file__).resolve().parent
y_all = np.load(ROOT / 'reference.npy').astype(np.float64)
median_h = float(np.median(pdist(y_all[:2048])))
hs = np.array([1., 2., 5., 10., 20., median_h])

def sums(a, b):
    result = np.zeros(len(hs))
    for start in range(0, len(a), 256):
        d = cdist(a[start:start+256], b, 'sqeuclidean')
        for j, h in enumerate(hs):
            result[j] += np.exp(-d/(2*h*h)).sum()
    return result

rows = []
for n in (2048, 4096, 10000):
    y = y_all[:n]
    yy = sums(y, y)
    for seed in range(5):
        x = np.load(ROOT / f'seed-{seed}/final/samples_full_policy.npy')[:n].astype(np.float64)
        xx, xy = sums(x, x), sums(x, y)
        u2 = (xx-n)/(n*(n-1)) + (yy-n)/(n*(n-1)) - 2*xy/(n*n)
        v2 = (xx+yy-2*xy)/(n*n)
        row = dict(n=n, seed=seed,
                   multikernel_u_mmd=float(np.sqrt(max(u2[:5].mean(), 0))),
                   multikernel_u_mmd2=float(u2[:5].mean()),
                   multikernel_v_mmd=float(np.sqrt(max(v2[:5].mean(), 0))),
                   single_bandwidth_u_mmd={str(h):float(np.sqrt(max(v, 0))) for h,v in zip(hs,u2)})
        if n == 2048:
            original = json.loads((ROOT / f'seed-{seed}/final/metrics_full_policy.json').read_text())
            assert np.isclose(row['multikernel_u_mmd'], original['mmd'], atol=1e-12, rtol=0)
        rows.append(row)
    print(json.dumps(dict(n=n, completed=True)), flush=True)

summary = {}
for n in (2048, 4096, 10000):
    sub = [r for r in rows if r['n']==n]
    summary[n] = {}
    values = {key: [r[key] for r in sub] for key in ('multikernel_u_mmd','multikernel_u_mmd2','multikernel_v_mmd')}
    values.update({'h='+str(h):[r['single_bandwidth_u_mmd'][str(h)] for r in sub] for h in hs})
    for key,v in values.items():
        summary[n][key] = dict(mean=float(np.mean(v)), seed_sd=float(np.std(v, ddof=1)))
report = dict(description='Saved full-policy samples, seeds 0-4. Prefix subsets; same bounded reference shared across runs. U means unbiased MMD-squared estimator before square root, V includes diagonals. SD is across training seeds, not evaluation uncertainty.',
              kernel='exp(-squared_distance/(2*h*h)) in physical coordinates',
              bandwidths=hs.tolist(), median_bandwidth_rule='median pairwise Euclidean distance of first 2048 reference samples only',
              rows=rows, summary=summary)
(ROOT/'mmd-sensitivity.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(summary, indent=2), flush=True)
